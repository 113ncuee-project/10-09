> 更新：使用者最新的16nm成本基準與strict/soft限制修正，請先看 [hybrid_cost16_sources.md](hybrid_cost16_sources.md) 與 [hybrid_cost16_progress.md](hybrid_cost16_progress.md)。本檔下方數值屬歷史版本；舊混合製程／MAC乘8推導已撤回，不可當成新成本。

# 多模型、通訊感知 mapping 與可操作 GUI

日期：2026-10-09 Asia/Taipei。依使用者10/08新增要求，以原10/06研究原型為基礎增修；舊benchmark資料保留，不混用source identities。

## 1. 硬體契約與圖形輸出

沿用INDM的「compute leaf → IO die → IO mesh → IO die → compute leaf」，每cluster至多4個compute dies、一個DRAM endpoint。跨die使用RapidChiplet原生placement、PHY/bump bandwidth ceiling、XY路由與path latency；片內則以实际單向L2→PE→L2 multi-ring，或XY mesh作對照。PE具16 lanes×16 INT8 vector MAC，INT24 partial sum在OL1累積，不跨die切C。

4/8/16 compute-die候選維持相同總256PE、64mapping cores、65.536TMAC/s、75GB/s DRAM與8GiB DRAM容量。每die數量如下；NoP/IO/controller/CPU/PHY複製、線長、共享buffer服務率各自付出成本。

| Compute dies | IO/DRAM | Cores/die | PEs/die | Rings/die | Reference UB/die（總128MiB） |
|---|---|---|---|---|---|
| 4 | 1 / 1 | 16 | 64 | 16 | 32MiB |
| 8 | 2 / 2 | 8 | 32 | 8 | 16MiB |
| 16 | 4 / 4 | 4 | 16 | 4 | 8MiB |

GUI提供總UB32/64/128/256MiB；上表是多模型驗證用128MiB配置，不冒稱仍為舊32MiB iso-total budget。L1 W/A/O為64/16/8KiB/PE/slot；reference16-die的L2 W/A/O為64/16/8KiB/die/slot，粗粒度die按資源守恆放大。SRAM容量不等於服務頻寬；NoC8.5GB/s/link、NoP12.5GB/s/direction/link、L1每PE/方向64GB/s、die L2/UB各64GB/s，保留controller與IO交叉開關限制。

雙緩衝 `prefetch_slots=2` **複製L1/L2 scratch banks**，SRAM面積、die尺寸、leakage與native placement同步重算。UB、NoC、NoP、controller和共享服務port不加倍。每個宏unit仍保留完整UB backing，讀下一unit可與目前compute重疊；core compute依序，bank只能在前兩個unit的drain完成後重用。容量統計包含read開始至write結束的concurrent scratch，不僅檢查單tile。

每次輸出 `package.svg`（實際mm幾何）、`die.svg`（實際PE/ring數）、`pe.svg`（實際lane/vector與buffer）、`architecture.svg`（三級架構圖）、`layout.json`／`layout.html`（點選die與切layer）。數量和mapping從此次evaluation導出，不套用固定16-chiplet圖。Package位置是評估輸入；die/PE inset是資源示意，沒有宣稱完成ASIC floorplan或place-and-route。

## 2. 多模型 frontend

`models.py` 使用torchvision `weights=None`、torch.fx及meta tensor shape propagation抽取真實operator DAG，不下載權重、不執行巨型VGG權重配置。支援ResNet18/34/50、VGG16/19、AlexNet、MobileNetV2；GUI另列toy供快速驗證。

Conv的kernel/stride/padding/dilation/groups、depthwise分組、bias與MAC數均從實際module和shape取得。保留Add residual、真實pool語意、Flatten、Linear及Dropout的inference identity。MobileNetV2 ReLU6計兩個vector comparisons；單consumer BN/ReLU可以與前一Conv/Linear/Add fusion。來源、input size、版本與DAG fingerprint保存在workload.json。

| 模型 | Fusion後layers | 每sample MAC（224×224） |
|---|---|---|
| ResNet18 | 32 | 1,814,073,344 |
| ResNet34 | 56 | 3,663,761,408 |
| ResNet50 | 73 | 4,089,184,256 |
| VGG16 | 25 | 15,470,264,320 |
| VGG19 | 28 | 19,632,062,464 |
| AlexNet | 15 | 714,188,480 |
| MobileNetV2 | 66 | 300,774,272 |

以上是operator形狀導出的數量，不是PPA背誦表。硬體儲存/運算用INT8、psum INT24；沒有PTQ、分類準確率或BFP/FP16校準結論。Flatten反向區域現在把完整plane／row合併，保留精確元素覆蓋，降低VGG FC frontend的冗餘box。

## 3. 筆記第1、2、4點：whole-DAG layer switching

`traffic_planner.py`以balanced allocation提供每group候選：output Part(B,K,H,W)、compact/striped core correspondence、core rotation、近端DRAM。Striped placement能把大型FC權重分散到多個die，而不是將前一layer全塞進少數die。

跨layer來源/需求由精確shard矩形、Conv halo、groups、pool、Flatten語意導出；同一region的multicast按directed-edge union計算，跨group依現有契約materialize到DRAM。K-sharded producer對dense下一Conv的C需求自然形成gather與再分配，沒有「免費all-gather」捷徑。輸出 `layer_switching.json` 記錄每条真實DAG edge的Part、logical demand、injection/delivery/NoP hop bytes及unicast/multicast/gather/boundary分類；共享transfer的edge歸因列不能直接相加。

規劃器保留所有尚有future consumers的producer mappings，包含非相鄰residual邊。Chain可化簡成O(G×M²)DP；一般DAG採bounded live-frontier beam（預設每group8states、beam8），不宣稱用pairwise DP就能精確求解所有DAG。累加的是proposal delay proxy與資源壓力，不相加local EDP。最後數個候選由完整batch oracle排序，檢查共享鏈路、DRAM競爭、buffer lifetimes、fill/drain，保留可行且目標較佳的plan。

RL維持Gemini五operator：Part變更、同layer core ordering swap、同group跨layer swap、core reallocation、DRAM source/interleave變更。Actor/critic以實際PPA回饋更新；cluster-guided proposals與既有公平baselines仍可使用。本文沒有跨模型預訓練或RL普遍優於greedy的主張。

## 4. 筆記第5、7點：片內 reuse 與 prefetch

片內是獨立選擇，不把activation reuse等同於跨die IC parallel。三種loop traversal為B,K,H,W,C（output_stationary）、B,H,W,K,C（activation_reuse）、B,K,W,H,C（weight_reuse）。所有C partial sum仍在同一output microtile的OL1累積。`microtile_shape`與C chunk也可選，named SRAM bank容量仍由內部tiling檢查；變更順序導出的cache miss、SRAM bits、NoC注入與整數SIMD cycles是真實計算，不使用人工OC/IC效率因子。

`dataflow=auto`以每layer最大/尾端兩個代表unit、六組loop/tile/C候選作bounded local proposals，再把改寫後整個plan交給full oracle；只有全域目標不差且合法時才接受。它不是窮舉所有loop permutation的最佳dataflow，更沒有免費partial-sum spill。硬體不支持的spill／精度或IC reduction不會拿來兌現收益。

預取是在宏read/compute/write層級，有明示兩個banks及dependency，保留fill/drain。不含fine-grained逐packet/microkernel streaming overlap。全模型仍需核對UB容量；多buffer並不保證改善DRAM-bound workload。

## 5. 四種偏好與多模型架構共探

Gemini V-A原目標是MC^α×E^β×D^γ，模型E/D用幾何平均。GUI採E^wE×A^wA×D^wD：E為whole-batch energy（J），A為total silicon area（mm²），D為whole-batch latency（s）。Area是MC的明示代理，沒有美元／良率模型。Power（平均W）另外顯示並保留硬性上限。低功耗偏好以降低推論能量實現；若直接用P=E/D與D同等相乘，均衡模式的延遲項會抵消，因此未採用此不適合本目標的公式。

| 偏好 | Energy | Area | Latency |
|---|---|---|---|
| 均衡 | 1/3 | 1/3 | 1/3 |
| 低功耗 | 0.8 | 0.1 | 0.1 |
| 小面積 | 0.1 | 0.8 | 0.1 |
| 低延遲 | 0.1 | 0.1 | 0.8 |

每個hardware candidate必須評估全部選定模型；用各模型相同的固定共同單位作reference後取score幾何平均。只要某個模型容量/上限失敗，該硬體不能成為共同最佳。內層RL可以固定自己的initial scales做穩定reward，但其相對improvement不拿來跨hardware比較。四种模式可能選到同一硬體，不能強制製造不同答案。

GUI的Power/Area/Latency上限以checkbox啟用，未勾選不施加限制，不以bonus/penalty取代feasibility。每個trial失敗獨立保存，其他候選仍可完成；完全沒有共同可行候選時呈現原因。結果包含平均power、energy、latency及silicon area，使用者可點開實際layout。

## 6. 可讀來源與精確限制

已讀使用者Word與補充PDF第1/2/4/5點，並視檢Word內兩張筆記/書頁圖；Gemini V-A及之前已核對的INDM硬體/communication方法。分享聊天公開頁可讀文字摘要，但16張圖片只顯示uploaded placeholders，匿名瀏覽器img=0；沒有取得原圖。依可讀摘要採用buffer hierarchy、partial-sum locality、FC bandwidth、ping-pong與input/output reuse原則，未冒稱逐張原圖精讀。書中BFP、DDR3與FPGA MMAC參數不直接混入此INT8 ASIC profile。

PPA仍是未經RTL/silicon校準的分析估算；NoP及geometry使用原生RapidChiplet，compute/NoC/SRAM/DRAM動態服務由本專案計算。部分係數為混合論文/原始碼近似；post-op cycles/SRAM accesses有計入，但獨立非MAC ALU的silicon與dynamic energy沒有單獨校準。多port/control overhead、bank arbitration、packet VC、實際placement routing/yield與thermal分析是後續硬體驗證工作，不能宣稱已完成物理晶片。
