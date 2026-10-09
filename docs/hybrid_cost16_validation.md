# Cost16 基準、限制與驗證結果

本報告以真正保存的 hardware.json / mapping.json 為準。舊混合製程 PPA 保留作歷史資料，不覆寫也不與此source混用。此版本移除unsupported MAC乘8與Gemini12nm SRAM/MAC密度；維持INDM硬體與Gemini/RL方法。

## 實際實驗設定

五個full224模型：ResNet18 / ResNet50 / VGG16 / MobileNetV2 / AlexNet。Batch1、seed7、traffic_dp+auto dataflow、RL4 unique mapping queries per trial（另有planner評估）；這是重現驗證的小搜尋預算，不是全域最優或RL優勝證據。
共用資源：256 PEs、64 mapping cores、每PE16 lanes x16 vector MAC、INT8/INT24、compute500MHz，共65.536 TOPS（每scalar MAC報2ops）。總UB128MiB，L1/L2雙緩衝。DRAM總容量8GiB、總BW75GB/s。IO clusters依die數為1/2/4，保持4compute/cluster。

| Compute dies | IO / memory dies | PE / die | UB / die | 總 SRAM 面積估算 mm² | GRS whole-footprint mm² | 總矽面積 mm² |
|---|---|---|---|---|---|---|
| 4 | 1 / 1 | 64 | 32MiB | 117.357 | 6.201 | 256.748 |
| 8 | 2 / 2 | 32 | 16MiB | 117.549 | 13.953 | 265.384 |
| 16 | 4 / 4 | 16 | 8MiB | 117.933 | 31.007 | 284.238 |

UB不隨雙緩衝再翻倍。新估算下總UB約77.916mm²（包含實體bank周邊）；舊flat density下103.51mm²只是歷史模型算式，兩者均非foundry認證數值。L1/L2額外bank周邊、MAC與PHY仍會影響不同die數的成本。

## 15組重新產生的PPA

| 架構 | 模型 | Latency ms | Energy mJ | 平均功率 W | Silicon mm² |
|---|---|---|---|---|---|
| dies4_multi_ring | resnet18 | 2.8607 | 24.700 | 8.634 | 256.748 |
| dies4_multi_ring | resnet50 | 8.5310 | 65.343 | 7.659 | 256.748 |
| dies4_multi_ring | vgg16 | 23.8365 | 200.811 | 8.424 | 256.748 |
| dies4_multi_ring | mobilenet_v2 | 2.2409 | 11.894 | 5.308 | 256.748 |
| dies4_multi_ring | alexnet | 6.8850 | 37.037 | 5.379 | 256.748 |
| dies8_multi_ring | resnet18 | 2.3535 | 23.185 | 9.851 | 265.384 |
| dies8_multi_ring | resnet50 | 6.4765 | 58.664 | 9.058 | 265.384 |
| dies8_multi_ring | vgg16 | 19.2442 | 187.402 | 9.738 | 265.384 |
| dies8_multi_ring | mobilenet_v2 | 1.9765 | 11.215 | 5.674 | 265.384 |
| dies8_multi_ring | alexnet | 5.8819 | 34.279 | 5.828 | 265.384 |
| dies16_multi_ring | resnet18 | 2.0285 | 22.643 | 11.162 | 284.238 |
| dies16_multi_ring | resnet50 | 5.2786 | 56.119 | 10.631 | 284.238 |
| dies16_multi_ring | vgg16 | 17.0237 | 184.867 | 10.859 | 284.238 |
| dies16_multi_ring | mobilenet_v2 | 1.8699 | 11.438 | 6.117 | 284.238 |
| dies16_multi_ring | alexnet | 5.3608 | 34.155 | 6.371 | 284.238 |

均衡 E/A/D 幾何平均分數（共同1J/1mm²/1s參考，越小越好）：dies4_multi_ring=0.407707, dies8_multi_ring=0.376898, dies16_multi_ring=0.368293.
未額外設上限時，此次候選／預算的推薦是 **dies16_multi_ring**。限制會改變可選候選，不能說它永遠最佳。

## 成本敏感度與限制可行性

OAT能量或leakage情境使用fixed-schedule component ledger精確重算；不冒稱每筆都重新跑physical oracle。SRAM area、1GHz及combined high情境完整重跑native placement/latency、microkernel與buffer schedule。所有情境另存actual hardware.json與baseline mapping SHA。範圍是探索假設，不是信賴區間；沒有重新搜尋，所以不能證明robust optimality。
範例限制為16W / 800mm² / 500ms；邊界probe為16W / 260mm² / 40ms，純粹用來檢查限制翻轉，不是使用者此次輸入。

| 情境 | 無上限推薦 | 範例限制 compliant trials /15 | 範例推薦 | 邊界限制 compliant trials /15 | 邊界推薦／狀態 |
|---|---|---|---|---|---|
| 基準 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| MAC 2 op/MAC | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| SRAM能量 x0.5 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| SRAM能量 x1.5 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| 漏電 x0.5 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| 漏電 x2 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| 片內互連能量 x0.5 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| 片內互連能量 x1.5 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| GRS能量 x0.8 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| GRS能量 x1.2 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| DRAM能量 x0.8 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| DRAM能量 x1.2 | 16 dies | 15 | 16 dies / 估算符合 | 5 | 4 dies / 估算符合 |
| SRAM面積 x0.75 | 16 dies | 15 | 16 dies / 估算符合 | 15 | 16 dies / 估算符合 |
| SRAM面積 x1.25 | 16 dies | 15 | 16 dies / 估算符合 | 0 | 4 dies / 最接近；未符合嚴格上限 |
| 1GHz（時序未驗證） | 16 dies | 15 | 16 dies / 估算符合 | 10 | 8 dies / 估算符合 |
| 成本假設合併提高 | 16 dies | 8 | 4 dies / 最接近；未符合嚴格上限 | 0 | 4 dies / 最接近；未符合嚴格上限 |

邊界probe相對nominal的推薦或compliance狀態改變：sram_area_075, sram_area_125, clock_1GHz_unverified, assumptions_high。實際差距請見sensitivity.json；即使無上限排名未改變，不能將成本不確定性忽略。

## 驗證證據與GUI

166 tests PASS，其中92個hybrid tests。15/15 full-model trials通過独立saved-artifact audit：source/native hashes、MAC conservation、energy component/event ledger、calendar/dependencies、UB/scratch capacity、actual die/PE count與SVG XML。
GUI Edge headless實測三情境：strict不可能上限仍有closest布局；取消strict保留soft target與gap；可達上限輸出estimated_compliant。四個偏好按鈕及strict config傳遞已檢查，沒有page JS error。

另以16-die ResNet18的完整physical oracle抽查mac_2ops / sram_energy_150 / leakage_200三個係數情境，latency、energy、area、power與全部energy components均與fixed-ledger重算相符（relative tolerance 1e-12）。

GUI：http://127.0.0.1:8765，載入完整工作ID **c16100000001**。啟動：`./simplified_rapidchiplet/scripts/start_hybrid_gui.ps1`。新預設結果根目錄為results/hybrid_gui_cost16。
重跑：`.venv/Scripts/python.exe tools/validate_hybrid_cost16.py --worker 4`（8/16同理），全部完成後 `--merge`；中斷用 `--resume`，source/settings/hardware不一致會拒絕reuse。敏感度工具：`tools/sensitivity_hybrid_cost16.py --scenario <name>`，四個full情境完成後 `--merge`。
新建GUI工作將以使用者數值與strict設定進行搜尋。找不到compliant時明示closest與gap；物理容量失敗的候選不會被fallback推薦。平均Power不是peak/TDP，silicon area不是package area，有限搜尋也不是不存在可行解的證明。

## 尚未校準／未完成硬體signoff

MAC reported op=scalar MAC仍是明示convention；原作者沒有提供可排除2op計數的定義。SRAM width/ports/read-write/area模型、DDR與memory die、router/CPU、NoC/crossbar、leakage等仍依來源notes標為anchor或假設。GRS只建立whole-macro accounting與array-fit條件，尚未完成electrical／thermal／timing驗證。所有方案提供的是nominal analytical estimate；不能宣稱達成使用者真實silicon PPA上限。
