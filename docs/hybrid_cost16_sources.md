# Hardware cost16 v1 - 來源、單位與工作條件

此版本是 **16 nm 文獻參考估算**，不是單一 foundry 的校準硬體。它保留 INDM cluster mesh / multi-ring、Gemini layer pipeline / mapping operators / RL 方法，將製程成本另行版本化。移除同一 compute die 使用 Gemini 12 nm MAC/SRAM 密度與 INDM 16 nm 能量的混合。未使用 (12/16)^2 製程縮放。

## MAC：撤回乘 8 的推導

NN-Baton Sec. V-A 報告 8-bit MAC 面積 135.1 um²、能量 0.024 pJ/op、500 MHz；模型由 UMC 28 nm 綜合縮放至 16 nm。它不是 TSMC 16 nm 實測值。INDM Table III 的 pJ/bit 表頭不足以支持再乘 8。

- 程式 `unit.macs` 計算 scalar MAC；TOPS 另以 2 operations / MAC 報告，不能把 TOPS 的 convention 自動套到原作者的 energy/op。
- 採用 convention：一個 reported op 對應一個 scalar MAC，因此 `mac_energy_pj=.024`、`mac_ops_per_mac=1`。這是對正文「8-bit MAC 能量」的推論，不是作者明確定義已經確認。敏感度另測 2 reported ops / MAC，即 .048 pJ/MAC。
- 每個 scalar MAC 面積採 135.1e-6 mm²。此次預設 compute clock **500 MHz**，NoC/NoP router clocks 是另外的未校準設定。
- 原始 MAC 電壓未知。1 GHz 情境不自動乘二能量；只在相同電壓／每操作能量假設下重算 latency 與必要 banks，並標明 timing 尚未驗證。不同電壓應使用另行取得的成本資料。

原始來源：[NN-Baton ISCA 2021](https://www.zhanhongtan.com/publication/isca21/isca21.pdf)，Table I、Fig.10、Sec.V-A。

## SRAM：每個實體 bank 的成本

NN-Baton Table I 的參考：1 KiB L1 .30 pJ/bit、32 KiB L2 .81 pJ/bit。它沒有提供這兩個能量點的完整 width/ports/讀寫 macro 表；Fig.10 的趨勢也不能視為能精確重建 Table I 的曲線。

新版採以下 **明示假設**，不將它們冒稱 memory compiler 結果：

| 參數 | v1 設定與涵蓋範圍 |
|---|---|
| 實體 bank 容量 | 上限 32 KiB；先滿足容量，再滿足要求頻寬；容量向 word 對齊 |
| word / ports | 128 bit、1 RW port；與原論文 anchors 的 width/ports 對應未知 |
| bank 數 | max(ceil(capacity/32KiB), ceil(BW/(wordBytes*computeClock*ports))) |
| 每 bit read energy | 1 KiB .30 與 32 KiB .81 之間按 log2(bank capacity) 插值；低於1KiB外插但最少 .01；屬近似 |
| width / port energy | (width/128)^.15；每增加 RW port 乘數增加 .30；屬假設 |
| write energy | read energy * `sram_write_energy_factor`，v1=1；讀寫可分別調整 |
| bank 面積 | allocatedBytes*.55e-6 mm² + 每bank .001 mm² * (width/128)*ports |
| 面積來源品質 | .55 um²/byte 與1000 um²周邊僅粗略參考Fig.10趨勢；不是精準digitization、bitcell資料或foundry macro |

WL1 / AL1 / OL1 / WL2 / AL2 / OL2 / UB 各自提供 physical cost descriptor。PE 的 L1 area 按實際 PE 數計；L2 / UB 為 shared die resources。每階層預留足夠 banks 服務所設頻寬，scheduler 仍限制共享服務，不會把 bank count 當成額外免費頻寬。銀行衝突與 RTL mux timing 尚未模擬。

核心流量以 named read_bits/write_bits ledger 計费；activation read、weight read、psum read/write、refill、UB backing與輸出寫回分開。所有 bank bits 的總和必須等於原 microkernel 的 SRAM access bits；能量總和也與排程帳本核對。Package transfer 僅對實際 compute endpoints 的 UB 計費，DRAM 目的端不算 UB write。

雙緩衝僅複製 L1/L2，容量與周邊都付面積／leakage；UB 不自動再翻倍。此模型可以探索不同 banks / ports / 容量的差異，但沒有證據能保證其絕對 PPA。必須連同敏感度結果解讀。

## GRS：whole macro、方向與 bumps

原始 GRS 為16 nm、25 Gb/s/pin、八條data lane與一條forwarded clock；每 brick 包含5x4共20bumps，餘下11個為power/ground。1.17 pJ/bit涵蓋Tx+Rx，測試條件包含10mm organic channel。686x565um = .387590mm² footprint包含bumps；403x202um = .081406mm² active circuitry已包含其中，不能再加一次。

本專案將 link 配置為每方向需要 ceil(requested bits/s / (8*25Gb/s)) 個whole macro，**每個endpoint分別付 TX與RX macro**。此為保守 full-duplex adaptation，不是聲稱原作者直接實作此端口。100Gb/s每方向會付兩個macro、共16data、2clock、22power/ground、40bumps；每方向名義容量200Gb/s，有未使用lane仍付成本。

Native Rapid 的 continuous bump budget不是pin placement。GRS端口分配已支付的whole footprint，採125um假設pitch可容納5x4array；此pitch不出自GRS量測。Native bit/wire bandwidth以25Gb/s序列lane rate換算，再由實際whole macro lane容量及要求rate限制。Router/PHY/wire延遲仍以獨立NoP clock換算，不能把25Gb/s誤作25GHz router。DDR端口仍是generic未校準bump模型。

此處只核對lane／direction／clock／PG数量、footprint與array容納條件；不能證明接線、electrical、thermal或timing signoff。不同長度link採相同1.17係數的可用性另列假設，敏感度測試energy變動。

來源：[NVIDIA GRS JSSC](https://research.nvidia.com/sites/default/files/pubs/2019-01_A-1.17-pJ/b%2C-25-Gb/s/pin/JSSCC_2019_GRS_FINAL.pdf)。

## 仍是探索假設的項目

| 成本 | 狀態 |
|---|---|
| NoC .4 pJ/bit-hop / IO crossbar .1 pJ/bit | 自訂近似，不是已量測係數 |
| compute/io/memory .02/.008/.005 W/mm² leakage | 未校準；欠缺voltage/temperature/state資料 |
| memory die 20 mm² / 2GiB | 容量線性面積假設，非memory IP資料 |
| DDR controller+PHY 10.5 mm² / 18.75GB/s | 頻寬線性縮放假設；沒有lane、ports或DDR IP timing校準 |
| interdie router .042 / CPU .109 mm² | INDM文獻參考anchor，套到目前規模與clock的可用性未驗證 |
| PE/ring/router/crossbar面積、link idle功率 | 自訂探索近似；不能因鄰近來源已核對而當成驗證數值 |
| DRAM 8.75 pJ/bit | 文獻參考；目前memory種類／工作條件對應仍未校準 |

每個結果含cost_profile、bank表、component能量與來源假設、oracle source hashes；hardware.json保存真正執行的全部欄位。無cost_profile的歷史JSON會被讀為 `legacy_mixed_v0`，不可靜默套用新成本。歷史 .192 或混合12/16nm結果保留作紀錄，已撤回作為可靠硬體成本的說法。

## PPA 限制與最接近方案

四種偏好仍是Gemini V-A啟發的 E^wE A^wA D^wD，多模型幾何平均、共同固定單位；Area代理製造成本，不是原論文完整monetary cost。每一項PPA可啟用數值並勾選strict。

- Strict：所有所選模型都必須通過；不可用平均隱藏某模型超限。物理容量／schedule／mapping錯誤不可放寬。
- Soft：超出目標不淘汰，乘以 max(1,estimated/target) 正規化penalty，並列出gap。
- Search：先讓compliant mapping優先；不compliant時以normalized excess繼續改善，避免第一個mapping超上限就放棄整個architecture。
- No compliant candidate：從所有模型都通過物理檢查的架構選擇最大strict relative excess最小者；同分依mean strict gap、worst soft gap、preference objective與ID排序。標記 `closest_exceeds_strict_limits`、提供各模型estimated、limit、strict及excess_percent，不宣稱符合。
- 沒有物理可行候選：明確回報，不能捏造晶片設計。有限搜尋未找到不等同數學證明整個design space不存在解。

Closest政策與soft penalty是本專案明示的extension，**不是Gemini V-A提供了這套fallback演算法**。Power是energy / batch makespan的平均推論功率，非peak/TDP；Area是總dies silicon area，非package footprint；Latency為所設batch的makespan。GUI與export均標為名義分析估算，不具實體限值保證。
