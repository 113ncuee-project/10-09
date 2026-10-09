# Cost16 執行與交付入口

以 `hybrid_cost16_sources.md` 的成本邊界與 `hybrid_cost16_validation.md` 的新數據為準。INDM＋Gemini架構／RL保留；歷史mixed-cost結果不覆寫。

## GUI

在專案根目錄PowerShell執行：

```powershell
./simplified_rapidchiplet/scripts/start_hybrid_gui.ps1
```

開啟 http://127.0.0.1:8765，輸入工作ID `c16100000001` 載入15組完整模型結果。新版預設output是 `simplified_rapidchiplet/results/hybrid_gui_cost16`，目前server已在此路徑執行。

選擇均衡／低功耗／小面積／低延遲；選擇多個模型、4/8/16dies與互連、UB與prefetch；輸入PPA並先啟用該數值，再逐項勾選strict。未勾選strict的是soft目標，仍回報gap。所有模型共同推薦一種架構；物理容量或mapping錯誤不能放寬。

Power是平均推論功率（E / batch latency），非peak/TDP；Area是total silicon，非package footprint；Latency為所設batch。沒找到strict compliant時提供closest與差距，不宣稱不存在全域可行設計，也不宣稱真實硬體signoff。

## 可重現完整基準

```powershell
.venv/Scripts/python.exe -X utf8 tools/validate_hybrid_cost16.py --worker 4
.venv/Scripts/python.exe -X utf8 tools/validate_hybrid_cost16.py --worker 8
.venv/Scripts/python.exe -X utf8 tools/validate_hybrid_cost16.py --worker 16
.venv/Scripts/python.exe -X utf8 tools/validate_hybrid_cost16.py --merge
```

三個workers也可在各自terminal執行；每個寫入不同worker資料夾。`--resume`只重用source／hardware／settings完全相符且有完整artifact的trial。source變更後必須使用新結果ID／folder重跑，不可將舊數字混入本版。

實際各組hardware在 `.../c16100000001/worker{4,8,16}/dies{4,8,16}_multi_ring/{model}/hardware.json`。此基準總UB128MiB：4die各32MiB，8die各16MiB，16die各8MiB；L1/L2兩個scratch slots，UB不倍增。compute500MHz、MAC energy依1reported op/scalar MAC convention；假設詳sources。

## 敏感度

```powershell
.venv/Scripts/python.exe -X utf8 tools/sensitivity_hybrid_cost16.py --scenario sram_area_075
.venv/Scripts/python.exe -X utf8 tools/sensitivity_hybrid_cost16.py --scenario sram_area_125
.venv/Scripts/python.exe -X utf8 tools/sensitivity_hybrid_cost16.py --scenario clock_1GHz_unverified
.venv/Scripts/python.exe -X utf8 tools/sensitivity_hybrid_cost16.py --scenario assumptions_high
.venv/Scripts/python.exe -X utf8 tools/sensitivity_hybrid_cost16.py --merge
.venv/Scripts/python.exe -X utf8 tools/verify_cost16_sensitivity.py
```

四個full場景各重跑15mapping；11個純係數場景按saved component ledger重算，三個代表場景另以full oracle抽查相符。保存每個scenario的hardware.json與mapping SHA。未重新搜尋，不是robust global optimum證明；範圍是探索假設，不是信賴區間。

## 驗證／給GPT

```powershell
Set-Location simplified_rapidchiplet
../.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -v
```

166 tests PASS。PDF可直接上傳給GPT；程式、hardware/mapping/metrics/source SHA與成本敏感度一起在 `output/hybrid_cost16/GPT_review_cost16.zip`。請實際上傳檔案，不只貼本機路徑。Primary PDF為 `output/pdf/INDM_Gemini_Cost16_Constraints_20261009.pdf`；原始sources與validation兩個Markdown也可直接上傳。
