# 10/09 GitHub 發布副本

這份 repository 為目前 INDM＋Gemini＋RL 專案的乾淨 snapshot。原本工作目錄與 `0917` remote 保留；新 repository 使用 `10-09`，因 GitHub 名稱不接受 `/`。開發暫存、venv、下載的原始論文、舊結果與大型產生檔不納入 Git。

## 發布副本變更

目前 hybrid numerical model 與凍結參考結果相同。`rapid_backend.py` 只增加 `RAPIDCHIPLET_ROOT` 環境變數與 repo-relative `external/rapidchiplet` default，取代個人 Downloads 絕對路徑。因此此檔案 source SHA 與原始實驗不同，不能將歷史 results 宣稱為發布版重新執行的完整 15 組數據，也不能拿歷史 checkpoint 直接 resume 新 source。setup script 鎖定原生 backend commit 與 Python source SHA，避免最新版 upstream 漂移。

早期研究程式的 defaults 也改為 repo-relative RapidChiplet 路徑，config loader 依設定檔所在位置解析相對路徑，避免全套測試同時匯入两份 native engine。PowerShell 啟動器支援 Python venv 的 Windows／POSIX 路徑並檢查 venv。敏感度工具新增 `--folder` 選項，可對新基準重跑。其餘變更為發布 README、安裝／匯入工具、Git ignore 與紀錄；live workspace 的原始程式及數據未覆寫。

## 凍結參考數據

`reports/2026-10-09/reference_snapshot.zip` 與 `reference_manifest.json` 保留原始 source、15 組 actual hardware／mapping／metrics／timeline、60 full-oracle 與 165 ledger 敏感度的原始 checksum。原始 JSON 的執行路徑僅是歷史 provenance，不要求另一台電腦具備同樣磁碟路徑。

`python tools/import_reference_results.py` 先逐檔驗證 archive SHA，再建立 `simplified_rapidchiplet/results/hybrid_gui_cost16/c16100000001` 的衍生 GUI viewing copy，只改 JSON 結果資料夾路徑。archive 與原始 hashes 不改。`--verify-only` 可再次檢查。viewing copy 的 `publication_view.json` 明確標示 archival；這不是以新 source 重跑的結果。

## 重新產生完整基準

安裝依賴後，用新的 output folder 重跑，避免覆寫或誤用 archival viewing copy：

```powershell
.venv/Scripts/python.exe -X utf8 tools/validate_hybrid_cost16.py --worker 4 --folder simplified_rapidchiplet/results/hybrid_gui_cost16/c16100000002
.venv/Scripts/python.exe -X utf8 tools/validate_hybrid_cost16.py --worker 8 --folder simplified_rapidchiplet/results/hybrid_gui_cost16/c16100000002
.venv/Scripts/python.exe -X utf8 tools/validate_hybrid_cost16.py --worker 16 --folder simplified_rapidchiplet/results/hybrid_gui_cost16/c16100000002
.venv/Scripts/python.exe -X utf8 tools/validate_hybrid_cost16.py --merge --folder simplified_rapidchiplet/results/hybrid_gui_cost16/c16100000002
```

此 workers／merge 保存目前 source SHA 與實際 hardware，audit 驗證 timeline 依賴、能量與容量。敏感度工具的預設 folder 指向原始 reference ID；重跑時以 `--help` 確認 `--folder` 選項並傳入新 folder。`validate_hybrid_oct09.py` 目前只作為 Cost16 auditor 共用 helper；不要直接執行其歷史 mixed-cost worker。

發布時另在此副本執行全套測試、native dependency checksum 與 toy RL smoke、archival GUI API／layout 讀取；結果記於 `publication_validation.json`。原始 PDF／validation 中描述的是凍結的完整模型驗證，與發布 portability 檢查分開保留。

## 第三方與成本限制

原生 RapidChiplet 以官方 [spcl/rapidchiplet](https://github.com/spcl/rapidchiplet) 為外部依賴，未複製其 source 進 Git。Gemini／INDM 的方法引用與硬體成本来源詳見 `hybrid_cost16_sources.md`、mapping source notes。未為第三方或本研究擅自加入新的開源授權；對外再散布前依各來源授權條件處理。

PPA 為分析模型內的估計，尚未完成硬體 signoff；latest 成本與 strict／soft 邊界以 Cost16 sources 和 validation 為準。其餘歷史 docs 只供架構脈絡。
