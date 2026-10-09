# 16 nm 成本基準與 PPA 限制修正

狀態：**complete，2026-10-09**。使用者最新成本核對、strict/soft PPA與closest要求已完成程式、回歸、實際hardware重跑、敏感度與交付。INDM架構與Gemini/RL mapping保留。舊結果保留，不可混用數值或source identity；此版不宣稱完成silicon成本校準。

已修改：MAC .024 pJ/op、明示1 op/scalar MAC convention與2op敏感度；compute預設500MHz；移除Gemini12nm MAC/SRAM密度；SRAM按實體bank大小／width／ports與read/write帳本；GRS整顆8data+1clock macro每方向配置並支付含bumps footprint；native serial wire rate與router clock分離；strict/soft PPA與最接近fallback、GUI checkbox／差距。

驗收：166 tests PASS（92 hybrid）；15/15 full224模型：ResNet18/50、VGG16、MobileNetV2、AlexNet x 4/8/16dies。全部同source、500MHz、總UB128MiB、L1/L2雙緩衝；actual hardware.json與native/oracle hashes、schedule/calendar/dependencies、MAC/energy/UB/scratch、die/PE/SVG皆通過audit。

成本敏感度完成：SRAM面積0.75/1.25、1GHz未驗證、combined high四場景，各15個saved mapping完整oracle重算（60組）；11種cost-only場景（165筆）按component ledger精確重算；另以16die ResNet18完整oracle抽查MAC2op/SRAMenergy1.5/leakage2三情境相符。無上限均衡候選仍16die最佳；260mm²邊界probe因SRAMarea變動改為16die/4die或none-compliant，展示不確定性影響。不是重新搜尋或robust global-optimum證據。

GUI Edge瀏覽器驗證strict不可能→closest與布局/gap、soft→推薦並列gap、可達strict→estimated_compliant；4偏好與checkbox/config傳递正確、無page JS error。GUI仍執行 http://127.0.0.1:8765，新output root hybrid_gui_cost16，工作c16100000001可載入完整15組。所有benchmark/process已結束，不要重啟已完成工作；先核對source identity再做新任務。既有接續heartbeat原已paused，完成後勿重啟。

交付：docs/hybrid_cost16_sources.md、hybrid_cost16_validation.md、hybrid_cost16_runbook.md；10頁可搜尋／embedded-font PDF output/pdf/INDM_Gemini_Cost16_Constraints_20261009.pdf已render視檢；output/hybrid_cost16/GPT_review_cost16.zip內有新程式/實際JSON/敏感度/source manifest。工具入口tools/validate_hybrid_cost16.py、sensitivity_hybrid_cost16.py、verify_cost16_sensitivity.py。結果 simplified_rapidchiplet/results/hybrid_gui_cost16/c16100000001（worker4/8/16）。

重要：0.024採用報告op=scalar MAC推論，非已得到作者明確定義。SRAM .55um2/byte+1000um2/bank是粗略Fig.10趨勢假設；不是精準digitization或PDK macro。125um bump pitch是參考array可容納的假設，非論文實測pitch。所有PPA限制為名義估算；無silicon signoff。
