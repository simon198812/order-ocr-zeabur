# order-ocr-zeabur
醫院訂單 OCR + Google Sheet + Ragic (SFinc 訂單總單)

## Ragic 寫入規則
- 以 `po_no` (訂單編號) 為分組鍵，同訂單號的多個品項合併為一筆 Ragic 訂單 (主表 + 子表)
- 主表必填欄位：`訂單編號 (1000320)`、`訂單日期 (1000322)` — 前端送出前會強制驗證
- 寫入順序：Google Sheet → Ragic；Ragic 失敗不擋 Sheet
- 重複偵測：同 `po_no` 已存在於 Ragic 會自動跳過
- 需設定環境變數 `RAGIC_API_KEY` (見 env.example)
