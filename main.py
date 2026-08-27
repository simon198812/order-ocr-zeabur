"""OCR 訂單處理系統 - FastAPI 後端 v2.2

基於 OCRtoSheet 重構優化:
- Gemini 2.5 Flash + 結構化輸出 (response_schema) → 穩定度大幅提升
- 支援 PDF (Gemini 原生支援,不用 poppler)
- 用 google-auth 取代 deprecated 的 oauth2client
- Google Sheet 連線快取 (不用每次重建)
- 批次寫入 (append_rows 一次寫完,減少 API call)
- 密碼保護 (環境變數 APP_PASSWORD)
- /preview 預覽模式 + /submit 確認後才寫入
- /health 健康檢查
- 格式清洗: po_no (M+9碼), ext (4~5碼) regex 驗證
- 檔案自然排序 (1.jpg < 2.jpg < 10.jpg)
- 保固月數獨立欄位 + 總金額自動計算
"""
import os
import io
import json
import base64
import hmac
import re
import secrets
import tempfile
from datetime import datetime
from functools import lru_cache
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Depends, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

# ------------------- 環境變數 -------------------
GEMINI_API_KEY          = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GOOGLE_SHEET_ID         = os.getenv("GOOGLE_SHEET_ID", "").strip()
GOOGLE_SHEET_TAB        = os.getenv("GOOGLE_SHEET_TAB", "工作表1").strip()
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()
APP_PASSWORD            = os.getenv("APP_PASSWORD", "").strip()

# Ragic (SFinc 帳號 / 訂單總單 KEY訂單 表單 ID = 4)
RAGIC_BASE_URL    = os.getenv("RAGIC_BASE_URL", "https://ap2.ragic.com/SFinc").strip().rstrip("/")
RAGIC_FORM_PATH   = os.getenv("RAGIC_FORM_PATH", "/forms2/4").strip()
RAGIC_CUSTOMER_PATH = os.getenv("RAGIC_CUSTOMER_PATH", "/e5aea2e688b6/1").strip()
RAGIC_PRODUCT_PATH  = os.getenv("RAGIC_PRODUCT_PATH", "/e5aea2e688b6/18").strip()
RAGIC_API_KEY     = os.getenv("RAGIC_API_KEY", "").strip()
RAGIC_ENABLED     = os.getenv("RAGIC_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

# Session (重啟失效,簡單安全)
SESSION_SECRET = secrets.token_urlsafe(32)
SESSION_COOKIE = "ocr_session"

# ------------------- 支援的檔案格式 -------------------
SUPPORTED_MIME = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
    ".pdf":  "application/pdf",
}

# ------------------- OCR Prompt -------------------
PROMPT = """你是醫院訂購單資料萃取助手。請仔細分析這份訂購單文件,判斷是「一般訂單」還是「合約品項」,並提取所有資訊。

重要規則:
1. 一份訂購單可能包含多個品項 (設備/耗材),每個品項要分開提取為獨立資料。
2. 訂單類型判斷:
   - 「一般訂單」: 有「訂購量/幣別/單價/訂購總金額」欄位
   - 「合約品項」: 有「資材代碼」欄位,備註通常寫「合約品項」,沒有單價
3. 若某個欄位找不到,字串欄位填空字串 "",數字欄位填 0
4. 日期維持民國年原格式 (例如 1150127)
5. hospital 是訂購單抬頭的開單醫院,loc 是交貨地點 (可能不同)
6. note 欄位放備註、其他補充資訊（不要放保固月數）

欄位格式規則（請嚴格遵守）:
- po_no (訂單號): 固定格式為大寫字母 M 開頭 + 9 位數字,共 10 碼,例如 M123456789
  圖片中可能因換行或空格拆成兩行,請合併為完整值。若找不到符合格式,填 ""
- ext (分機/聯絡電話): 專指「申購人本人」的聯絡方式，可能為以下三種格式之一：
  A. 4~6 碼純數字分機 (例如 1698、16895、160892)
  B. 台灣手機號碼 10 碼 09xxxxxxxx (例如 0912345678)
  C. 手機可能出現「-」或空白 (例如 0912-345-678、0912 345 678)，全部去掉只留數字

⚠️ 訂單常出現多個聯絡方式,請只抓申購人的,排除以下:
  - 採購人員分機 (例如「再聯絡採購人員XXX分機:14738」)
  - 職安室/廠商安全分機 (例如「分機4104~4105」)

抓取步驟:
  STEP 1: 鎖定「申購人:XXX」這個人名
  STEP 2: 從申購人附近找「分機:」「分機號碼:」「電話:」「手機:」標籤
  STEP 3: 標籤同行有 4~6 碼數字 或 10 碼 09xx 手機 → 採用
  STEP 4: ⚠️ 標籤同行只有空白或冒號 →
          往「下一行的開頭」找第一個 4~6 碼分機或 10 碼手機
          (常見排版,例如「分機:[換行]16895」)
  STEP 5: 整個申購人區域都找不到 → 才填 ""
- 金額欄位 (price, total): 只填數字,不含貨幣符號 ($ NT 等)
- warranty (保固月數): 只填純數字,例如 24、12。從備註或保固欄位提取,不要把「保固月數:24」放在 note 裡
- order_total (總金額): 該訂單所有品項的 total (訂購總金額) 加總。如果只有一個品項,order_total = total

請依照 JSON schema 輸出,orders 陣列中每個品項一筆資料。
"""

# ------------------- 結構化輸出 Schema -------------------
_ORDER_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id":      {"type": "string", "description": "訂單號-項次 (例如 M150101275-1)"},
        "po_no":        {"type": "string", "description": "訂單號 M+9碼"},
        "type":         {"type": "string", "description": "一般訂單 或 合約品項"},
        "hospital":     {"type": "string", "description": "開單醫院"},
        "date":         {"type": "string", "description": "訂購日期 (民國年)"},
        "vendor":       {"type": "string", "description": "廠商名稱"},
        "loc":          {"type": "string", "description": "交貨地點"},
        "item":         {"type": "string", "description": "品名"},
        "qty":          {"type": "number", "description": "訂購量/數量"},
        "unit":         {"type": "string", "description": "單位 (個/台/箱/支)"},
        "price":        {"type": "number", "description": "單價"},
        "total":        {"type": "number", "description": "訂購總金額 (單品項 qty × price)"},
        "order_total":  {"type": "number", "description": "總金額 (該訂單所有品項的 total 加總)"},
        "brand":        {"type": "string", "description": "廠牌型號"},
        "deadline":     {"type": "string", "description": "交貨期限/預計到貨日"},
        "dept":         {"type": "string", "description": "請購部門"},
        "user":         {"type": "string", "description": "申購人"},
        "ext":          {"type": "string", "description": "分機 4~5碼"},
        "sign_no":      {"type": "string", "description": "簽辦案號"},
        "material_id":  {"type": "string", "description": "資材代碼 (合約品項才有)"},
        "note":         {"type": "string", "description": "備註 (不含保固月數)"},
        "warranty":     {"type": "number", "description": "保固月數 (純數字,例如 24)"},
    },
    "required": [
        "item_id", "po_no", "type", "hospital", "date", "vendor", "loc",
        "item", "qty", "unit", "price", "total", "order_total", "brand", "deadline",
        "dept", "user", "ext", "sign_no", "material_id", "note", "warranty",
    ],
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "orders": {"type": "array", "items": _ORDER_ITEM_SCHEMA}
    },
    "required": ["orders"],
}

# Sheet 欄位順序 (23 欄,與 Google Sheet 表頭一致)
SHEET_COLUMNS = [
    "hospital", "date", "deadline", "vendor", "loc", "type", "material_id",
    "item", "brand", "qty", "unit", "price", "total",
    "dept", "user", "ext",
    "po_no", "item_id", "idx", "sign_no", "order_total",
    "note", "warranty",
]

# ------------------- Gemini client (快取) -------------------
@lru_cache(maxsize=1)
def get_gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 未設定")
    return genai.Client(api_key=GEMINI_API_KEY)

# ------------------- Google Sheets (快取) -------------------
@lru_cache(maxsize=1)
def get_sheet():
    if not GOOGLE_CREDENTIALS_JSON:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON 未設定")
    raw = GOOGLE_CREDENTIALS_JSON
    if raw.startswith("{"):
        creds_dict = json.loads(raw)
    else:
        creds_dict = json.loads(base64.b64decode(raw))
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_TAB)

# ------------------- 格式清洗 + 排序 -------------------
def _clean_po_no(v):
    """訂單號: M + 9位數字，處理換行拆分"""
    m = re.search(r'M\d{9}', re.sub(r'\s+', '', str(v)), re.I)
    return m.group(0).upper() if m else ''

def _clean_ext(v):
    """分機/聯絡電話：優先手機 (10 碼 09xx)，其次 4~6 碼分機。處理換行拆分。"""
    s = str(v or '')
    # 移除空白、破折號、括號等雜訊
    compact = re.sub(r'[\s\-\(\)（）]+', '', s)
    # 1. 台灣手機 09xxxxxxxx (10 碼) 優先
    m = re.search(r'09\d{8}', compact)
    if m:
        return m.group(0)
    # 2. 純數字判斷 4~6 碼
    d = re.sub(r'\D', '', s)
    if re.fullmatch(r'\d{4,6}', d):
        return d
    # 3. fallback：找 4~6 碼數字片段（貪婪，優先長的）
    m = re.search(r'\d{4,6}', compact)
    return m.group(0) if m else ''

def _clean_order(o):
    """對單筆訂單做格式後處理"""
    o = dict(o)
    if o.get('po_no'):
        o['po_no'] = _clean_po_no(o['po_no'])
    if o.get('ext'):
        o['ext'] = _clean_ext(o['ext'])
    # item_id 跟著 po_no 更新
    if o.get('po_no') and o.get('item_id'):
        pts = str(o['item_id']).rsplit('-', 1)
        if len(pts) == 2:
            o['item_id'] = f"{o['po_no']}-{pts[1]}"
    # warranty: 確保是數字
    if o.get('warranty'):
        w = re.sub(r'\D', '', str(o['warranty']))
        o['warranty'] = int(w) if w else 0
    return o

def _natural_sort_key(f):
    """1.jpg < 2.jpg < 10.jpg 自然排序"""
    pts = re.split(r'(\d+)', (f.filename or '').lower())
    return [int(p) if p.isdigit() else p for p in pts]

def _try_recover_truncated_json(text: str) -> Optional[str]:
    """Gemini 輸出 JSON 被截斷時的緊急救援。
    找最後一個完整的 order 物件結尾「},」，截斷後補上 ]}，
    至少保住前面成功的訂單資料。"""
    if not text:
        return None
    last_complete = text.rfind('},')
    if last_complete < 0:
        # 沒有任何完整的 order，或者整個輸出已經有完整 }}
        last_complete = text.rfind('}}')
        if last_complete < 0:
            return None
        return text[:last_complete + 2]
    return text[:last_complete + 1] + ']}'

# ------------------- OCR 核心 -------------------
def extract_orders_from_file(file_bytes: bytes, filename: str) -> list[dict]:
    """用 Gemini 從檔案 (圖片/PDF) 萃取訂單資料"""
    ext = os.path.splitext(filename)[1].lower()
    mime = SUPPORTED_MIME.get(ext)
    if not mime:
        raise ValueError(f"不支援的檔案格式: {ext}，支援: {', '.join(SUPPORTED_MIME.keys())}")

    client = get_gemini_client()

    # 大檔 (>15MB) 用 Files API
    if len(file_bytes) > 15 * 1024 * 1024:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            uploaded = client.files.upload(file=tmp_path)
            contents = [uploaded, PROMPT]
        finally:
            os.unlink(tmp_path)
    else:
        contents = [
            types.Part.from_bytes(data=file_bytes, mime_type=mime),
            PROMPT,
        ]

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.1,
            max_output_tokens=65536,  # 大幅放寬避免品項多時被截斷
        ),
    )

    raw_text = response.text
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # 嘗試恢復截斷的 JSON (保住前面成功的訂單)
        recovered = _try_recover_truncated_json(raw_text or "")
        if recovered:
            try:
                data = json.loads(recovered)
            except Exception:
                raise ValueError(
                    f"Gemini 輸出 JSON 解析失敗 ({e})；輸出長度 {len(raw_text or '')} 字元，"
                    f"恢復截斷後仍失敗。請拆分 PDF 成更小檔案再試。"
                ) from e
        else:
            raise ValueError(
                f"Gemini 輸出 JSON 解析失敗 ({e})；輸出長度 {len(raw_text or '')} 字元。"
                f"請拆分 PDF 成更小檔案再試。"
            ) from e

    orders = data.get("orders", [])
    # 格式後處理
    orders = [_clean_order(o) for o in orders]
    return orders

# ------------------- Sheet 寫入 -------------------
def orders_to_rows(orders: list[dict]) -> list[list]:
    rows = []
    for o in orders:
        item_id = str(o.get("item_id", "") or "")
        idx = ""
        if "-" in item_id:
            try:
                idx = item_id.rsplit("-", 1)[1]
            except Exception:
                idx = ""
        row = []
        for col in SHEET_COLUMNS:
            if col == "idx":
                row.append(idx)
            else:
                v = o.get(col, "")
                if isinstance(v, (int, float)) and v == 0:
                    row.append("")
                else:
                    row.append(v)
        rows.append(row)
    return rows

def batch_write_to_sheet(orders: list[dict]) -> int:
    if not orders:
        return 0
    sheet = get_sheet()
    rows = orders_to_rows(orders)
    sheet.append_rows(rows, value_input_option="USER_ENTERED")
    return len(rows)

# ------------------- Ragic 整合 -------------------
# vendor 名稱關鍵字 → 公司單選值 (Ragic 1002403)
# 含常見 OCR 誤識別字，例如「長州」是「長洲」的誤識、「靖晨」是「靖展」誤識
_VENDOR_TO_COMPANY = [
    ("2 尚鋒", ["尚鋒", "尚峰", "尚锋"]),
    ("1 長洲", ["長洲", "長州", "长洲", "长州"]),
    ("3 靖展", ["靖展", "靖晨", "静展", "静晨"]),
]
# 首字 fallback：三家公司名首字都獨特
_VENDOR_FALLBACK = [
    ("2 尚鋒", "尚"),
    ("1 長洲", "長"),  # 亦包含「长」
    ("1 長洲", "长"),
    ("3 靖展", "靖"),
    ("3 靖展", "静"),
]

def _map_vendor_to_company(vendor: str) -> str:
    if not vendor:
        return ""
    s = str(vendor)
    for val, kws in _VENDOR_TO_COMPANY:
        for kw in kws:
            if kw in s:
                return val
    # fallback：找首字（處理更嚴重的 OCR 錯字，例如「尚〇工業」）
    for val, first_char in _VENDOR_FALLBACK:
        if first_char in s:
            return val
    return ""

def _minguo_to_western(date_str) -> str:
    """民國年 → 西元年。1150127 → 2026/01/27。無法解析回傳空字串。"""
    s = re.sub(r"\D", "", str(date_str or ""))
    if not s:
        return ""
    if len(s) == 7:  # yyyMMdd 民國 (例如 1150127)
        yyy, mm, dd = s[0:3], s[3:5], s[5:7]
        try:
            return f"{int(yyy) + 1911}/{mm}/{dd}"
        except Exception:
            return ""
    if len(s) == 6:  # yyMMdd 民國 (例如 991027)
        yyy, mm, dd = s[0:2], s[2:4], s[4:6]
        try:
            return f"{int(yyy) + 1911}/{mm}/{dd}"
        except Exception:
            return ""
    if len(s) == 8:  # 已是西元 yyyyMMdd
        return f"{s[0:4]}/{s[4:6]}/{s[6:8]}"
    return ""

def _group_orders_by_po(orders: list[dict]) -> dict:
    """以 po_no 分組；空 po_no 不參與分組。保留原始順序。"""
    groups = {}
    for o in orders:
        po = str(o.get("po_no") or "").strip()
        if not po:
            continue
        groups.setdefault(po, []).append(o)
    return groups

def _ragic_url() -> str:
    return f"{RAGIC_BASE_URL}{RAGIC_FORM_PATH}"

def _ragic_url_for(path: str) -> str:
    """組其他表單的 API 網址 (例如商品資料表)。"""
    return f"{RAGIC_BASE_URL}{path}"

def _ragic_headers(extra: Optional[dict] = None) -> dict:
    """Ragic 官方建議用法：Authorization: Basic {api_key} (直接放，不再 base64 encode)。
    Ragic 的 API Key 本身已經是 Base64 字串。"""
    h = {"Authorization": f"Basic {RAGIC_API_KEY}"}
    if extra:
        h.update(extra)
    return h

def _extract_record_ids(data) -> list[str]:
    """從 Ragic GET 回應 dict 中只挑出真實 record_id (純數字 key)。
    過濾掉 Ragic 可能夾帶的 metadata key (例如 status, msg)。"""
    if not isinstance(data, dict):
        return []
    return [str(k) for k in data.keys() if str(k).isdigit()]

def _ragic_check_duplicate(po_no: str) -> Optional[str]:
    """查 Ragic 1000398 (訂單號碼客戶=醫院 po_no) 是否已有同值 record。

    使用者 Ragic 設定：
    - 1000320 訂單編號 = 可重複 (使用者批次號)
    - 1000398 訂單號碼(客戶) = 唯一 (識別一張訂單的鍵)
    所以重複偵測查 1000398，傳入值是 OCR 抓的 po_no。
    """
    if not po_no:
        return None
    encoded_po = requests.utils.quote(str(po_no), safe='')
    url = f"{_ragic_url()}?api&naming=EID&subtables=0&limit=0,5&where=1000398,eq,{encoded_po}"
    try:
        r = requests.get(url, headers=_ragic_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    target = str(po_no).strip()
    for rid, rec in data.items():
        if not str(rid).isdigit() or not isinstance(rec, dict):
            continue
        stored = str(rec.get("1000398", "") or "").strip()
        if stored == target:
            return str(rid)
    return None

def _pick_customer(items: list[dict]) -> tuple[str, str]:
    """從群組挑第一個有值的 ragic_customer_code 與類別；回傳 (code, category)。"""
    for it in items:
        code = str(it.get("ragic_customer_code", "") or "").strip()
        if code:
            cat = str(it.get("ragic_customer_category", "") or "").strip()
            return code, cat
    return "", ""

def _find_product_by_name(name: str) -> dict:
    """從商品清單找對應商品；找不到回空 dict。"""
    if not name:
        return {}
    target = str(name).strip()
    for p in _ragic_load_products():
        if p.get("name") == target:
            return p
    return {}

def _find_customer_by_code(code: str) -> dict:
    """從 customerList 找對應客戶；找不到回空 dict。"""
    if not code:
        return {}
    for c in _ragic_load_customers():
        if c.get("code") == code:
            return c
    return {}

def _build_ragic_payload(po_no: str, items: list[dict]) -> dict:
    """構造 Ragic POST form 參數 (主表 + 子表 1000341)。

    - 1000320 訂單編號：來自使用者手填的 ragic_po
    - 1000319 客戶編號：使用者前端從下拉選的 ragic_customer_code (Ragic 客戶資料表編號)
    - 1000654 營業稅：客戶類別=醫院 → 0；其他 → 0.05
    - 1000398 訂單號碼(客戶)：仍用 OCR 抓的 po_no
    """
    head = items[0]

    def _s(k):
        return str(head.get(k, "") or "").strip()

    ragic_po = _pick_ragic_po(items, fallback=po_no)
    cust_code, cust_category = _pick_customer(items)

    # 客戶編號 fallback: 沒選的話用 OCR loc/hospital (純文字，連結會斷)
    if not cust_code:
        cust_code = _s("loc") or _s("hospital")

    # 從客戶清單抓對應資料，明確寫入訂單載入欄位 (Ragic API 寫入不會自動載入)
    cust = _find_customer_by_code(cust_code)
    if cust and not cust_category:
        cust_category = cust.get("category", "")

    # 營業稅: 醫院 0%，其他 5%
    tax_rate = "0" if cust_category == "醫院" else "0.05"

    apply_unit_parts = [v for v in (_s("dept"), _s("user"), _s("ext")) if v]
    apply_unit = " / ".join(apply_unit_parts)

    payload = {
        "1002403": _map_vendor_to_company(head.get("vendor", "")),  # 公司 (單選)
        "1000320": ragic_po,                                        # 訂單編號 (使用者手填)
        "1000322": _minguo_to_western(head.get("date", "")),        # 訂單日期 (必填)
        "1007107": _minguo_to_western(head.get("deadline", "")),    # 交貨期限 (OCR deadline)
        "1000319": cust_code,                                       # 客戶編號 (連結欄位，必填)
        "1000654": tax_rate,                                        # 營業稅(選%) (必填)
        "1000398": po_no,                                           # 訂單號碼(客戶) — 醫院 po_no
        "1000399": apply_unit,                                      # 申請單位
        "1000339": _s("note"),                                      # 主表備註
    }

    # 明確寫入客戶相關載入欄位 (Ragic API 寫入不會自動觸發載入)
    # 文件保證「手動設定載入欄位的值，該值會被保留」
    if cust:
        if cust.get("category"):
            payload["1000530"] = cust["category"]      # 客戶類別
        if cust.get("name"):
            payload["1000321"] = cust["name"]          # 客戶名稱
        if cust.get("short_name"):
            payload["1000414"] = cust["short_name"]    # 客戶簡稱
        if cust.get("tax_id"):
            payload["1000415"] = cust["tax_id"]        # 客戶統編
        if cust.get("phone"):
            payload["1000325"] = cust["phone"]         # 電話
        if cust.get("address"):
            payload["1000329"] = cust["address"]       # 地址
        if cust.get("ship_address"):
            payload["1000330"] = cust["ship_address"]  # 送貨地址
        if cust.get("sales"):
            payload["1006517"] = cust["sales"]         # 負責業務

    # 子表 1000341：每個品項一行；row id 用負數
    for i, it in enumerate(items, start=1):
        rid = -i
        idx_val = str(it.get("idx", "") or i)
        qty = it.get("qty", "")
        price = it.get("price", "")
        payload[f"1000331_{rid}"] = idx_val                                  # 項次
        # 品名規格 (連結欄位)：優先用前端智慧比對/手選的 Ragic 商品名，
        # 沒有才 fallback 用 OCR 原文 (連結會斷，需人工修)
        product_name = str(it.get("ragic_product_name", "") or "").strip() \
                       or str(it.get("item", "") or "")
        payload[f"1000332_{rid}"] = product_name
        # 單位 (載入欄位)：Ragic API 寫入連結欄位不會自動觸發載入，需明確帶值。
        # 優先用商品主檔的單位，其次 OCR 抓到的
        prod = _find_product_by_name(product_name)
        unit = (prod.get("unit") or "").strip() or str(it.get("unit", "") or "").strip()
        if unit:
            payload[f"1000342_{rid}"] = unit
        payload[f"1000334_{rid}"] = "" if qty in (None, "", 0) else str(qty)
        payload[f"1000333_{rid}"] = "" if price in (None, "", 0) else str(price)
        # 子表備註：合約品項附上資材代碼，方便人工對帳追溯
        note_parts = []
        mat_id = str(it.get("material_id", "") or "").strip()
        if mat_id:
            note_parts.append(f"資材碼 {mat_id}")
        if str(it.get("note", "") or "").strip():
            note_parts.append(str(it["note"]).strip())
        payload[f"1006516_{rid}"] = " ".join(note_parts)                     # 子表備註

    # 過濾空值；保留必填欄位即使空也要送 (送空才好定位錯誤)
    keep = {"1000320", "1000322"}
    return {k: v for k, v in payload.items() if v != "" or k in keep}

def _ragic_create_order(po_no: str, items: list[dict]) -> dict:
    payload = _build_ragic_payload(po_no, items)
    headers = _ragic_headers({"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
    url = f"{_ragic_url()}?api"
    try:
        # 編碼成 UTF-8 byte string，避免 requests 預設用 latin-1 處理中文
        encoded = "&".join(
            f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in payload.items()
        ).encode("utf-8")
        r = requests.post(url, data=encoded, headers=headers, timeout=30)
        if r.status_code != 200:
            return {"po_no": po_no, "status": "error", "items": len(items),
                    "message": f"HTTP {r.status_code}: {r.text[:300]}"}
        try:
            body = r.json()
        except Exception:
            return {"po_no": po_no, "status": "error", "items": len(items),
                    "message": f"非 JSON 回應 (檢查 ?api 與 API Key): {r.text[:300]}"}
        if body.get("status") == "SUCCESS":
            ragic_id = body.get("ragicId")
            # 觸發 Ragic 連結載入：寫入後立刻 POST 同 record_id 重送客戶編號
            # 這通常會逼 Ragic 重新計算「載入欄位」(客戶名稱/類別/地址等)
            _ragic_trigger_link_reload(ragic_id, payload)
            return {"po_no": po_no, "status": "success",
                    "ragic_id": ragic_id, "items": len(items)}
        return {"po_no": po_no, "status": "error", "items": len(items),
                "message": str(body)[:300]}
    except Exception as e:
        return {"po_no": po_no, "status": "error", "items": len(items),
                "message": f"{type(e).__name__}: {e}"}

def _ragic_upload_attachment(ragic_id, filename: str, file_bytes: bytes,
                             field_id: str = "1007106", mime: str = "application/pdf") -> bool:
    """上傳檔案到 Ragic record 的檔案欄位。best-effort，失敗不擋主流程。"""
    if not ragic_id or not file_bytes or not filename:
        return False
    url = f"{_ragic_url()}/{ragic_id}?api"
    # multipart/form-data: requests 會自動加 boundary，不能手動設 Content-Type
    headers = _ragic_headers()
    files = {field_id: (filename, file_bytes, mime)}
    try:
        r = requests.post(url, files=files, headers=headers, timeout=60)
        if r.status_code != 200:
            return False
        try:
            body = r.json()
            return body.get("status") == "SUCCESS"
        except Exception:
            return False
    except Exception:
        return False

def _guess_mime(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")

def _ragic_trigger_link_reload(ragic_id, payload: dict):
    """寫入連結欄位後，發第二次 POST 更新同 record_id 觸發 Ragic 重新計算載入欄位。
    失敗也不影響主寫入結果 (best-effort)。"""
    if not ragic_id:
        return
    customer_code = payload.get("1000319", "")
    if not customer_code:
        return
    try:
        headers = _ragic_headers({"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
        url = f"{_ragic_url()}/{ragic_id}?api"
        # 重送客戶編號 + 觸發載入欄位重算 (連結欄位變化時 Ragic 應重新載入)
        trigger_payload = {"1000319": str(customer_code)}
        encoded = "&".join(
            f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in trigger_payload.items()
        ).encode("utf-8")
        requests.post(url, data=encoded, headers=headers, timeout=15)
    except Exception:
        pass  # best-effort，不影響主寫入

@lru_cache(maxsize=1)
def _ragic_load_customers() -> list:
    """從 (尚鋒) 客戶資料 表單抓全部客戶；快取於記憶體 (lru_cache)。
    回傳 [{ragic_id, code, name, short_name, tax_id, phone, address, ship_address,
           sales, category}]。"""
    if not RAGIC_API_KEY:
        return []
    url = (
        f"{RAGIC_BASE_URL}{RAGIC_CUSTOMER_PATH}?api&naming=EID"
        f"&subtables=0&limit=0,1000"
    )
    try:
        r = requests.get(url, headers=_ragic_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    customers = []
    for rid, rec in data.items():
        if not str(rid).isdigit() or not isinstance(rec, dict):
            continue
        code = str(rec.get("1000001", "") or "").strip()
        if not code:
            continue
        customers.append({
            "ragic_id": str(rid),
            "code": code,
            "name": str(rec.get("1000002", "") or "").strip(),       # 客戶名稱
            "short_name": str(rec.get("1000003", "") or "").strip(), # 客戶簡稱
            "tax_id": str(rec.get("1000008", "") or "").strip(),     # 統一編號
            "phone": str(rec.get("1000010", "") or "").strip(),      # 電話號碼
            "address": str(rec.get("1000017", "") or "").strip(),    # 完整地址
            "ship_address": str(rec.get("1000021", "") or "").strip(),# 送貨地址
            "sales": str(rec.get("1000009", "") or "").strip(),      # 負責業務
            "category": str(rec.get("1000646", "") or "").strip(),   # 客戶類別 (醫院/同行)
        })
    # 按客戶編號排序，方便前端瀏覽
    customers.sort(key=lambda c: c["code"])
    return customers

# 商品備註 (1000632) 的資材代碼格式範例：
#   長洲 資材碼93512031
#   長洲 合約品項 資材碼 93103012
#   長洲 合約品項 資材代碼: 95111002
#   長洲 合約品項 94105001          (省略「資材碼」字樣)
#   雷射_ EM12240A                 (不是資材碼，應跳過)
_MATERIAL_COMPANIES = ["長洲", "尚鋒", "靖展"]
_MATERIAL_RE = re.compile(
    r"(?:資材\s*(?:代)?\s*[碼码]|合約品項)?"   # 可選的標記詞
    r"\s*[:：]?\s*"                            # 可選的分隔符
    r"(\d{5,12})"                              # 資材代碼：5~12 位純數字
)

def _parse_material_note(note: str) -> tuple:
    """從商品備註解析 (公司, 資材代碼)。解析不到回 (公司或'', '')。
    只認純數字代碼，避開「雷射_ EM12240A」這類字母混合的備註。"""
    if not note:
        return ("", "")
    s = str(note).strip()
    company = ""
    for c in _MATERIAL_COMPANIES:
        if c in s:
            company = c
            break
    # 只在有「資材」或「合約品項」字樣時才抽代碼，避免把其他數字誤判
    if "資材" not in s and "合約品項" not in s:
        return (company, "")
    m = _MATERIAL_RE.search(s)
    return (company, m.group(1) if m else "")

@lru_cache(maxsize=1)
def _ragic_load_products() -> list:
    """從 (尚鋒) 商品資料 表單抓全部商品；快取於記憶體。
    回傳 [{ragic_id, name, model, code, unit, category, price}]。
    name (產品名稱 1000617) 就是訂單子表「品名規格」連結欄位要寫入的值。"""
    if not RAGIC_API_KEY:
        return []
    url = (
        f"{RAGIC_BASE_URL}{RAGIC_PRODUCT_PATH}?api&naming=EID"
        f"&subtables=0&limit=0,5000"
    )
    try:
        r = requests.get(url, headers=_ragic_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    products = []
    for rid, rec in data.items():
        if not str(rid).isdigit() or not isinstance(rec, dict):
            continue
        name = str(rec.get("1000617", "") or "").strip()   # 產品名稱 (連結欄位寫入值)
        if not name:
            continue
        products.append({
            "ragic_id": str(rid),
            "name": name,
            "model": str(rec.get("1000912", "") or "").strip(),     # 廠牌型號
            "code": str(rec.get("1000916", "") or "").strip(),      # 商品編號
            "unit": str(rec.get("1000618", "") or "").strip(),      # 單位
            "category": str(rec.get("1000615", "") or "").strip(),  # 種類
            "price": str(rec.get("1000619", "") or "").strip(),     # 銷售單價
            # 合約品項：結構化的公司 + 資材代碼 (供 OCR 精準比對)
            "contract_company": str(rec.get("1007109", "") or "").strip(),
            "material_code": str(rec.get("1007110", "") or "").strip(),
        })
    products.sort(key=lambda p: p["name"])
    return products

@lru_cache(maxsize=1)
def _ragic_load_order_history() -> dict:
    """從訂單總單抓歷史紀錄，建立「客戶編號 → 曾訂購品名次數」對照表，
    供商品模糊比對加權 (該客戶買過的優先推薦)。
    註：資材代碼的精準對應改由商品資料表的 1007109/1007110 提供。
    回傳 {"history": {客戶編號: {品名: 次數}}}。"""
    if not RAGIC_API_KEY:
        return {"history": {}}
    history: dict = {}
    # 分頁抓全部訂單 (含子表 1000341 訂購項目)
    page, page_size, max_pages = 0, 1000, 20
    while page < max_pages:
        url = (
            f"{_ragic_url()}?api&naming=EID"
            f"&limit={page * page_size},{page_size}"
        )
        try:
            r = requests.get(url, headers=_ragic_headers(), timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception:
            break
        if not isinstance(data, dict):
            break
        rows = 0
        for rid, rec in data.items():
            if not str(rid).isdigit() or not isinstance(rec, dict):
                continue
            rows += 1
            cust = str(rec.get("1000319", "") or "").strip()
            if not cust:
                continue
            sub = rec.get("_subtable_1000341") or {}
            if not isinstance(sub, dict):
                continue
            bucket = history.setdefault(cust, {})
            for _, line in sub.items():
                if not isinstance(line, dict):
                    continue
                pname = str(line.get("1000332", "") or "").strip()
                if not pname:
                    continue
                bucket[pname] = bucket.get(pname, 0) + 1
        if rows < page_size:
            break
        page += 1
    return {"history": history}

def _pick_ragic_po(items: list[dict], fallback: str) -> str:
    """從群組內挑第一個有值的 ragic_po；都空就 fallback。"""
    for it in items:
        v = str(it.get("ragic_po", "") or "").strip()
        if v:
            return v
    return fallback

def write_orders_to_ragic(orders: list[dict], file_map: dict = None) -> dict:
    """主入口：分組 → 重複偵測 → 寫入主表 → (有檔案的話) 上傳附件。
    file_map: {filename: bytes}，根據 order._filename 找對應檔案上傳。"""
    summary = {"success": 0, "skipped": 0, "error": 0, "no_po": 0, "attachments": 0}
    results: list[dict] = []
    if not RAGIC_ENABLED:
        return {"enabled": False, "configured": False, "results": results, "summary": summary}
    if not RAGIC_API_KEY:
        return {"enabled": True, "configured": False, "results": results, "summary": summary,
                "message": "RAGIC_API_KEY 未設定"}

    summary["no_po"] = sum(1 for o in orders if not str(o.get("po_no") or "").strip())
    groups = _group_orders_by_po(orders)
    file_map = file_map or {}

    for po_no, items in groups.items():
        ragic_po = _pick_ragic_po(items, fallback=po_no)
        existing = _ragic_check_duplicate(po_no)
        if existing:
            summary["skipped"] += 1
            view_url = f"{_ragic_url()}/{existing}"
            results.append({"po_no": po_no, "ragic_po": ragic_po, "status": "skipped",
                            "items": len(items), "ragic_id": existing,
                            "view_url": view_url,
                            "message": f"Ragic 已存在同訂單號碼 (record_id={existing})"})
            continue
        res = _ragic_create_order(po_no, items)
        res["ragic_po"] = ragic_po

        # 寫入成功且有原始檔案 → 上傳附件到 1007106
        if res.get("status") == "success" and file_map:
            src_filename = next(
                (it.get("_filename") for it in items if it.get("_filename")),
                None,
            )
            if src_filename and src_filename in file_map:
                ok = _ragic_upload_attachment(
                    res["ragic_id"], src_filename, file_map[src_filename],
                    field_id="1007106", mime=_guess_mime(src_filename),
                )
                if ok:
                    summary["attachments"] += 1
                    res["attachment"] = src_filename
                else:
                    res["attachment_error"] = "上傳失敗"

        results.append(res)
        if res.get("status") == "success":
            summary["success"] += 1
        else:
            summary["error"] += 1

    return {"enabled": True, "configured": True, "results": results, "summary": summary}

# ------------------- 密碼驗證 -------------------
def require_auth(session: Optional[str] = Cookie(None, alias=SESSION_COOKIE)):
    if not APP_PASSWORD:
        return
    if not session or not hmac.compare_digest(session, SESSION_SECRET):
        raise HTTPException(status_code=401, detail="需要登入")

# ------------------- FastAPI App -------------------
app = FastAPI(title="OCR 訂單處理系統", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY),
        "sheet_configured": bool(GOOGLE_CREDENTIALS_JSON),
        "ragic_enabled": RAGIC_ENABLED,
        "ragic_configured": bool(RAGIC_API_KEY),
        "ragic_url": _ragic_url() if RAGIC_API_KEY else "未設定 API Key",
        "password_protected": bool(APP_PASSWORD),
        "model": GEMINI_MODEL,
    }

def _api_key_fingerprint(k: str) -> dict:
    """回傳 API Key 的指紋資訊 (不洩漏完整 Key)，方便排查環境變數有沒有夾雜垃圾。"""
    if not k:
        return {"length": 0, "has_control_char": False}
    has_ctrl = any(ord(c) < 32 or ord(c) == 127 for c in k)
    has_space = any(c in (" ", "\t", "\n", "\r") for c in k)
    return {
        "length": len(k),
        "first4": k[:4],
        "last4": k[-4:] if len(k) >= 4 else "",
        "has_control_char": has_ctrl,
        "has_whitespace": has_space,
    }

@app.get("/ragic/check-dup", dependencies=[Depends(require_auth)])
def ragic_check_dup(po: str):
    """診斷重複偵測：傳訂單號碼 (對應 1000398，OCR 抓的 po_no)，
    回傳 Ragic 實際匹配的紀錄。1000398 才是唯一鍵；1000320 可重複。"""
    out = {"query_po": po, "field_checked": "1000398 (訂單號碼客戶)"}
    if not RAGIC_ENABLED or not RAGIC_API_KEY:
        out["error"] = "Ragic 未啟用或未設定 API Key"
        return out
    encoded_po = requests.utils.quote(str(po), safe='')
    url = f"{_ragic_url()}?api&naming=EID&subtables=0&limit=0,5&where=1000398,eq,{encoded_po}"
    out["url"] = url
    try:
        r = requests.get(url, headers=_ragic_headers(), timeout=15)
        out["http_status"] = r.status_code
        try:
            data = r.json()
        except Exception:
            out["error"] = "回應非 JSON"
            out["body_preview"] = r.text[:300]
            return out
        if not isinstance(data, dict):
            out["decision"] = "not_exists"
            out["raw_data_type"] = type(data).__name__
            return out
        out["response_top_keys"] = list(data.keys())[:20]
        target = str(po).strip()
        samples = []
        matched_id = None
        for rid, rec in data.items():
            if not str(rid).isdigit() or not isinstance(rec, dict):
                continue
            stored = str(rec.get("1000398", "") or "").strip()
            samples.append({
                "record_id": str(rid),
                "stored_1000398": stored,
                "matches_query": stored == target,
            })
            if stored == target and not matched_id:
                matched_id = str(rid)
        out["samples"] = samples[:10]
        out["matched_record_id"] = matched_id
        out["decision"] = "exists" if matched_id else "not_exists"
        out["where_filter_works"] = (len(samples) == 0) or all(s["matches_query"] for s in samples)
        if matched_id:
            out["view_url"] = f"{_ragic_url()}/{matched_id}"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out

@app.get("/ragic/customers", dependencies=[Depends(require_auth)])
def ragic_customers(refresh: int = 0):
    """回傳 (尚鋒) 客戶資料表清單給前端做下拉。?refresh=1 強制清快取重抓。"""
    if refresh:
        _ragic_load_customers.cache_clear()
    customers = _ragic_load_customers()
    return {"count": len(customers), "customers": customers}

@app.get("/ragic/products", dependencies=[Depends(require_auth)])
def ragic_products(refresh: int = 0):
    """回傳 (尚鋒) 商品資料表清單給前端做智慧比對 + 下拉。?refresh=1 強制重抓。"""
    if refresh:
        _ragic_load_products.cache_clear()
    products = _ragic_load_products()
    return {"count": len(products), "products": products}

@app.get("/ragic/raw-product", dependencies=[Depends(require_auth)])
def ragic_raw_product(keyword: Optional[str] = None, limit: int = 2):
    """診斷用：回傳商品資料表的完整原始欄位 (含所有 EID)，
    用來確認資材代碼存在哪個欄位。"""
    if not RAGIC_API_KEY:
        return {"error": "未設定 RAGIC_API_KEY"}
    url = f"{RAGIC_BASE_URL}{RAGIC_PRODUCT_PATH}?api&naming=EID&limit=0,{max(1, min(limit, 5))}"
    if keyword:
        url += f"&where=1000617,like,{requests.utils.quote(keyword, safe='')}"
    try:
        r = requests.get(url, headers=_ragic_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if not isinstance(data, dict):
        return {"error": "非預期回應"}
    out = []
    for rid, rec in data.items():
        if not str(rid).isdigit() or not isinstance(rec, dict):
            continue
        # 只留有值的欄位，方便閱讀
        out.append({
            "ragic_id": rid,
            "fields": {k: v for k, v in rec.items()
                       if v not in ("", None, [], {}) and not k.startswith("_")},
            "subtable_keys": [k for k in rec.keys() if k.startswith("_subtable")],
        })
    return {"count": len(out), "records": out}

@app.get("/ragic/scan-material", dependencies=[Depends(require_auth)])
def ragic_scan_material(samples: int = 25):
    """唯讀掃描：分析商品備註 1000632 的資材代碼格式分布。
    不修改任何資料，只回傳統計與樣本，供確認解析規則。"""
    if not RAGIC_API_KEY:
        return {"error": "未設定 RAGIC_API_KEY"}

    stats = {
        "total_products": 0,
        "with_note": 0,
        "parsed": 0,
        "unparsed": 0,
    }
    by_company: dict = {}
    code_lengths: dict = {}
    parsed_samples: list = []
    unparsed_samples: list = []
    dup_codes: dict = {}

    page, page_size, max_pages = 0, 1000, 10
    while page < max_pages:
        url = (f"{RAGIC_BASE_URL}{RAGIC_PRODUCT_PATH}?api&naming=EID"
               f"&subtables=0&limit={page * page_size},{page_size}")
        try:
            r = requests.get(url, headers=_ragic_headers(), timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "stats": stats}
        if not isinstance(data, dict):
            break
        rows = 0
        for rid, rec in data.items():
            if not str(rid).isdigit() or not isinstance(rec, dict):
                continue
            rows += 1
            stats["total_products"] += 1
            note = str(rec.get("1000632", "") or "").strip()
            if not note:
                continue
            stats["with_note"] += 1
            name = str(rec.get("1000617", "") or "").strip()
            company, code = _parse_material_note(note)
            if code:
                stats["parsed"] += 1
                by_company[company or "(未標公司)"] = by_company.get(company or "(未標公司)", 0) + 1
                code_lengths[len(code)] = code_lengths.get(len(code), 0) + 1
                key = f"{company}|{code}"
                dup_codes.setdefault(key, []).append(name)
                if len(parsed_samples) < samples:
                    parsed_samples.append(
                        {"note": note, "company": company, "code": code, "product": name})
            else:
                stats["unparsed"] += 1
                if len(unparsed_samples) < samples:
                    unparsed_samples.append({"note": note, "product": name})
        if rows < page_size:
            break
        page += 1

    duplicates = {k: v for k, v in dup_codes.items() if len(v) > 1}
    return {
        "stats": stats,
        "by_company": by_company,
        "code_length_distribution": code_lengths,
        "duplicate_codes": dict(list(duplicates.items())[:10]),
        "duplicate_count": len(duplicates),
        "parsed_samples": parsed_samples,
        "unparsed_samples": unparsed_samples,
    }

@app.get("/ragic/backfill-material", dependencies=[Depends(require_auth)])
def ragic_backfill_material(dry_run: int = 1, overwrite: int = 0):
    """把商品備註中的資材代碼回填到結構化欄位。
    - 1007109 合約品項-公司 (單選 長洲/尚鋒/靖展)
    - 1007110 資材代碼 (文字，保留前導 0)

    預設 dry_run=1 只預覽不寫入。dry_run=0 才真的寫。
    overwrite=0 時只填空欄位，不覆蓋已有值。
    """
    if not RAGIC_API_KEY:
        return {"error": "未設定 RAGIC_API_KEY"}

    plan, skipped = [], []
    page, page_size, max_pages = 0, 1000, 10
    while page < max_pages:
        url = (f"{RAGIC_BASE_URL}{RAGIC_PRODUCT_PATH}?api&naming=EID"
               f"&subtables=0&limit={page * page_size},{page_size}")
        try:
            r = requests.get(url, headers=_ragic_headers(), timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        if not isinstance(data, dict):
            break
        rows = 0
        for rid, rec in data.items():
            if not str(rid).isdigit() or not isinstance(rec, dict):
                continue
            rows += 1
            note = str(rec.get("1000632", "") or "").strip()
            if not note:
                continue
            company, code = _parse_material_note(note)
            if not code:
                continue
            cur_company = str(rec.get("1007109", "") or "").strip()
            cur_code = str(rec.get("1007110", "") or "").strip()
            if not overwrite and (cur_company or cur_code):
                skipped.append({"ragic_id": rid,
                                "product": str(rec.get("1000617", "") or "")[:40],
                                "reason": "欄位已有值",
                                "current": {"company": cur_company, "code": cur_code}})
                continue
            plan.append({
                "ragic_id": rid,
                "product": str(rec.get("1000617", "") or "")[:45],
                "note": note.replace("\r", " ").replace("\n", " ")[:45],
                "write_company": company,      # 沒解析到公司就留空 (使用者手動補)
                "write_code": code,
            })
        if rows < page_size:
            break
        page += 1

    result = {
        "dry_run": bool(dry_run),
        "planned": len(plan),
        "skipped": len(skipped),
        "no_company": sum(1 for p in plan if not p["write_company"]),
        "items": plan,
        "skipped_items": skipped[:20],
    }
    if dry_run:
        result["message"] = "預覽模式，未寫入任何資料。確認後用 ?dry_run=0 執行。"
        return result

    # 實際寫入
    ok, fail = 0, []
    headers = _ragic_headers({"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
    for item in plan:
        payload = {"1007110": item["write_code"]}
        if item["write_company"]:
            payload["1007109"] = item["write_company"]
        try:
            encoded = "&".join(
                f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in payload.items()
            ).encode("utf-8")
            rr = requests.post(f"{_ragic_url_for(RAGIC_PRODUCT_PATH)}/{item['ragic_id']}?api",
                               data=encoded, headers=headers, timeout=30)
            body = rr.json() if rr.status_code == 200 else {}
            if body.get("status") == "SUCCESS":
                ok += 1
            else:
                fail.append({"ragic_id": item["ragic_id"], "resp": str(body)[:150]})
        except Exception as e:
            fail.append({"ragic_id": item["ragic_id"], "error": f"{type(e).__name__}: {e}"})
    result["written"] = ok
    result["failed"] = len(fail)
    result["failures"] = fail[:10]
    result["message"] = f"已寫入 {ok} 筆，失敗 {len(fail)} 筆。"
    _ragic_load_products.cache_clear()
    return result

@app.get("/ragic/history", dependencies=[Depends(require_auth)])
def ragic_history(customer: Optional[str] = None, refresh: int = 0):
    """回傳客戶歷史訂購 {客戶編號: {品名: 次數}}，供商品模糊比對加權。
    ?customer=35 只回該客戶；?refresh=1 強制重抓。"""
    if refresh:
        _ragic_load_order_history.cache_clear()
    hist = _ragic_load_order_history().get("history", {})
    if customer:
        c = str(customer).strip()
        return {"customer": c, "items": hist.get(c, {}), "count": len(hist.get(c, {}))}
    return {
        "customers": len(hist),
        "total_items": sum(len(v) for v in hist.values()),
        "history": hist,
    }

@app.get("/ragic/test-write", dependencies=[Depends(require_auth)])
def ragic_test_write(ragic_po: Optional[str] = None, customer_code: Optional[str] = None):
    """測試寫入一筆最小資料到 Ragic 訂單總單，驗證權限與連線。
    寫入成功會立刻讀回 record 顯示載入欄位是否被自動填入。

    成功會在 Ragic 真的建立一筆紀錄，記得手動刪掉！
    參數：
    - ragic_po: 訂單編號 (1000320)；預設時間戳
    - customer_code: 客戶編號 (1000319)；預設 'TEST-CUSTOMER'，可傳真實如 '35'
    """
    out = {
        "enabled": RAGIC_ENABLED,
        "configured": bool(RAGIC_API_KEY),
        "url": _ragic_url(),
    }
    if not RAGIC_ENABLED:
        out["ok"] = False
        out["message"] = "RAGIC_ENABLED=false"
        return out
    if not RAGIC_API_KEY:
        out["ok"] = False
        out["message"] = "未設定 RAGIC_API_KEY"
        return out

    if not ragic_po:
        ragic_po = "TEST-" + datetime.now().strftime("%Y%m%d%H%M%S")
    if not customer_code:
        customer_code = "TEST-CUSTOMER"

    today = datetime.now().strftime("%Y/%m/%d")
    payload = {
        "1000320": ragic_po,                          # 訂單編號 (必填)
        "1000322": today,                             # 訂單日期 (必填)
        "1000319": customer_code,                     # 客戶編號 (必填，連結欄位)
        "1000654": "0.05",                            # 營業稅(選%) (必填，5%)
        "1000339": "OCR 系統測試寫入 — 確認後可刪除",  # 備註
    }
    out["request_payload"] = payload

    url = f"{_ragic_url()}?api"
    headers = _ragic_headers({"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
    try:
        encoded = "&".join(
            f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in payload.items()
        ).encode("utf-8")
        r = requests.post(url, data=encoded, headers=headers, timeout=30)
        out["http_status"] = r.status_code
        try:
            body = r.json()
            out["response"] = body
            if body.get("status") == "SUCCESS":
                rid = body.get("ragicId")
                out["ok"] = True
                out["ragic_id"] = rid
                out["view_url"] = f"{_ragic_url()}/{rid}"
                # 觸發 Ragic 重算載入欄位 (跟主流程相同的修法)
                _ragic_trigger_link_reload(rid, payload)
                out["message"] = f"✅ 寫入成功 (ragic_id={rid})，已觸發載入欄位重算"
                # 立刻讀回 record 看載入欄位是否被自動填
                try:
                    vr = requests.get(
                        f"{_ragic_url()}/{rid}?api&naming=EID",
                        headers=_ragic_headers(), timeout=10,
                    )
                    if vr.status_code == 200:
                        rec = vr.json()
                        if isinstance(rec, dict):
                            out["loaded_fields"] = {
                                "客戶編號_1000319": rec.get("1000319", ""),
                                "客戶類別_1000530": rec.get("1000530", ""),
                                "客戶名稱_1000321": rec.get("1000321", ""),
                                "客戶簡稱_1000414": rec.get("1000414", ""),
                                "客戶統編_1000415": rec.get("1000415", ""),
                                "電話_1000325": rec.get("1000325", ""),
                                "地址_1000329": rec.get("1000329", ""),
                                "送貨地址_1000330": rec.get("1000330", ""),
                                "負責業務_1006517": rec.get("1006517", ""),
                            }
                except Exception as e:
                    out["verify_error"] = f"{type(e).__name__}: {e}"
            else:
                out["ok"] = False
                out["message"] = f"❌ 失敗 (code {body.get('code')}): {body.get('msg', '')[:300]}"
        except Exception:
            out["ok"] = False
            out["response_text"] = r.text[:500]
            out["message"] = "❌ Ragic 回應非 JSON"
    except Exception as e:
        out["ok"] = False
        out["message"] = f"連線錯誤: {type(e).__name__}: {e}"
    return out

@app.get("/ragic/diag", dependencies=[Depends(require_auth)])
def ragic_diag():
    """診斷 Ragic 連線：GET 訂單總單前 1 筆，回傳連線狀態。"""
    out = {
        "enabled": RAGIC_ENABLED,
        "configured": bool(RAGIC_API_KEY),
        "url": _ragic_url(),
        "api_key_fingerprint": _api_key_fingerprint(RAGIC_API_KEY),
    }
    if not RAGIC_ENABLED:
        out["result"] = "RAGIC_ENABLED=false，跳過"
        return out
    if not RAGIC_API_KEY:
        out["result"] = "未設定 RAGIC_API_KEY"
        return out
    try:
        r = requests.get(
            f"{_ragic_url()}?api&naming=EID&listing=true&subtables=0&limit=0,1",
            headers=_ragic_headers(), timeout=15,
        )
        out["http_status"] = r.status_code
        if r.status_code != 200:
            out["result"] = "失敗"
            out["body_preview"] = r.text[:500]
            return out
        try:
            data = r.json()
            ids = _extract_record_ids(data)
            if isinstance(data, dict):
                out["response_top_keys"] = list(data.keys())[:5]
                # 若回的是錯誤格式 (含 status/msg/code)，把實際錯誤訊息吐出來
                if "status" in data and "msg" in data:
                    out["result"] = f"Ragic 回錯誤：{data.get('status')} (code {data.get('code')})"
                    out["ragic_error_msg"] = data.get("msg", "")[:500]
                else:
                    out["result"] = "成功"
                    out["record_count_returned"] = len(ids)
                    out["sample_record_id"] = ids[0] if ids else None
            else:
                out["result"] = "成功"
                out["record_count_returned"] = len(ids)
                out["sample_record_id"] = ids[0] if ids else None
        except Exception:
            out["result"] = "回應非 JSON (可能 API Key 錯誤或網址錯誤)"
            out["body_preview"] = r.text[:500]
    except Exception as e:
        out["result"] = f"連線錯誤: {type(e).__name__}: {e}"
    return out

@app.post("/login")
async def login(request: Request, response: Response):
    if not APP_PASSWORD:
        return {"ok": True, "message": "未啟用密碼保護"}
    try:
        body = await request.json()
        password = body.get("password", "")
    except Exception:
        raise HTTPException(status_code=400, detail="無效請求")
    if not hmac.compare_digest(password, APP_PASSWORD):
        raise HTTPException(status_code=401, detail="密碼錯誤")
    response.set_cookie(
        key=SESSION_COOKIE, value=SESSION_SECRET,
        httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7,
    )
    return {"ok": True}

@app.get("/auth/status")
def auth_status(session: Optional[str] = Cookie(None, alias=SESSION_COOKIE)):
    if not APP_PASSWORD:
        return {"required": False, "authenticated": True}
    authed = bool(session and hmac.compare_digest(session, SESSION_SECRET))
    return {"required": True, "authenticated": authed}

@app.post("/preview", dependencies=[Depends(require_auth)])
async def preview(files: list[UploadFile] = File(...)):
    """只做 OCR,不寫入 Sheet。每筆 order 附 _filename 方便 /submit 對應檔案。"""
    files = sorted(files, key=_natural_sort_key)
    results = []
    for file in files:
        try:
            content = await file.read()
            if len(content) > 20 * 1024 * 1024:
                raise ValueError("檔案超過 20MB")
            orders = extract_orders_from_file(content, file.filename)
            # 標記每筆 order 的來源檔案，submit 時用來找對應的 PDF 上傳到 Ragic
            for o in orders:
                o["_filename"] = file.filename
            results.append({
                "filename": file.filename,
                "status": "success",
                "orders": orders,
                "items_count": len(orders),
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e),
            })
    return {"results": results}

@app.post("/submit", dependencies=[Depends(require_auth)])
async def submit(
    orders: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
    """把預覽確認的資料寫入 Sheet → 再寫入 Ragic (含附件)。
    multipart/form-data: orders 欄位放 JSON 字串、files 放原始檔案 (可選)。"""
    try:
        orders_data = json.loads(orders)
    except Exception:
        raise HTTPException(status_code=400, detail="orders 欄位不是有效 JSON")
    if not orders_data:
        raise HTTPException(status_code=400, detail="沒有資料可寫入")

    # 收集檔案內容 (filename → bytes) 給 Ragic 附件上傳用
    file_map = {}
    for f in files:
        try:
            file_map[f.filename] = await f.read()
        except Exception:
            pass

    try:
        n = batch_write_to_sheet(orders_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"寫入 Sheet 失敗: {e}")
    ragic = write_orders_to_ragic(orders_data, file_map=file_map)
    return {"ok": True, "rows_written": n, "ragic": ragic}

@app.post("/upload", dependencies=[Depends(require_auth)])
async def upload_and_submit(files: list[UploadFile] = File(...)):
    """一步完成: OCR + 立即寫入 (含 Ragic 附件)"""
    files = sorted(files, key=_natural_sort_key)
    all_orders = []
    results = []
    file_map = {}
    for file in files:
        try:
            content = await file.read()
            if len(content) > 20 * 1024 * 1024:
                raise ValueError("檔案超過 20MB")
            file_map[file.filename] = content
            orders = extract_orders_from_file(content, file.filename)
            for o in orders:
                o["_filename"] = file.filename
            all_orders.extend(orders)
            results.append({
                "filename": file.filename,
                "status": "success",
                "items_count": len(orders),
                "orders": orders,
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e),
            })
    rows_written = 0
    ragic = None
    if all_orders:
        try:
            rows_written = batch_write_to_sheet(all_orders)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"results": results, "error": f"寫入 Sheet 失敗: {e}"},
            )
        ragic = write_orders_to_ragic(all_orders, file_map=file_map)
    return {"results": results, "rows_written": rows_written, "ragic": ragic}

# ------------------- 靜態檔案 -------------------
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
