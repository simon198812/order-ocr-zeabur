"""OCR 訂單處理系統 - FastAPI 後端 v2.1

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
"""
import os
import io
import json
import base64
import hmac
import re
import secrets
import tempfile
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Cookie, Response
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
6. note 欄位放保固月數、備註、其他補充資訊

欄位格式規則（請嚴格遵守）:
- po_no (訂單號): 固定格式為大寫字母 M 開頭 + 9 位數字,共 10 碼,例如 M123456789
  圖片中可能因換行或空格拆成兩行,請合併為完整值。若找不到符合格式,填 ""
- ext (分機): 純數字 4~5 碼,例如 1234 或 12345
  圖片中可能因換行分兩段,請只取數字合併後填入。若找不到,填 ""
- 金額欄位 (price, total): 只填數字,不含貨幣符號 ($ NT 等)

請依照 JSON schema 輸出,orders 陣列中每個品項一筆資料。
"""

# ------------------- 結構化輸出 Schema -------------------
_ORDER_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id":     {"type": "string", "description": "訂單號-項次 (例如 M150101275-1)"},
        "po_no":       {"type": "string", "description": "訂單號 M+9碼"},
        "type":        {"type": "string", "description": "一般訂單 或 合約品項"},
        "hospital":    {"type": "string", "description": "開單醫院"},
        "date":        {"type": "string", "description": "訂購日期 (民國年)"},
        "vendor":      {"type": "string", "description": "廠商名稱"},
        "loc":         {"type": "string", "description": "交貨地點"},
        "item":        {"type": "string", "description": "品名"},
        "qty":         {"type": "number", "description": "訂購量/數量"},
        "unit":        {"type": "string", "description": "單位 (個/台/箱)"},
        "price":       {"type": "number", "description": "單價"},
        "total":       {"type": "number", "description": "訂購總金額"},
        "brand":       {"type": "string", "description": "廠牌型號"},
        "deadline":    {"type": "string", "description": "交貨期限"},
        "dept":        {"type": "string", "description": "請購部門"},
        "user":        {"type": "string", "description": "申購人"},
        "ext":         {"type": "string", "description": "分機 4~5碼"},
        "sign_no":     {"type": "string", "description": "簽辦案號"},
        "material_id": {"type": "string", "description": "資材代碼 (合約品項才有)"},
        "note":        {"type": "string", "description": "備註/保固月數"},
    },
    "required": [
        "item_id", "po_no", "type", "hospital", "date", "vendor", "loc",
        "item", "qty", "unit", "price", "total", "brand", "deadline",
        "dept", "user", "ext", "sign_no", "material_id", "note",
    ],
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "orders": {"type": "array", "items": _ORDER_ITEM_SCHEMA}
    },
    "required": ["orders"],
}

# Sheet 欄位順序
SHEET_COLUMNS = [
    "hospital", "date", "vendor", "loc", "item", "qty", "price", "total",
    "brand", "deadline", "dept", "user", "ext", "note", "po_no", "item_id",
    "idx", "sign_no", "type", "material_id", "unit",
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
    """分機: 4~5位純數字，處理換行拆分"""
    d = re.sub(r'\D', '', str(v))
    if re.fullmatch(r'\d{4,5}', d):
        return d
    m = re.search(r'\d{4,5}', re.sub(r'\s+', '', str(v)))
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
    return o

def _natural_sort_key(f):
    """1.jpg < 2.jpg < 10.jpg 自然排序"""
    pts = re.split(r'(\d+)', (f.filename or '').lower())
    return [int(p) if p.isdigit() else p for p in pts]

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
        ),
    )

    data = json.loads(response.text)
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

# ------------------- 密碼驗證 -------------------
def require_auth(session: Optional[str] = Cookie(None, alias=SESSION_COOKIE)):
    if not APP_PASSWORD:
        return
    if not session or not hmac.compare_digest(session, SESSION_SECRET):
        raise HTTPException(status_code=401, detail="需要登入")

# ------------------- FastAPI App -------------------
app = FastAPI(title="OCR 訂單處理系統", version="2.1.0")
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
        "password_protected": bool(APP_PASSWORD),
        "model": GEMINI_MODEL,
    }

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
    """只做 OCR,不寫入 Sheet"""
    files = sorted(files, key=_natural_sort_key)
    results = []
    for file in files:
        try:
            content = await file.read()
            if len(content) > 20 * 1024 * 1024:
                raise ValueError("檔案超過 20MB")
            orders = extract_orders_from_file(content, file.filename)
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
async def submit(request: Request):
    """把預覽確認的資料寫入 Sheet"""
    try:
        body = await request.json()
        orders = body.get("orders", [])
    except Exception:
        raise HTTPException(status_code=400, detail="無效的 JSON")
    if not orders:
        raise HTTPException(status_code=400, detail="沒有資料可寫入")
    try:
        n = batch_write_to_sheet(orders)
        return {"ok": True, "rows_written": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"寫入 Sheet 失敗: {e}")

@app.post("/upload", dependencies=[Depends(require_auth)])
async def upload_and_submit(files: list[UploadFile] = File(...)):
    """一步完成: OCR + 立即寫入"""
    files = sorted(files, key=_natural_sort_key)
    all_orders = []
    results = []
    for file in files:
        try:
            content = await file.read()
            if len(content) > 20 * 1024 * 1024:
                raise ValueError("檔案超過 20MB")
            orders = extract_orders_from_file(content, file.filename)
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
    if all_orders:
        try:
            rows_written = batch_write_to_sheet(all_orders)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"results": results, "error": f"寫入 Sheet 失敗: {e}"},
            )
    return {"results": results, "rows_written": rows_written}

# ------------------- 靜態檔案 -------------------
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
