"""
PerfDog 다운로더 MCP 서버 (최종 버전)
- 로그인 → csrf-token 자동 추출
- API로 Case ID 조회
- {디바이스명}_{Case명}.xlsx 형식으로 다운로드
"""

import asyncio
import json
import sys
from pathlib import Path
import logging

# Windows UTF-8 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import aiohttp
import aiofiles
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# 로깅
LOG_FILE = Path("C:/Users/kimjeonghyun/Desktop/설치/MCP/perfdog_downloader.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# 설정
BASE = "https://perfdog.wetest.net"
DOWN_DIR = Path("C:/Users/kimjeonghyun/Desktop/perfdog_downloads")
SESS_FILE = Path("C:/Users/kimjeonghyun/Desktop/설치/MCP/perfdog_session.json")

DOWN_DIR.mkdir(exist_ok=True)

server = Server("perfdog-downloader")


# ========================================
# 세션 관리
# ========================================
class Session:
    def __init__(self):
        self.cookies = {}
        self.token = ""
    
    def save(self, cookies, token=""):
        data = {"cookies": cookies, "csrf_token": token}
        SESS_FILE.write_text(json.dumps(data, indent=2), encoding='utf-8')
        self.cookies = cookies
        self.token = token
        log.info(f"[SESSION] ✅ 저장 ({len(cookies)} cookies)")
    
    def load(self):
        if not SESS_FILE.exists():
            log.info("[SESSION] 파일 없음")
            return False
        data = json.loads(SESS_FILE.read_text(encoding='utf-8'))
        self.cookies = data.get("cookies", {})
        self.token = data.get("csrf_token", "")
        log.info(f"[SESSION] ✅ 로드 ({len(self.cookies)} cookies)")
        return bool(self.cookies)

sess = Session()


# ========================================
# MCP 도구
# ========================================
@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="login",
            description="PerfDog 로그인 + csrf-token 자동 추출",
            inputSchema={
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "password": {"type": "string"}
                },
                "required": ["email", "password"]
            }
        ),
        types.Tool(
            name="check_session",
            description="세션 상태 확인",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="get_case_ids",
            description="Case ID 및 메타데이터 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_url": {"type": "string"}
                },
                "required": ["project_url"]
            }
        ),
        types.Tool(
            name="download_all",
            description="전체 다운로드 ({디바이스명}_{Case명}.xlsx 형식)",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_url": {"type": "string"},
                    "project_name": {"type": "string"}
                },
                "required": ["project_url", "project_name"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name, args):
    log.info(f"[TOOL] {name}")
    
    try:
        if name == "login":
            r = await login(args["email"], args["password"])
        elif name == "check_session":
            r = check()
        elif name == "get_case_ids":
            r = await get_ids(args["project_url"])
        elif name == "download_all":
            r = await download(args["project_url"], args["project_name"])
        else:
            r = {"status": "error", "msg": f"Unknown: {name}"}
        
        return [types.TextContent(type="text", text=json.dumps(r, ensure_ascii=False, indent=2))]
    
    except Exception as e:
        log.error(f"[ERROR] {e}", exc_info=True)
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "msg": str(e)}, ensure_ascii=False))]


# ========================================
# 세션 확인
# ========================================
def check():
    if not SESS_FILE.exists():
        return {"status": "success", "logged_in": False, "msg": "로그인 필요"}
    
    data = json.loads(SESS_FILE.read_text(encoding='utf-8'))
    cookies = data.get("cookies", {})
    token = data.get("csrf_token", "")
    
    return {
        "status": "success",
        "logged_in": True,
        "cookies": len(cookies),
        "has_csrf": bool(token),
        "csrf_preview": token[:15] + "..." if token else "None"
    }


# ========================================
# 로그인
# ========================================
async def login(email, pwd):
    log.info(f"[LOGIN] {email}")
    
    try:
        h = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0',
            'Origin': BASE,
            'Referer': f'{BASE}/login'
        }
        
        async with aiohttp.ClientSession() as s:
            # 1. 로그인
            log.info("[LOGIN] POST")
            async with s.post(f"{BASE}/account/email/login", json={"email": email, "password": pwd}, headers=h) as r:
                if r.status != 200:
                    txt = await r.text()
                    log.error(f"[LOGIN] 실패: {r.status}")
                    return {"status": "error", "msg": f"HTTP {r.status}", "details": txt[:200]}
                log.info("[LOGIN] ✅ 성공")
            
            # 2. csrf-token 가져오기
            log.info("[TOKEN] GET")
            async with s.get(f"{BASE}/taskdata", headers=h) as r:
                log.info(f"[TOKEN] {r.status}")
            
            # 3. 쿠키/토큰 추출
            cookies = {}
            token = ""
            for c in s.cookie_jar:
                cookies[c.key] = c.value
                if c.key == 'csrf-token':
                    token = c.value
            
            if not cookies:
                return {"status": "error", "msg": "No cookies"}
            
            # 4. 저장
            sess.save(cookies, token)
            
            return {
                "status": "success",
                "msg": "로그인 성공! csrf-token 저장됨",
                "cookies": len(cookies),
                "has_csrf": bool(token)
            }
    
    except Exception as e:
        log.error(f"[LOGIN ERROR] {e}", exc_info=True)
        return {"status": "error", "msg": str(e)}


# ========================================
# Case ID 조회
# ========================================
async def get_ids(url):
    log.info(f"[GET_IDS] {url}")
    
    try:
        task_id = url.split('/taskdata/')[-1].split('/')[0]
        log.info(f"[TASK_ID] {task_id}")
        
        if not sess.load():
            return {"status": "error", "msg": "로그인 필요"}
        
        if not sess.token:
            return {"status": "error", "msg": "csrf-token 없음"}
        
        h = {
            'x-csrf-token': sess.token,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0',
            'Referer': url
        }
        
        all_cases = []
        page = 1
        
        async with aiohttp.ClientSession(cookies=sess.cookies) as s:
            while True:
                payload = {
                    "taskId": int(task_id),
                    "pageSize": 100,
                    "pageNo": page,
                    "sortType": 0,
                    "caseType": 1
                }
                
                async with s.post(f"{BASE}/service/api/case/list", json=payload, headers=h) as r:
                    if r.status != 200:
                        return {"status": "error", "msg": f"HTTP {r.status}"}
                    
                    data = await r.json()
                    
                    if data.get("ret") != 0:
                        return {"status": "error", "msg": data.get("msg")}
                    
                    cases = data.get("data", {}).get("cases", [])
                    total = data.get("data", {}).get("count", 0)
                    
                    for c in cases:
                        cid = c.get("cid")
                        device = c.get("deviceModel", "Unknown")
                        case_name = c.get("caseName", f"case_{cid}")
                        
                        if cid:
                            all_cases.append({
                                "cid": str(cid),
                                "device": device,
                                "case_name": case_name
                            })
                    
                    log.info(f"[API] Page {page}: {len(cases)}/{total}")
                    
                    if len(all_cases) >= total or len(cases) < 100:
                        break
                    
                    page += 1
        
        log.info(f"[GET_IDS] ✅ {len(all_cases)} Cases")
        
        return {
            "status": "success",
            "total": len(all_cases),
            "cases": all_cases,
            "preview": [f"{c['device']}_{c['case_name']}" for c in all_cases[:5]]
        }
    
    except Exception as e:
        log.error(f"[GET_IDS ERROR] {e}", exc_info=True)
        return {"status": "error", "msg": str(e)}


# ========================================
# 다운로드
# ========================================
async def dl_one(s, case_info, folder):
    """단일 Case 다운로드"""
    try:
        cid = case_info["cid"]
        device = case_info["device"]
        case_name = case_info["case_name"]
        
        url = f"{BASE}/service/api/export/{cid}?hidelabels=0"
        
        async with s.get(url) as r:
            if r.status == 200:
                # 파일명: {디바이스}_{Case명}.xlsx
                safe_device = device.replace("/", "-").replace("\\", "-").replace(":", "-").replace("?", "").replace("*", "").replace('"', "").replace("<", "").replace(">", "").replace("|", "")
                safe_case = case_name.replace("/", "-").replace("\\", "-").replace(":", "-").replace("?", "").replace("*", "").replace('"', "").replace("<", "").replace(">", "").replace("|", "")
                
                filename = f"{safe_device}_{safe_case}.xlsx"
                file = folder / filename
                
                content = await r.read()
                
                async with aiofiles.open(file, 'wb') as f:
                    await f.write(content)
                
                kb = len(content) / 1024
                log.info(f"[DL] ✅ {filename} ({kb:.1f}KB)")
                return {"cid": cid, "ok": True, "kb": round(kb, 1), "filename": filename}
            else:
                log.warning(f"[DL] ❌ {cid}: HTTP {r.status}")
                return {"cid": cid, "ok": False}
    
    except Exception as e:
        log.error(f"[DL ERROR] {cid}: {e}")
        return {"cid": cid, "ok": False}


async def download(url, name):
    """전체 다운로드"""
    log.info(f"[DOWNLOAD] {name}")
    
    try:
        # 1. Case 메타데이터 조회
        cases_r = await get_ids(url)
        if cases_r["status"] != "success":
            return cases_r
        
        cases = cases_r["cases"]
        if not cases:
            return {"status": "error", "msg": "No cases"}
        
        log.info(f"[DOWNLOAD] {len(cases)} cases 다운로드 시작")
        
        # 2. 폴더 생성
        folder = DOWN_DIR / name
        folder.mkdir(exist_ok=True)
        
        # 3. 세션 확인
        if not sess.load():
            return {"status": "error", "msg": "로그인 필요"}
        
        # 4. 헤더 설정 (403 에러 방지)
        h = {
            'x-csrf-token': sess.token,
            'Accept': 'application/json, text/plain, */*',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': url,
            'Origin': BASE
        }
        
        # 5. 다운로드 (배치 처리)
        results = []
        
        async with aiohttp.ClientSession(cookies=sess.cookies, headers=h) as s:
            batch = 10
            for i in range(0, len(cases), batch):
                b = cases[i:i+batch]
                tasks = [dl_one(s, case_info, folder) for case_info in b]
                rs = await asyncio.gather(*tasks)
                results.extend(rs)
                log.info(f"[PROGRESS] {len(results)}/{len(cases)}")
                await asyncio.sleep(0.5)  # Rate limit 방지
        
        # 6. 결과
        ok = [r for r in results if r.get("ok")]
        fail = [r for r in results if not r.get("ok")]
        size = sum(r.get("kb", 0) for r in ok)
        
        log.info(f"[DOWNLOAD] ✅ 완료: {len(ok)}/{len(cases)}")
        
        return {
            "status": "success",
            "project": name,
            "total": len(cases),
            "downloaded": len(ok),
            "failed": len(fail),
            "folder": str(folder),
            "size_mb": round(size / 1024, 1),
            "sample_files": [r.get("filename") for r in ok[:5]]
        }
    
    except Exception as e:
        log.error(f"[DOWNLOAD ERROR] {e}", exc_info=True)
        return {"status": "error", "msg": str(e)}


# ========================================
# 서버 시작
# ========================================
async def main():
    log.info("=" * 60)
    log.info("PerfDog Downloader MCP Server")
    log.info(f"Download dir: {DOWN_DIR}")
    log.info(f"Session file: {SESS_FILE}")
    log.info("=" * 60)
    
    if SESS_FILE.exists():
        log.info("✅ 세션 파일 확인됨")
    else:
        log.info("⚠️  세션 없음 - 로그인 필요")
    
    try:
        async with stdio_server() as (read, write):
            log.info("✅ MCP server ready")
            log.info("Waiting for Claude Desktop...")
            
            await server.run(read, write, server.create_initialization_options())
    
    except Exception as e:
        log.error(f"❌ Server error: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("\n👋 Server stopped")
    except Exception as e:
        log.error(f"❌ Fatal error: {e}", exc_info=True)