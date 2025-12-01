"""
Tableau CSV Converter MCP 서버 (D1 패키지명 포함)
- Platform 기반 CPU 추출 (파일명 의존성 제거)
- Device Type, Product Type, Platform 자동 추출
- D1 셀 패키지명을 파일명에 포함 (예: AP196#1_com.pubg.krmobile)
- 프로젝트명 기반 파일명: fps_{project_name}.csv, summary_{project_name}.csv
"""

import asyncio
import json
import sys
import os
from pathlib import Path
import logging

# Windows UTF-8 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# 로깅
LOG_FILE = Path("C:/Users/kimjeonghyun/Desktop/설치/MCP/tableau_converter.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

server = Server("tableau-converter")

# ========================================
# 스키마 정의
# ========================================
EXPECTED_SUMMARY_COLS = [
    "Filename","Measure Date","Project Name","Version",
    "Device Type","Product Type","Platform",
    "Avg FPS","Median FPS","Max FPS","1% Low FPS",
    "MedRange(FPS)[%]","Stability (%)","Std(FPS)","Smooth Index","Drop FPS (/h)",
    "Jank(/10min)","BigJank(/10min)",
    "Avg Memory [MB]","Peak Memory [MB]",
    "Avg CPU [%]","Max CPU [%]",
]

EXPECTED_TS_COLS = [
    "TimeSec","FPS","Smooth","1%Low(FPS)","Stutter[%]",
    "AppCPU[%]","TotalCPU[%]","Memory[MB]","AvailableMemory[MB]",
    "BigJank","Jank","SwapMemory[MB]","GL[MB]","Gfx[MB]","EGL mtrack[MB]",
    "BTemp[℃]","GTemp[℃]","ScreenShot",
    "Filename","Version","Measure Date","Project Name","Processing",
    "Device Type","Product Type","Platform",
    "Recv[KB/s]","Send[KB/s]","Render[%]","Tiler[%]",
    "CPUEnergy","GPUEnergy","DisplayEnergy","OverheadEnergy","NetworkEnergy",
    "CSwitch"
]


# ========================================
# 핵심 추출 함수들
# ========================================
def extract_package_name(df_all):
    """D1 셀에서 패키지명 추출"""
    try:
        d1_value = df_all.iloc[0, 3]  # D1 = 0행 3열
        if pd.notna(d1_value):
            package_name = str(d1_value).strip()
            if package_name and package_name.lower() not in ['nan', 'none', '']:
                return package_name
    except Exception as e:
        log.warning(f"D1 셀 패키지명 추출 실패: {e}")
    return None


def extract_device_info(df_all):
    """DeviceInfo 섹션에서 Device Type과 Product Type 추출"""
    device_info = {
        "Device Type": None,
        "Product Type": None,
        "Platform": None
    }
    
    deviceinfo_row = None
    for i in range(min(30, len(df_all))):
        if str(df_all.iat[i, 0]).strip().lower() == "deviceinfo":
            deviceinfo_row = i
            break
    
    if deviceinfo_row is None:
        return device_info
    
    try:
        headers = df_all.iloc[deviceinfo_row + 1].tolist()
        values = df_all.iloc[deviceinfo_row + 2].tolist()
        
        for idx, header in enumerate(headers):
            header_str = str(header).strip()
            
            if header_str == "Device Type" and idx < len(values):
                device_type = str(values[idx]).strip()
                if device_type not in ["nan", "None", "", "unknown"]:
                    device_info["Device Type"] = device_type
            
            elif header_str == "Product Type" and idx < len(values):
                product_type = str(values[idx]).strip()
                if product_type not in ["nan", "None", "", "unknown"]:
                    device_info["Product Type"] = product_type
        
        # Platform 판단
        if device_info["Product Type"]:
            device_info["Platform"] = "iOS"
        elif device_info["Device Type"]:
            device_info["Platform"] = "Android"
            
    except Exception as e:
        log.warning(f"Device Info 추출 오류: {e}")
    
    return device_info


def extract_summary_cpu_and_fps_summary_row(path):
    """Summary 데이터 추출 - Platform 기반 CPU 추출 + D1 패키지명 포함"""
    df_all = pd.read_excel(path, sheet_name="all", header=None)
    device_info = extract_device_info(df_all)
    package_name = extract_package_name(df_all)

    def _find_appcpu_values(df):
        """AppCPU[%] 컬럼에서 Avg, Max 추출 (Normalized 제외)"""
        appcpu_col = None
        for c in range(df.shape[1]):
            for r in range(df.shape[0]):
                v = str(df.iat[r, c]).replace("\n", "").replace(" ", "").lower()
                if "appcpu" in v and "%" in v and "normalized" not in v:
                    appcpu_col = c
                    break
            if appcpu_col is not None:
                break

        avg_val = max_val = None
        if appcpu_col is not None:
            for r in range(df.shape[0]):
                label = str(df.iat[r, 0]).strip().lower()
                try:
                    if avg_val is None and label.startswith("avg"):
                        avg_val = float(df.iat[r, appcpu_col])
                    elif max_val is None and label.startswith("max"):
                        max_val = float(df.iat[r, appcpu_col])
                except Exception:
                    continue
                if avg_val is not None and max_val is not None:
                    break
        return avg_val, max_val

    def _find_normalized_appcpu_values(df):
        """AppCPU[%] (Normalized) 컬럼에서 Avg, Max 추출"""
        norm_col = None
        for c in range(df.shape[1]):
            for r in range(df.shape[0]):
                v = str(df.iat[r, c]).replace("\n", "").replace(" ", "").lower()
                if "appcpu" in v and "normalized" in v:
                    norm_col = c
                    break
            if norm_col is not None:
                break

        avg_val = max_val = None
        if norm_col is not None:
            for r in range(df.shape[0]):
                label = str(df.iat[r, 0]).strip().lower()
                try:
                    if avg_val is None and label.startswith("avg"):
                        avg_val = float(df.iat[r, norm_col])
                    elif max_val is None and label.startswith("max"):
                        max_val = float(df.iat[r, norm_col])
                except Exception:
                    continue
                if avg_val is not None and max_val is not None:
                    break
        return avg_val, max_val

    def _find_cpu_core_count(df):
        """CPU CoreNum 찾기"""
        core = None
        for c in range(df.shape[1]):
            for r in range(df.shape[0]):
                if str(df.iat[r, c]).strip().lower() == "cpu corenum":
                    try:
                        core = float(df.iat[r + 1, c])
                    except Exception:
                        core = None
                    break
            if core is not None:
                break
        return core

    # 메타데이터
    measure_date = str(df_all.iloc[0, 0]).strip()
    project_name = str(df_all.iloc[0, 1]).strip()
    version = str(df_all.iloc[0, 2]).strip()
    
    # 파일명 생성: Case명_패키지명 (예: "IP036#1_com.pubg.krmobile")
    full_filename = os.path.basename(path).replace(".xlsx", "")
    if "_" in full_filename:
        # 마지막 underscore 이후만 추출 (Case명)
        case_name = full_filename.split("_")[-1]
    else:
        case_name = full_filename
    
    # 패키지명이 있으면 추가
    if package_name:
        filename = f"{case_name}_{package_name}"
    else:
        filename = case_name

    metrics = {
        "Filename": filename, 
        "Measure Date": measure_date,
        "Project Name": project_name, 
        "Version": version,
        "Device Type": device_info["Device Type"],
        "Product Type": device_info["Product Type"],
        "Platform": device_info["Platform"]
    }

    # FPS 메트릭
    summary_row = None
    for i in range(len(df_all)):
        if str(df_all.iat[i, 0]).strip().lower() == "summary":
            summary_row = i
            break
    
    if summary_row is not None:
        fps_col = None
        for c in range(df_all.shape[1]):
            if str(df_all.iat[summary_row, c]).strip().lower() == "fps":
                fps_col = c
                break
        if fps_col is not None:
            try:
                metrics["Avg FPS"] = float(df_all.iat[summary_row + 1, fps_col])
                metrics["Max FPS"] = float(df_all.iat[summary_row + 2, fps_col])
            except Exception:
                pass

    # 라벨 매핑
    mapping = {
        "Median(FPS)": "Median FPS",
        "1%Low(FPS)": "1% Low FPS",
        "MedRange(FPS)[%]": "MedRange(FPS)[%]",
        "Std(FPS)": "Std(FPS)",
        "Jank(/10min)": "Jank(/10min)",
        "BigJank(/10min)": "BigJank(/10min)",
        "Smooth": "Smooth Index",
        "Drop(FPS)[/h]": "Drop FPS (/h)",
        "Avg(Memory)[MB]": "Avg Memory [MB]",
        "Peak(Memory)[MB]": "Peak Memory [MB]",
    }
    
    str_df = df_all.applymap(str)
    for key, field in mapping.items():
        m = str_df.eq(key)
        if m.any().any():
            r, c = m.stack()[lambda x: x].index[0]
            try:
                v = float(df_all.iat[r + 1, c])
                if field not in ["1% Low FPS","Smooth Index"]: 
                    v = round(v, 2)
                metrics[field] = v
            except Exception:
                pass

    # ========================================
    # CPU 메트릭 (Platform 기반)
    # ========================================
    platform = device_info.get("Platform")

    if platform == "iOS":
        # iOS: AppCPU * Core 수
        avg_val, max_val = _find_appcpu_values(df_all)
        core = _find_cpu_core_count(df_all)
        
        if core and avg_val is not None:
            metrics["Avg CPU [%]"] = round(avg_val * core, 2)
        if core and max_val is not None:
            metrics["Max CPU [%]"] = round(max_val * core, 2)

    elif platform == "Android":
        # Android: Normalized AppCPU 우선, 없으면 AppCPU * Core
        norm_avg, norm_max = _find_normalized_appcpu_values(df_all)
        
        if norm_avg is not None:
            # Normalized 값이 있으면 그대로 사용
            metrics["Avg CPU [%]"] = round(norm_avg, 2)
            metrics["Max CPU [%]"] = round(norm_max, 2) if norm_max is not None else None
        else:
            # Normalized 없으면 AppCPU * Core
            avg_val, max_val = _find_appcpu_values(df_all)
            core = _find_cpu_core_count(df_all)
            
            if core and avg_val is not None:
                metrics["Avg CPU [%]"] = round(avg_val * core, 2)
            if core and max_val is not None:
                metrics["Max CPU [%]"] = round(max_val * core, 2)

    # 누락 컬럼 기본값
    for col in EXPECTED_SUMMARY_COLS:
        metrics.setdefault(col, None)
    
    return pd.DataFrame([metrics])[EXPECTED_SUMMARY_COLS]


def extract_timeseries_cleaned(path):
    """TimeSeries 데이터 추출 - D1 패키지명 포함"""
    df_raw = pd.read_excel(path, sheet_name="all", header=None)
    device_info = extract_device_info(df_raw)
    package_name = extract_package_name(df_raw)
    
    header_row_index = None
    for i in range(min(60, len(df_raw))):
        row_values = set(str(v).strip() for v in df_raw.iloc[i])
        if {"time","FPS","AppCPU[%]","Memory[MB]"}.intersection(row_values):
            header_row_index = i
            break
    
    if header_row_index is None:
        raise ValueError("시계열 헤더 행을 찾지 못했습니다.")

    df = pd.read_excel(path, sheet_name="all", header=header_row_index)

    if "time" in df.columns:
        df["TimeSec"] = pd.to_numeric(df["time"], errors="coerce") / 1000.0

    meta = pd.read_excel(path, sheet_name="all", header=None, nrows=1, usecols="A:C")
    measure_date = str(meta.iloc[0, 0]).strip()
    project_name = str(meta.iloc[0, 1]).strip()
    version      = str(meta.iloc[0, 2]).strip()
    
    # 파일명 생성: Case명_패키지명 (예: "IP036#1_com.pubg.krmobile")
    full_filename = os.path.basename(path).replace(".xlsx", "")
    if "_" in full_filename:
        case_name = full_filename.split("_")[-1]
    else:
        case_name = full_filename
    
    # 패키지명이 있으면 추가
    if package_name:
        filename = f"{case_name}_{package_name}"
    else:
        filename = case_name
    
    processing = os.path.basename(path)

    out = pd.DataFrame()
    for col in EXPECTED_TS_COLS:
        if col in {"Filename","Version","Measure Date","Project Name","Processing",
                   "Device Type","Product Type","Platform"}:
            continue
        out[col] = df[col] if col in df.columns else pd.NA

    keep = [c for c in ["TimeSec","FPS"] if c in out.columns]
    if keep:
        out = out.dropna(how="all", subset=keep)
    out = out.reset_index(drop=True)

    n = len(out)
    out["Filename"]     = [filename] * n
    out["Version"]      = [version] * n
    out["Measure Date"] = [measure_date] * n
    out["Project Name"] = [project_name] * n
    out["Processing"]   = [processing] * n
    out["Device Type"]  = [device_info["Device Type"]] * n
    out["Product Type"] = [device_info["Product Type"]] * n
    out["Platform"]     = [device_info["Platform"]] * n

    out = out[EXPECTED_TS_COLS]
    return out


# ========================================
# MCP 도구
# ========================================
@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="convert_to_tableau",
            description="PerfDog Excel 파일들을 Tableau용 CSV로 변환 (Summary + TimeSeries). 저장 위치와 프로젝트명을 반드시 지정해야 합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "excel_folder": {
                        "type": "string",
                        "description": "PerfDog Excel 파일들이 있는 폴더 경로"
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "CSV 파일을 저장할 폴더 경로 (필수 - 매번 다른 위치 지정 가능)"
                    },
                    "project_name": {
                        "type": "string",
                        "description": "프로젝트명 (파일명에 사용됨: fps_{project_name}.csv, summary_{project_name}.csv)"
                    },
                    "summary_filename": {
                        "type": "string",
                        "description": "Summary CSV 파일명 (선택 - 미지정시 summary_{project_name}.csv 사용)"
                    },
                    "timeseries_filename": {
                        "type": "string",
                        "description": "TimeSeries CSV 파일명 (선택 - 미지정시 fps_{project_name}.csv 사용)"
                    }
                },
                "required": ["excel_folder", "output_folder", "project_name"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name, args):
    log.info(f"[TOOL] {name}")
    
    try:
        if name == "convert_to_tableau":
            project_name = args["project_name"]
            
            # 기본 파일명
            summary_filename = args.get("summary_filename", f"summary_{project_name}.csv")
            timeseries_filename = args.get("timeseries_filename", f"fps_{project_name}.csv")
            
            result = await convert_to_tableau(
                args["excel_folder"],
                args["output_folder"],
                summary_filename,
                timeseries_filename,
                project_name
            )
        else:
            result = {"status": "error", "msg": f"Unknown tool: {name}"}
        
        return [types.TextContent(
            type="text", 
            text=json.dumps(result, ensure_ascii=False, indent=2)
        )]
    
    except Exception as e:
        log.error(f"[ERROR] {e}", exc_info=True)
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "msg": str(e)}, ensure_ascii=False)
        )]


async def convert_to_tableau(
    excel_folder,
    output_folder,
    summary_filename,
    timeseries_filename,
    project_name
):
    """Excel → CSV 변환 메인 로직"""
    log.info(f"[CONVERT] 시작")
    log.info(f"  프로젝트: {project_name}")
    log.info(f"  Excel 폴더: {excel_folder}")
    log.info(f"  출력 폴더: {output_folder}")
    
    try:
        excel_path = Path(excel_folder)
        output_path = Path(output_folder)
        
        if not excel_path.exists():
            return {"status": "error", "msg": f"Excel 폴더 없음: {excel_folder}"}
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        excel_files = [f for f in excel_path.glob("*.xlsx") if not f.name.startswith("~$")]
        excel_files.sort(key=lambda p: p.name.lower())
        
        if not excel_files:
            return {"status": "error", "msg": "Excel 파일 없음"}
        
        log.info(f"[CONVERT] {len(excel_files)}개 파일 발견")
        
        summary_all = []
        ts_all = []
        errors = []
        
        for i, file_path in enumerate(excel_files, start=1):
            filename = file_path.name
            log.info(f"[{i}/{len(excel_files)}] 처리 중: {filename}")
            
            try:
                df_summary = extract_summary_cpu_and_fps_summary_row(str(file_path))
                df_ts = extract_timeseries_cleaned(str(file_path))
                
                summary_all.append(df_summary)
                ts_all.append(df_ts)
                
                log.info(f"  ✅ {filename}")
            
            except Exception as e:
                log.error(f"  ❌ {filename}: {e}")
                errors.append({"file": filename, "error": str(e)})
        
        # CSV 저장
        summary_path = output_path / summary_filename
        ts_path = output_path / timeseries_filename
        
        summary_rows = 0
        timeseries_rows = 0
        
        if summary_all:
            df_summary = pd.concat(summary_all, ignore_index=True)
            for col in EXPECTED_SUMMARY_COLS:
                if col not in df_summary.columns:
                    df_summary[col] = pd.NA
            df_summary = df_summary[EXPECTED_SUMMARY_COLS]
            df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
            summary_rows = len(df_summary)
            log.info(f"[SAVE] ✅ Summary: {summary_path} ({summary_rows}행)")
        
        if ts_all:
            df_ts_all = pd.concat(ts_all, ignore_index=True)
            for col in EXPECTED_TS_COLS:
                if col not in df_ts_all.columns:
                    df_ts_all[col] = pd.NA
            df_ts_all = df_ts_all[EXPECTED_TS_COLS]
            df_ts_all.to_csv(ts_path, index=False, encoding="utf-8-sig")
            timeseries_rows = len(df_ts_all)
            log.info(f"[SAVE] ✅ TimeSeries: {ts_path} ({timeseries_rows}행)")
        
        result = {
            "status": "success",
            "project_name": project_name,
            "total_files": len(excel_files),
            "processed": len(summary_all),
            "failed": len(errors),
            "summary_csv": str(summary_path) if summary_all else None,
            "timeseries_csv": str(ts_path) if ts_all else None,
            "summary_rows": summary_rows,
            "timeseries_rows": timeseries_rows,
            "output_folder": str(output_path)
        }
        
        if errors:
            result["errors"] = errors[:5]
        
        log.info(f"[CONVERT] ✅ 완료: {len(summary_all)}/{len(excel_files)}")
        
        return result
    
    except Exception as e:
        log.error(f"[CONVERT ERROR] {e}", exc_info=True)
        return {"status": "error", "msg": str(e)}


# ========================================
# 서버 시작
# ========================================
async def main():
    log.info("=" * 60)
    log.info("Tableau CSV Converter MCP Server (D1 패키지명 포함)")
    log.info("Platform 기반 CPU 추출")
    log.info("=" * 60)
    
    async with stdio_server() as (read, write):
        log.info("✅ MCP server ready")
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())