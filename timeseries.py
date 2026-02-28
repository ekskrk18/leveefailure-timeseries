import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 경로 설정
# =========================
BASE_DIR = Path(r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\24th Urban Flood (Timeseries)\python")

RAIN_DIR = BASE_DIR / "kma_rainfall_rn_60m"
SMAP_DIR = BASE_DIR / "smap_L4_surface_rootzone"

# 파일명은 예시 기준 (필요 시 수정)
SMAP_ALL = SMAP_DIR / "smap_L4_sm_all_events.csv"

OUT_DIR = BASE_DIR / "plots_timeseries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 시간 파싱 유틸
# =========================
def parse_kst_no_tz(series: pd.Series) -> pd.Series:
    """
    '2020-07-09 22:30:00+09:00' 또는 '...+0900' 같은 tz suffix를 제거하고
    naive datetime (KST 기준)로 파싱
    """
    s = series.astype(str)

    # +09:00 또는 +0900 같은 tz 정보 제거
    s = s.str.replace(r"\+\d{2}:\d{2}$", "", regex=True)
    s = s.str.replace(r"\+\d{4}$", "", regex=True)

    return pd.to_datetime(s, errors="coerce")

# =========================
# 1) Rain CSV 전체 로드(이벤트별 파일 합치기)
# =========================
rain_files = sorted(RAIN_DIR.glob("rain_rn_60m_event_*.csv"))
if not rain_files:
    raise FileNotFoundError(f"Rain 파일을 찾지 못했습니다: {RAIN_DIR}")

rain_list = []
for fp in rain_files:
    df = pd.read_csv(fp)
    # 컬럼 확인: id, event_time_kst, tm_kst, rn_60m_mm
    df["tm_kst"] = pd.to_datetime(df["tm_kst"], errors="coerce")  # tz 없음
    df["event_time_kst_dt"] = pd.to_datetime(df["event_time_kst"], errors="coerce")
    rain_list.append(df)

rain = pd.concat(rain_list, ignore_index=True)
rain = rain.dropna(subset=["id", "tm_kst"])

# =========================
# 2) SMAP all-events 로드
# =========================
smap = pd.read_csv(SMAP_ALL)

# time_kst는 +09:00이 붙어있으니 제거 후 파싱
smap["time_kst_dt"] = parse_kst_no_tz(smap["time_kst"])
# event_time_kst도 +0900 같은 게 붙어있을 수 있어 제거 후 파싱
smap["event_time_kst_dt"] = parse_kst_no_tz(smap["event_time_kst"])

smap = smap.dropna(subset=["id", "time_kst_dt"])

# =========================
# 3) 이벤트별 플로팅
# =========================
event_ids = sorted(set(rain["id"].unique()).union(set(smap["id"].unique())))

for eid in event_ids:
    rain_e = rain[rain["id"] == eid].copy()
    smap_e = smap[smap["id"] == eid].copy()

    if rain_e.empty and smap_e.empty:
        continue

    # 사고 시점(event_time): rain 쪽이든 smap 쪽이든 있는 값 사용
    event_time = None
    if not rain_e.empty and rain_e["event_time_kst_dt"].notna().any():
        event_time = rain_e["event_time_kst_dt"].dropna().iloc[0]
    elif not smap_e.empty and smap_e["event_time_kst_dt"].notna().any():
        event_time = smap_e["event_time_kst_dt"].dropna().iloc[0]

    if event_time is None:
        print(f"[Event {eid}] event_time 파싱 실패 -> 스킵")
        continue

    # 14일 전 ~ 1일 후 윈도우 (CSV가 이미 그 윈도우로 만들어졌지만 안전하게 한번 더 필터)
    t_start = event_time - pd.Timedelta(days=14)
    t_end = event_time + pd.Timedelta(days=1)

    rain_e = rain_e[(rain_e["tm_kst"] >= t_start) & (rain_e["tm_kst"] <= t_end)]
    smap_e = smap_e[(smap_e["time_kst_dt"] >= t_start) & (smap_e["time_kst_dt"] <= t_end)]

    # 정렬
    rain_e = rain_e.sort_values("tm_kst")
    smap_e = smap_e.sort_values("time_kst_dt")

    # Figure (3개 패널, x축 공유)
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]}
    )

    # --- (1) Rainfall (bar)
    ax0 = axes[0]
    if not rain_e.empty:
        ax0.bar(rain_e["tm_kst"], rain_e["rn_60m_mm"], width=0.03)  # width는 datetime 단위에서 대략치
    ax0.axvline(event_time, linestyle="--", linewidth=1.5, color="red")
    ax0.set_ylabel("Rain (mm/60m)")
    ax0.set_title(f"Event {eid} | {event_time.strftime('%Y-%m-%d %H:%M')} (KST)")

    # --- (2) Surface SM
    ax1 = axes[1]
    if not smap_e.empty:
        ax1.plot(smap_e["time_kst_dt"], smap_e["sm_surface"], marker="o", markersize=3, linewidth=1)
    ax1.axvline(event_time, linestyle="--", linewidth=1.5, color="red")
    ax1.set_ylabel("SM surface")

    # --- (3) Root-zone SM
    ax2 = axes[2]
    if not smap_e.empty:
        ax2.plot(smap_e["time_kst_dt"], smap_e["sm_rootzone"], marker="o", markersize=3, linewidth=1)
    ax2.axvline(event_time, linestyle="--", linewidth=1.5, color="red")
    ax2.set_ylabel("SM rootzone")
    ax2.set_xlabel("Time (KST)")

    # x축 범위 고정
    ax2.set_xlim(t_start, t_end)

    plt.tight_layout()

    out_png = OUT_DIR / f"event_{eid}_rain_smap_timeseries.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[Saved] {out_png}")

print(f"\nDone. Outputs in: {OUT_DIR}")
