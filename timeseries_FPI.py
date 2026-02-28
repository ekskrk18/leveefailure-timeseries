from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 경로 설정
# =========================================================
BASE_DIR = Path(r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\24th Urban Flood (Timeseries)\python")
RAIN_DIR = BASE_DIR / "kma_rainfall_rn_60m"
SMAP_DIR = BASE_DIR / "smap_L4_surface_rootzone"
SMAP_ALL = SMAP_DIR / "smap_L4_sm_all_events.csv"

OUT_DIR = BASE_DIR / "plots_fpi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS_BEFORE = 14
DAYS_AFTER  = 1

# =========================================================
# 그릴 조합(5개)
# =========================================================
COMBOS = [
    (1, 1, 3),  # 1h-1d-3h
    (1, 1, 9),  # 1h-1d-9h
    (1, 5, 3),  # 1h-5d-3h  (사용자 메시지 '1h-5d-1h'는 3h로 대체)
    (6, 1, 3),  # 6h-1d-3h
    (6, 5, 9),  # 6h-5d-9h
]

# =========================================================
# 유틸: time 파싱 (KST tz suffix 제거)
# =========================================================
def parse_kst_strip_tz(s: pd.Series) -> pd.Series:
    x = s.astype(str)
    x = x.str.replace(r"\+\d{2}:\d{2}$", "", regex=True)
    x = x.str.replace(r"\+\d{4}$", "", regex=True)
    return pd.to_datetime(x, errors="coerce")

def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd

def compute_imperv_proxy(df: pd.DataFrame, n_lag_h: int) -> pd.Series:
    # imperv = n * surface(t-n) / rootzone(t)
    surf_lag = df["sm_surface"].shift(n_lag_h)
    rz = df["sm_rootzone"]
    eps = 1e-6
    return n_lag_h * (surf_lag / (rz + eps))

def compute_fpi(df: pd.DataFrame, short_h: int, ant_d: int, n_lag_h: int) -> pd.Series:
    rain = df["rn_60m_mm"].fillna(0.0)
    I  = rain.rolling(window=short_h, min_periods=1).sum()
    AR = rain.rolling(window=ant_d * 24, min_periods=1).sum()
    SM = df["sm_rootzone"]
    IMP = compute_imperv_proxy(df, n_lag_h)
    return zscore(I) + zscore(AR) + zscore(SM) + zscore(IMP)

# =========================================================
# 1) Rain 로드
# =========================================================
rain_files = sorted(RAIN_DIR.glob("rain_rn_60m_event_*.csv"))
if not rain_files:
    raise FileNotFoundError(f"Rain 파일을 찾지 못했습니다: {RAIN_DIR}")

rain_list = []
for fp in rain_files:
    d = pd.read_csv(fp)
    d["tm_kst"] = pd.to_datetime(d["tm_kst"], errors="coerce")
    d["event_time_kst"] = pd.to_datetime(d["event_time_kst"], errors="coerce")
    d["id"] = d["id"].astype(int)
    d["rn_60m_mm"] = pd.to_numeric(d["rn_60m_mm"], errors="coerce").fillna(0.0)
    rain_list.append(d)

rain = pd.concat(rain_list, ignore_index=True).dropna(subset=["id", "tm_kst"])

# =========================================================
# 2) SMAP 로드
# =========================================================
smap = pd.read_csv(SMAP_ALL)
smap["time_kst_dt"] = parse_kst_strip_tz(smap["time_kst"])
smap["event_time_kst_dt"] = parse_kst_strip_tz(smap["event_time_kst"])
smap["id"] = smap["id"].astype(int)

# 숫자 컬럼 강제 변환 (문자 섞이면 NaN 될 수 있음)
smap["sm_surface"] = pd.to_numeric(smap["sm_surface"], errors="coerce")
smap["sm_rootzone"] = pd.to_numeric(smap["sm_rootzone"], errors="coerce")

smap = smap.dropna(subset=["id", "time_kst_dt"])

# =========================================================
# 이벤트별 1시간 격자 + SMAP 재표본화(핵심 수정)
# =========================================================
def build_event_timeseries(eid: int) -> pd.DataFrame:
    r = rain[rain["id"] == eid].copy()
    s = smap[smap["id"] == eid].copy()

    # event_time
    if not r.empty and r["event_time_kst"].notna().any():
        event_time = r["event_time_kst"].dropna().iloc[0]
    elif not s.empty and s["event_time_kst_dt"].notna().any():
        event_time = s["event_time_kst_dt"].dropna().iloc[0]
    else:
        raise ValueError(f"[Event {eid}] event_time을 찾지 못했습니다.")

    t_start = event_time - pd.Timedelta(days=DAYS_BEFORE)
    t_end   = event_time + pd.Timedelta(days=DAYS_AFTER)

    # 1시간 격자 (정각)
    tgrid = pd.date_range(t_start.floor("h"), t_end.ceil("h"), freq="1h")
    base = pd.DataFrame({"time": tgrid})
    base["id"] = eid
    base["event_time"] = event_time

    # ---- Rain (정각 기준으로 집계)
    rr = r[(r["tm_kst"] >= t_start) & (r["tm_kst"] <= t_end)].copy()
    rr = rr.groupby("tm_kst", as_index=False)["rn_60m_mm"].sum().rename(columns={"tm_kst": "time"})
    base = base.merge(rr, on="time", how="left")
    base["rn_60m_mm"] = base["rn_60m_mm"].fillna(0.0)

    # ---- SMAP: (중요) 정각 merge하지 말고 "시간 보간으로 1시간 격자에 재표본화"
    ss = s[(s["time_kst_dt"] >= t_start) & (s["time_kst_dt"] <= t_end)].copy()
    ss = ss.sort_values("time_kst_dt").dropna(subset=["sm_surface", "sm_rootzone"])

    # SMAP이 비어있으면 그대로 NaN
    if ss.empty:
        base["sm_surface"] = np.nan
        base["sm_rootzone"] = np.nan
        return base

    # SMAP series (DatetimeIndex = 실제 관측 시각(:30 포함))
    surf = ss.set_index("time_kst_dt")["sm_surface"]
    root = ss.set_index("time_kst_dt")["sm_rootzone"]

    # 1시간 격자에 reindex 후 time 보간
    idx = pd.DatetimeIndex(tgrid)

    surf_h = surf.reindex(surf.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)
    root_h = root.reindex(root.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)

    # 양끝 채움
    base["sm_surface"] = surf_h.ffill().bfill().values
    base["sm_rootzone"] = root_h.ffill().bfill().values

    return base

# =========================================================
# 3) 이벤트별 FPI plot
# =========================================================
event_ids = sorted(set(rain["id"]).union(set(smap["id"])))

for eid in event_ids:
    try:
        df = build_event_timeseries(eid)
    except Exception as e:
        print(f"[Event {eid}] build 실패: {e}")
        continue

    event_time = df["event_time"].iloc[0]
    t_start = event_time - pd.Timedelta(days=DAYS_BEFORE)
    t_end   = event_time + pd.Timedelta(days=DAYS_AFTER)

    plt.figure(figsize=(13, 5))

    # 혹시라도 전부 NaN이면 바로 알림
    for short_h, ant_d, nlag in COMBOS:
        fpi = compute_fpi(df, short_h, ant_d, nlag)
        label = f"{short_h}h-{ant_d}d-{nlag}h"
        plt.plot(df["time"], fpi, linewidth=1.6, label=label)

    plt.axvline(event_time, linestyle="--", linewidth=2.0, color="red")

    plt.xlim(t_start, t_end)
    plt.xlabel("Time (KST)")
    plt.ylabel("FPI (standardized sum)")
    plt.title(f"Event {eid} | Event time = {event_time.strftime('%Y-%m-%d %H:%M')} (KST)")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_png = OUT_DIR / f"event_{eid}_FPI_5combos.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[Saved] {out_png}")

print(f"\nDone. Output folder: {OUT_DIR}")
