from pathlib import Path
import numpy as np
import pandas as pd

# =========================================================
# 경로 설정
# =========================================================
BASE_DIR = Path(r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\24th Urban Flood (Timeseries)\python")
RAIN_DIR = BASE_DIR / "kma_rainfall_rn_60m"
SMAP_DIR = BASE_DIR / "smap_L4_surface_rootzone"
OUT_DIR  = BASE_DIR / "fpi_rank"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMAP_ALL = SMAP_DIR / "smap_L4_sm_all_events.csv"

# =========================================================
# 사용자 설정(조합)
# =========================================================
SHORT_WINDOWS_H = [1, 3, 6]          # 단기 강우 (시간)
ANT_WINDOWS_D   = [1, 3, 5]          # 선행 강우 (일)
N_LAG_H         = [3, 6, 9]          # 불투수 proxy lag n (시간)

DAYS_BEFORE = 14
DAYS_AFTER  = 1

# Score 가중치
W_PCTL = 0.60
W_LT   = 0.25
W_DUR  = 0.15   # (기존 AUC 대신 DurationScore 사용)

# lead time scoring
LT_TARGET_H = 12.0
LT_MAX_H    = 72.0

# metric window (요청 반영)
LEAD_SEARCH_H     = 72     # lead peak 탐색 창
DURATION_WINDOW_H = 72     # duration 계산 창
THR_Q             = 0.95   # pre 전체 기준 threshold 분위수

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

def max_consecutive_true(mask: pd.Series) -> int:
    """Boolean mask에서 True의 최대 연속 길이(시간 수)"""
    if mask is None or len(mask) == 0:
        return 0
    arr = mask.astype(int).values
    best = cur = 0
    for v in arr:
        if v == 1:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)

# =========================================================
# 1) Rain 로드
# =========================================================
rain_files = sorted(RAIN_DIR.glob("rain_rn_60m_event_*.csv"))
if not rain_files:
    raise FileNotFoundError(f"Rain 파일을 찾지 못했습니다: {RAIN_DIR}")

rain_list = []
for fp in rain_files:
    df = pd.read_csv(fp)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["tm_kst"] = pd.to_datetime(df["tm_kst"], errors="coerce")
    df["event_time_kst"] = pd.to_datetime(df["event_time_kst"], errors="coerce")
    df["rn_60m_mm"] = pd.to_numeric(df["rn_60m_mm"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["id", "tm_kst", "event_time_kst"])
    df["id"] = df["id"].astype(int)

    rain_list.append(df)

rain = pd.concat(rain_list, ignore_index=True)

# =========================================================
# 2) SMAP 로드
# =========================================================
if not SMAP_ALL.exists():
    raise FileNotFoundError(f"SMAP 파일을 찾지 못했습니다: {SMAP_ALL}")

smap = pd.read_csv(SMAP_ALL)

smap["id"] = pd.to_numeric(smap["id"], errors="coerce")
smap["time_kst_dt"] = parse_kst_strip_tz(smap["time_kst"])
smap["event_time_kst_dt"] = parse_kst_strip_tz(smap["event_time_kst"])

smap["sm_surface"] = pd.to_numeric(smap["sm_surface"], errors="coerce")
smap["sm_rootzone"] = pd.to_numeric(smap["sm_rootzone"], errors="coerce")

smap = smap.dropna(subset=["id", "time_kst_dt"])
smap["id"] = smap["id"].astype(int)

# =========================================================
# 3) 이벤트별 공통 1시간 time grid 생성 + (핵심) SMAP 재표본화
# =========================================================
event_ids = sorted(set(rain["id"]).union(set(smap["id"])))

def build_event_timeseries(eid: int) -> pd.DataFrame:
    r = rain[rain["id"] == eid].copy()
    s = smap[smap["id"] == eid].copy()

    # event_time 결정 (rain 우선)
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

    # ---- Rain merge
    rr = r[(r["tm_kst"] >= t_start) & (r["tm_kst"] <= t_end)].copy()
    rr = rr.groupby("tm_kst", as_index=False)["rn_60m_mm"].sum().rename(columns={"tm_kst": "time"})
    base = base.merge(rr, on="time", how="left")
    base["rn_60m_mm"] = pd.to_numeric(base["rn_60m_mm"], errors="coerce").fillna(0.0)

    # ---- SMAP reindex+time interpolation (중요: :30 오프셋 문제 해결)
    if s.empty:
        base["sm_surface"] = np.nan
        base["sm_rootzone"] = np.nan
        return base

    ss = s[(s["time_kst_dt"] >= t_start) & (s["time_kst_dt"] <= t_end)].copy()
    ss = ss.sort_values("time_kst_dt").dropna(subset=["time_kst_dt", "sm_surface", "sm_rootzone"])

    if ss.empty:
        base["sm_surface"] = np.nan
        base["sm_rootzone"] = np.nan
        return base

    idx = pd.DatetimeIndex(tgrid)

    surf = ss.set_index("time_kst_dt")["sm_surface"]
    root = ss.set_index("time_kst_dt")["sm_rootzone"]

    # union -> time interpolate -> 정각 idx로 reindex
    surf_h = surf.reindex(surf.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)
    root_h = root.reindex(root.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)

    base["sm_surface"] = surf_h.ffill().bfill().values
    base["sm_rootzone"] = root_h.ffill().bfill().values

    return base

event_ts = {eid: build_event_timeseries(eid) for eid in event_ids}

# =========================================================
# 4) 조합별 FPI 계산
# =========================================================
def compute_impervious_proxy(df: pd.DataFrame, n_lag_h: int) -> pd.Series:
    # IMP = n * surface(t-n) / rootzone(t)
    surf_lag = df["sm_surface"].shift(n_lag_h)
    rz = df["sm_rootzone"]
    eps = 1e-6
    return n_lag_h * (surf_lag / (rz + eps))

def compute_fpi_timeseries(df: pd.DataFrame, short_h: int, ant_d: int, n_lag_h: int) -> pd.Series:
    rain1h = df["rn_60m_mm"].fillna(0.0)

    I  = rain1h.rolling(window=short_h, min_periods=1).sum()
    AR = rain1h.rolling(window=ant_d * 24, min_periods=1).sum()

    SM  = df["sm_rootzone"]
    IMP = compute_impervious_proxy(df, n_lag_h)

    return zscore(I) + zscore(AR) + zscore(SM) + zscore(IMP)

# =========================================================
# 5) 이벤트별 metric (Percentile / Lead(72h) / Duration(72h max consecutive))
# =========================================================
def event_metrics(df: pd.DataFrame, fpi: pd.Series) -> dict:
    d = df.copy()
    d["FPI"] = fpi.values

    event_time = d["event_time"].iloc[0]

    # 사고시각: 가장 가까운 1h step
    idx_evt = (d["time"] - event_time).abs().idxmin()
    f_evt = d.loc[idx_evt, "FPI"]

    vals = d["FPI"].dropna().values
    if len(vals) < 5 or np.isnan(f_evt):
        pctl = np.nan
    else:
        pctl = (vals <= f_evt).mean()  # 0~1

    pre = d[d["time"] <= event_time].copy()

    # threshold (pre 전체 기준)
    if pre["FPI"].notna().sum() >= 20:
        thr = float(np.nanquantile(pre["FPI"].values, THR_Q))
    else:
        thr = np.nan

    # ---- Lead time: 최근 72h 창에서 peak
    t0 = event_time - pd.Timedelta(hours=LEAD_SEARCH_H)
    pre_win = pre[(pre["time"] >= t0) & (pre["time"] <= event_time)].copy()

    if pre_win["FPI"].notna().any():
        idx_peak = pre_win["FPI"].idxmax()
        t_peak = pre_win.loc[idx_peak, "time"]
        lead_h = (event_time - t_peak).total_seconds() / 3600.0
    else:
        lead_h = np.nan

    # ---- Duration: 최근 72h 창에서 thr 초과 최대 연속시간
    t1 = event_time - pd.Timedelta(hours=DURATION_WINDOW_H)
    dur_win = pre[(pre["time"] >= t1) & (pre["time"] <= event_time)].copy()

    if np.isfinite(thr) and dur_win["FPI"].notna().any():
        mask = (dur_win["FPI"] >= thr).fillna(False)
        dur_h = float(max_consecutive_true(mask))
    else:
        dur_h = np.nan

    return {"pctl": pctl, "lead_h": lead_h, "dur_h": dur_h}

def lead_time_score(lead_h: float) -> float:
    """
    lead_h가 0~LT_MAX 사이에서:
      - LT_TARGET 근처면 점수 높음
      - 너무 이르거나(>>target) 너무 늦거나(0 근접) 페널티
      - 음수(사고 이후 피크)는 0
    """
    if np.isnan(lead_h):
        return np.nan
    if lead_h < 0:
        return 0.0
    if lead_h > LT_MAX_H:
        lead_h = LT_MAX_H

    if lead_h <= LT_TARGET_H:
        return lead_h / LT_TARGET_H
    else:
        return (LT_MAX_H - lead_h) / (LT_MAX_H - LT_TARGET_H)

# duration은 조합 간 스케일이 다를 수 있어 min-max로 0~1 정규화해서 score로 사용
# (조합 내 이벤트 평균 dur_h -> 조합 간 비교)
# =========================================================
# 6) 27조합 랭킹
# =========================================================
rows = []
per_combo_event_table = []  # (옵션) 조합×이벤트 상세 저장용

for short_h in SHORT_WINDOWS_H:
    for ant_d in ANT_WINDOWS_D:
        for nlag in N_LAG_H:
            combo_name = f"I{short_h}h_AR{ant_d}d_IMP{nlag}h"
            per_event = []

            for eid in event_ids:
                df = event_ts[eid]
                fpi = compute_fpi_timeseries(df, short_h, ant_d, nlag)
                m = event_metrics(df, fpi)
                m["id"] = eid
                per_event.append(m)
                per_combo_event_table.append({
                    "combo": combo_name, "id": eid,
                    "pctl": m["pctl"], "lead_h": m["lead_h"], "dur_h": m["dur_h"]
                })

            met = pd.DataFrame(per_event)

            rows.append({
                "combo": combo_name,
                "I_short_h": short_h,
                "AR_days": ant_d,
                "IMP_lag_h": nlag,
                "mean_event_percentile": met["pctl"].mean(skipna=True),
                "mean_lead_time_h": met["lead_h"].mean(skipna=True),
                "mean_lead_time_score_0to1": met["lead_h"].apply(lead_time_score).mean(skipna=True),
                "mean_duration_maxconsec_h": met["dur_h"].mean(skipna=True),
            })

rank_df = pd.DataFrame(rows)
detail_df = pd.DataFrame(per_combo_event_table)

# ---- Duration score (조합 간 0~1 min-max)
dur_min = rank_df["mean_duration_maxconsec_h"].min(skipna=True)
dur_max = rank_df["mean_duration_maxconsec_h"].max(skipna=True)
if np.isfinite(dur_min) and np.isfinite(dur_max) and dur_max > dur_min:
    rank_df["duration_score_0to1"] = (rank_df["mean_duration_maxconsec_h"] - dur_min) / (dur_max - dur_min)
else:
    rank_df["duration_score_0to1"] = np.nan

# ---- 최종 Score
rank_df["Score"] = (
    W_PCTL * rank_df["mean_event_percentile"] +
    W_LT   * rank_df["mean_lead_time_score_0to1"] +
    W_DUR  * rank_df["duration_score_0to1"]
)

rank_df = rank_df.sort_values("Score", ascending=False).reset_index(drop=True)
rank_df.insert(0, "Rank", rank_df.index + 1)

# 저장
out_csv  = OUT_DIR / "rank_27_combinations.csv"
out_xlsx = OUT_DIR / "rank_27_combinations.xlsx"
out_detail_csv = OUT_DIR / "rank_27_event_metrics_long.csv"

rank_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
detail_df.to_csv(out_detail_csv, index=False, encoding="utf-8-sig")

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    rank_df.to_excel(w, sheet_name="rank", index=False)
    detail_df.to_excel(w, sheet_name="per_event", index=False)

print(f"Saved:\n- {out_csv}\n- {out_xlsx}\n- {out_detail_csv}")
print(rank_df.head(10))
