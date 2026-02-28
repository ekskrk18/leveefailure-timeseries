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
OUT_DIR  = BASE_DIR / "fpi_metrics_pir_no_outliers"
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

# 평가 구간
PRE_HOURS = 72            # pre72: event-72h ~ event
LEAD_MAX_H = 72           # lead time 최대
DUR_MAX_H  = 72           # duration 최대(최대 연속 지속시간도 72h로 cap)

# duration 임계치(상위 q 분위)
DUR_THR_Q = 0.95

EPS = 1e-9

# =========================================================
# 유틸
# =========================================================
def parse_kst_strip_tz(s: pd.Series) -> pd.Series:
    """time_kst가 '...+09:00' 형태면 suffix 제거 후 datetime으로"""
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

def max_consecutive_true(mask: np.ndarray) -> int:
    """boolean mask에서 True의 최대 연속 길이(개수)"""
    if mask.size == 0:
        return 0
    max_run = 0
    run = 0
    for v in mask:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

# --- (NEW) IQR outlier 제거 (box plot용: fliers 자체를 데이터에서 삭제)
def remove_outliers_iqr(x: pd.Series, k: float = 1.5) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 4:
        return x  # 표본이 너무 적으면 그냥 반환
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return x
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return x[(x >= lo) & (x <= hi)]

# --- (NEW) “가장 튀는 값 1개” 제거 후 평균 (heatmap용)
def mean_drop_most_extreme(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return np.nan
    if n <= 2:
        return float(x.mean())
    med = float(x.median())
    idx = (x - med).abs().idxmax()   # median에서 가장 멀리 떨어진 1개
    x2 = x.drop(index=idx)
    return float(x2.mean())

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
    df["rn_60m_mm"] = pd.to_numeric(df["rn_60m_mm"], errors="coerce")

    df = df.dropna(subset=["id", "tm_kst", "event_time_kst"])
    df["id"] = df["id"].astype(int)
    df["rn_60m_mm"] = df["rn_60m_mm"].fillna(0.0)

    rain_list.append(df)

rain = pd.concat(rain_list, ignore_index=True)

# =========================================================
# 2) SMAP 로드
# =========================================================
smap = pd.read_csv(SMAP_ALL)

smap["id"] = pd.to_numeric(smap["id"], errors="coerce")
smap["time_kst_dt"] = parse_kst_strip_tz(smap["time_kst"])
smap["event_time_kst_dt"] = parse_kst_strip_tz(smap["event_time_kst"])

smap["sm_surface"] = pd.to_numeric(smap["sm_surface"], errors="coerce")
smap["sm_rootzone"] = pd.to_numeric(smap["sm_rootzone"], errors="coerce")

smap = smap.dropna(subset=["id", "time_kst_dt"])
smap["id"] = smap["id"].astype(int)

# =========================================================
# 3) 이벤트별 공통 1시간 time grid 생성 + SMAP 1시간 보간
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

    # 1시간 격자
    tgrid = pd.date_range(t_start.floor("h"), t_end.ceil("h"), freq="1h")
    base = pd.DataFrame({"time": tgrid})
    base["id"] = eid
    base["event_time"] = event_time

    # ---- Rain
    if not r.empty:
        rr = r[(r["tm_kst"] >= t_start) & (r["tm_kst"] <= t_end)].copy()
        rr = rr.groupby("tm_kst", as_index=False)["rn_60m_mm"].sum().rename(columns={"tm_kst": "time"})
        base = base.merge(rr, on="time", how="left")
    base["rn_60m_mm"] = pd.to_numeric(base.get("rn_60m_mm", 0.0), errors="coerce").fillna(0.0)

    # ---- SMAP -> 1h time interpolation
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
    surf = ss.set_index("time_kst_dt")["sm_surface"].sort_index()
    root = ss.set_index("time_kst_dt")["sm_rootzone"].sort_index()

    surf_h = surf.reindex(surf.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)
    root_h = root.reindex(root.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)

    base["sm_surface"]  = surf_h.ffill().bfill().values
    base["sm_rootzone"] = root_h.ffill().bfill().values
    return base

event_ts = {eid: build_event_timeseries(eid) for eid in event_ids}

# =========================================================
# 4) FPI 계산 (27 combos)
# =========================================================
def compute_impervious_proxy(df: pd.DataFrame, n_lag_h: int) -> pd.Series:
    surf_lag = df["sm_surface"].shift(n_lag_h)
    rz = df["sm_rootzone"]
    return n_lag_h * (surf_lag / (rz + 1e-6))

def compute_fpi_timeseries(df: pd.DataFrame, short_h: int, ant_d: int, n_lag_h: int) -> pd.Series:
    rain_ = df["rn_60m_mm"].fillna(0.0)
    I  = rain_.rolling(window=short_h, min_periods=1).sum()
    AR = rain_.rolling(window=ant_d * 24, min_periods=1).sum()

    SM  = df["sm_rootzone"]
    IMP = compute_impervious_proxy(df, n_lag_h)
    return zscore(I) + zscore(AR) + zscore(SM) + zscore(IMP)

# =========================================================
# 5) 지표: Peak Intensity Ratio + lead + duration (pre72)
# =========================================================
def metrics_pre72(df: pd.DataFrame, fpi: pd.Series) -> dict:
    df2 = df.copy()
    df2["FPI"] = fpi.values

    event_time = df2["event_time"].iloc[0]
    pre_start = event_time - pd.Timedelta(hours=PRE_HOURS)

    pre = df2[(df2["time"] >= pre_start) & (df2["time"] <= event_time)].copy()
    pre = pre.dropna(subset=["FPI"])

    if pre.empty or pre["FPI"].notna().sum() < 5:
        return {"peak_ratio": np.nan, "lead_h": np.nan, "duration_h": np.nan}

    f_max = float(pre["FPI"].max())
    f_mean = float(pre["FPI"].mean())
    peak_ratio = f_max / (f_mean + EPS)

    idx_peak = pre["FPI"].idxmax()
    t_peak = df2.loc[idx_peak, "time"]
    lead_h = (event_time - t_peak).total_seconds() / 3600.0
    lead_h = float(np.clip(lead_h, 0.0, LEAD_MAX_H))

    thr = float(np.nanquantile(pre["FPI"].values, DUR_THR_Q))
    mask = (pre["FPI"].values >= thr)
    max_run = max_consecutive_true(mask)
    duration_h = float(np.clip(max_run, 0.0, DUR_MAX_H))

    return {"peak_ratio": peak_ratio, "lead_h": lead_h, "duration_h": duration_h}

# =========================================================
# 6) 27조합 × 이벤트별 metrics 계산
# =========================================================
records = []
for short_h in SHORT_WINDOWS_H:
    for ant_d in ANT_WINDOWS_D:
        for nlag in N_LAG_H:
            combo = f"{short_h}h_AR{ant_d}d_N{nlag}h"
            for eid in event_ids:
                df = event_ts[eid]
                fpi = compute_fpi_timeseries(df, short_h, ant_d, nlag)
                m = metrics_pre72(df, fpi)
                records.append({
                    "id": eid,
                    "combo": combo,
                    "short_h": short_h,
                    "ant_d": ant_d,
                    "nlag_h": nlag,
                    "peak_ratio": m["peak_ratio"],
                    "lead_h": m["lead_h"],
                    "duration_h": m["duration_h"],
                })

met_long = pd.DataFrame(records)
met_long.to_csv(OUT_DIR / "metrics_27combos_per_event_raw.csv", index=False, encoding="utf-8-sig")

# =========================================================
# 7) Box plot용: combo별 outlier(IQR) 제거한 버전 생성
# =========================================================
def build_box_clean_df(met_long: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    out = []
    for combo, g in met_long.groupby("combo"):
        x = remove_outliers_iqr(g[metric_col])
        if len(x) == 0:
            continue
        gg = g.loc[x.index].copy()  # outlier 제외된 row만 남김
        out.append(gg)
    if not out:
        return met_long.iloc[0:0].copy()
    return pd.concat(out, ignore_index=True)

met_box_clean = {}
for col in ["peak_ratio", "lead_h", "duration_h"]:
    met_box_clean[col] = build_box_clean_df(met_long, col)

# 저장(참고용)
for col, dfc in met_box_clean.items():
    dfc.to_csv(OUT_DIR / f"metrics_27combos_per_event_boxclean_{col}.csv",
               index=False, encoding="utf-8-sig")

# =========================================================
# 8) Box plot (outlier 제거 + fliers 미표시)
# =========================================================
def make_boxplot(metric_col: str, title: str, ylabel: str, out_png: Path):
    df = met_box_clean[metric_col].dropna(subset=[metric_col]).copy()

    order = []
    for sh in SHORT_WINDOWS_H:
        for ad in ANT_WINDOWS_D:
            for nl in N_LAG_H:
                order.append(f"{sh}h_AR{ad}d_N{nl}h")

    data = [df.loc[df["combo"] == c, metric_col].values for c in order]

    plt.figure(figsize=(22, 6))
    plt.boxplot(data, showfliers=False)  # 점(튀는값) 표시 자체도 제거
    plt.xticks(np.arange(1, len(order) + 1), order, rotation=55, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

make_boxplot(
    metric_col="peak_ratio",
    title="27-combo distribution by event: Peak Intensity Ratio (pre72) [OUTLIERS REMOVED]",
    ylabel="PIR = max(FPI_pre72) / mean(FPI_pre72)",
    out_png=OUT_DIR / "box_peak_ratio_27_no_outliers.png"
)

make_boxplot(
    metric_col="lead_h",
    title="27-combo distribution by event: lead time (pre72 peak → event) [OUTLIERS REMOVED]",
    ylabel="Lead time (hours, 0~72; positive=pre-event)",
    out_png=OUT_DIR / "box_leadtime_27_no_outliers.png"
)

make_boxplot(
    metric_col="duration_h",
    title=f"27-combo distribution by event: duration above top{int(DUR_THR_Q*100)}% threshold (pre72) [OUTLIERS REMOVED]",
    ylabel=f"Max consecutive duration (hours, cap {DUR_MAX_H}h)",
    out_png=OUT_DIR / "box_duration_27_no_outliers.png"
)

# =========================================================
# 9) Heatmap: 셀(조합)별 “가장 튀는 값 1개” 제외 평균으로 계산
# =========================================================
def panel_heatmap_drop1(metric_col: str, title_prefix: str, cbar_label: str, out_png: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    ims = []

    for i, sh in enumerate(SHORT_WINDOWS_H):
        ax = axes[i]
        sub = met_long[met_long["short_h"] == sh].copy()

        # (ant_d, nlag_h) 셀별: most-extreme 1개 drop 후 mean
        agg = (
            sub.groupby(["ant_d", "nlag_h"])[metric_col]
            .apply(mean_drop_most_extreme)
            .reset_index(name=f"mean_{metric_col}_drop1")
        )

        piv = (
            agg.pivot(index="ant_d", columns="nlag_h", values=f"mean_{metric_col}_drop1")
            .reindex(index=ANT_WINDOWS_D, columns=N_LAG_H)
        )

        mat = piv.values.astype(float)
        im = ax.imshow(mat, aspect="auto")
        ims.append(im)

        ax.set_title(f"{title_prefix}\nShort={sh}h (drop 1 extreme per cell)")
        ax.set_xlabel("n (lag) for IMP")
        ax.set_ylabel("Antecedent rainfall window")

        ax.set_xticks(np.arange(len(N_LAG_H)))
        ax.set_xticklabels([f"{x}h" for x in N_LAG_H])
        ax.set_yticks(np.arange(len(ANT_WINDOWS_D)))
        ax.set_yticklabels([f"{y}d" for y in ANT_WINDOWS_D])

        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=10)

    # 공통 color scale
    vmin = np.nanmin([np.nanmin(im.get_array()) for im in ims])
    vmax = np.nanmax([np.nanmax(im.get_array()) for im in ims])
    for im in ims:
        im.set_clim(vmin, vmax)

    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), location="right",
                        fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)

    plt.savefig(out_png, dpi=200)
    plt.close()

panel_heatmap_drop1(
    metric_col="peak_ratio",
    title_prefix="Mean Peak Intensity Ratio (pre72)",
    cbar_label="PIR (drop 1 extreme)",
    out_png=OUT_DIR / "panel_peak_ratio_1x3_drop1.png"
)

panel_heatmap_drop1(
    metric_col="lead_h",
    title_prefix="Mean lead time (hours) (pre72 peak → event)",
    cbar_label="lead_h (drop 1 extreme)",
    out_png=OUT_DIR / "panel_leadtime_1x3_drop1.png"
)

panel_heatmap_drop1(
    metric_col="duration_h",
    title_prefix=f"Mean duration above top{int(DUR_THR_Q*100)}% threshold (pre72)",
    cbar_label="duration_h (drop 1 extreme)",
    out_png=OUT_DIR / "panel_duration_1x3_drop1.png"
)

print("Saved (raw per-event):", OUT_DIR / "metrics_27combos_per_event_raw.csv")
print("Saved (box-clean per-event csvs):")
print("-", OUT_DIR / "metrics_27combos_per_event_boxclean_peak_ratio.csv")
print("-", OUT_DIR / "metrics_27combos_per_event_boxclean_lead_h.csv")
print("-", OUT_DIR / "metrics_27combos_per_event_boxclean_duration_h.csv")
print("Saved boxplots (no outliers):")
print("-", OUT_DIR / "box_peak_ratio_27_no_outliers.png")
print("-", OUT_DIR / "box_leadtime_27_no_outliers.png")
print("-", OUT_DIR / "box_duration_27_no_outliers.png")
print("Saved heatmaps (drop 1 extreme per cell):")
print("-", OUT_DIR / "panel_peak_ratio_1x3_drop1.png")
print("-", OUT_DIR / "panel_leadtime_1x3_drop1.png")
print("-", OUT_DIR / "panel_duration_1x3_drop1.png")
print("\nDone.")
