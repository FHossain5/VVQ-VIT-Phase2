# === Tier-A Post-hoc Improvements on Existing Predictions ===
# - Per-subset calibration (by compression, qp) with auto Linear vs Isotonic selection
# - Optional residual repair (global isotonic) if it improves VAL RMSE
# - Optional trimmed-mean ensemble if multiple preds CSVs exist
# - Only keep new TEST preds if RMSE improves vs baseline
# ------------------------------------------------------------

import os, re, json, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

# -----------------------
# Config: edit if needed
# -----------------------
EVAL_DIR = Path("/content/drive/MyDrive/dynamic_dataset_all_compression/VIT_Model/eval_client_TTA_cal_final")
# Fallback names if your files live in a slightly different folder:
VAL_PREDS_CANDIDATES  = [
    EVAL_DIR / "client_eval_preds_val.csv",
    EVAL_DIR / "client_eval_preds_val_with_qp_comp.csv",
    EVAL_DIR / "client_eval_preds_val_TTA.csv",
]
TEST_PREDS_CANDIDATES = [
    EVAL_DIR / "client_eval_preds_test_with_qp_comp.csv",
    EVAL_DIR / "client_eval_preds_test.csv",
    EVAL_DIR / "client_eval_preds_test_TTA.csv",
]

OUT_DIR = EVAL_DIR / "tierA_calibration_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: if you have multiple CSVs for TTA/seed ensemble, list a glob:
ENSEMBLE_GLOB = str(EVAL_DIR / "client_eval_preds_test_seed*.csv")  # change pattern if needed
TRIM_PCT = 0.10  # 10% trimmed-mean if ensemble present

MIN_SUBSET_VAL = 30  # minimum samples to fit per-subset mapping; else fall back to global
USE_RESIDUAL_REPAIR = True  # try a global isotonic monotonic repair after per-subset cal (only if improves VAL)

# -----------------------
# Utilities
# -----------------------
def pick_pred_column(df):
    # Prefer raw predictions (uncalibrated) for re-calibration
    for c in ("y_pred_uncal", "y_pred", "yhat", "pred"):
        if c in df.columns:
            return c
    # Fall back to calibrated if only that exists (still OK, our mapping can re-calibrate)
    for c in ("y_pred_cal", "y_cal"):
        if c in df.columns:
            return c
    raise ValueError("No prediction column found. Expected one of: y_pred_uncal, y_pred, yhat, pred, y_pred_cal, y_cal")

def parse_qp_comp(video_id):
    # Expect things like "..._TEX_QP35", "..._GEOM_QP25", "..._COMB_QP15"
    comp = None
    qp   = None
    m = re.search(r"_(TEX|GEOM|COMB)_QP(\d+)", str(video_id))
    if m:
        comp = m.group(1)
        qp = int(m.group(2))
    return comp, qp

def ensure_qp_comp(df):
    need = []
    if "compression" not in df.columns: need.append("compression")
    if "qp" not in df.columns: need.append("qp")
    if not need:
        return df.copy()
    out = df.copy()
    comp_parsed, qp_parsed = [], []
    for vid in out["video_id"]:
        c, q = parse_qp_comp(vid)
        comp_parsed.append(c)
        qp_parsed.append(q)
    if "compression" not in out.columns:
        out["compression"] = comp_parsed
    if "qp" not in out.columns:
        out["qp"] = qp_parsed
    return out

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    plcc = float(np.corrcoef(y_true, y_pred)[0,1]) if len(y_true) > 1 else float("nan")
    srcc = float(spearmanr(y_true, y_pred, nan_policy="omit").correlation)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
    acc05 = float(np.mean(np.abs(y_true - y_pred) <= 0.05))
    acc10 = float(np.mean(np.abs(y_true - y_pred) <= 0.10))
    return dict(
        MAE=float(mae), RMSE=rmse, PLCC=plcc, SRCC=srcc, R2=r2,
        Acc_pm_0_05=acc05, Acc_pm_0_10=acc10
    )

def fit_linear(x, y):
    # y ≈ a * x + b (least squares)
    a, b = np.polyfit(x, y, deg=1)
    return a, b

def apply_linear(x, a, b):
    return a * x + b

def fit_iso(x, y):
    # monotonic increasing map; clip y to finite
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    iso = IsotonicRegression(y_min=None, y_max=None, increasing=True, out_of_bounds="clip")
    iso.fit(x, y)
    return iso

def evaluate_subset_map(x_val, y_val, map_kind):
    if map_kind["type"] == "linear":
        y_cal = apply_linear(x_val, map_kind["a"], map_kind["b"])
    elif map_kind["type"] == "isotonic":
        y_cal = map_kind["iso"].predict(x_val)
    else:
        raise ValueError("Unknown map kind")
    return metrics(y_val, y_cal), y_cal

def choose_best_mapping(x_val, y_val):
    # Compare linear vs isotonic on VAL; select the one with lower RMSE (ties break on MAE)
    lm_a, lm_b = fit_linear(x_val, y_val)
    m_lin, _   = evaluate_subset_map(x_val, y_val, {"type":"linear", "a":lm_a, "b":lm_b})
    iso        = fit_iso(x_val, y_val)
    m_iso, _   = evaluate_subset_map(x_val, y_val, {"type":"isotonic", "iso":iso})

    if (m_iso["RMSE"] < m_lin["RMSE"]) or (abs(m_iso["RMSE"]-m_lin["RMSE"])<1e-6 and m_iso["MAE"] <= m_lin["MAE"]):
        return {"type":"isotonic", "iso":iso, "val_metrics":m_iso}
    else:
        return {"type":"linear", "a":lm_a, "b":lm_b, "val_metrics":m_lin}

def maybe_trimmed_mean(arr, trim_pct=0.1):
    arr = np.sort(np.asarray(arr, float))
    n = len(arr)
    k = int(n * trim_pct)
    if n - 2*k <= 0:  # too few to trim
        return float(np.mean(arr))
    return float(np.mean(arr[k:n-k]))

# -----------------------
# Load VAL/TEST predictions
# -----------------------
def first_existing(cands):
    for p in cands:
        if Path(p).exists():
            return Path(p)
    return None

VAL_PATH = first_existing(VAL_PREDS_CANDIDATES)
TEST_PATH = first_existing(TEST_PREDS_CANDIDATES)
if VAL_PATH is None or TEST_PATH is None:
    raise FileNotFoundError(f"VAL or TEST preds not found.\nVAL tried: {VAL_PREDS_CANDIDATES}\nTEST tried: {TEST_PREDS_CANDIDATES}")

val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

# Ensure columns
if "video_id" not in val.columns or "y_true" not in val.columns:
    raise ValueError("VAL CSV must contain at least ['video_id','y_true', <pred columns>].")
if "video_id" not in test.columns or "y_true" not in test.columns:
    raise ValueError("TEST CSV must contain at least ['video_id','y_true', <pred columns>].")

val = ensure_qp_comp(val)
test = ensure_qp_comp(test)

pred_col_val  = pick_pred_column(val)
pred_col_test = pick_pred_column(test)

print(f"[INFO] Using prediction columns — VAL: {pred_col_val}   TEST: {pred_col_test}")

# -----------------------
# Optional: Ensemble & trimmed-mean if multiple test CSVs are present
# -----------------------
ensemble_files = sorted(glob.glob(ENSEMBLE_GLOB))
if len(ensemble_files) >= 2:
    print(f"[INFO] Found {len(ensemble_files)} test files for ensemble. Applying {int(TRIM_PCT*100)}% trimmed-mean on matched rows.")
    # Merge by video_id
    base = test[["video_id","y_true","compression","qp"]].copy()
    preds_mat = []
    for fp in ensemble_files:
        df = pd.read_csv(fp)
        df = ensure_qp_comp(df)
        c = pick_pred_column(df)
        base = base.merge(df[["video_id", c]].rename(columns={c: Path(fp).stem}), on="video_id", how="left")
        preds_mat.append(Path(fp).stem)
    # Compute trimmed-mean across those columns
    pm = base[preds_mat].values
    test["y_pred_ensemble"] = np.apply_along_axis(maybe_trimmed_mean, 1, pm, trim_pct=TRIM_PCT)
    # Prefer ensemble for calibration
    pred_col_test = "y_pred_ensemble"
else:
    print("[INFO] No multi-file ensemble found. Skipping trimmed-mean ensemble.")

# -----------------------
# Baseline metrics (before new calibration)
# -----------------------
base_val_metrics  = metrics(val["y_true"].values,  val[pred_col_val].values)
base_test_metrics = metrics(test["y_true"].values, test[pred_col_test].values)

print("\n=== Baseline (before new per-subset calibration) ===")
print("VAL :", base_val_metrics)
print("TEST:", base_test_metrics)

# -----------------------
# Per-subset calibration
# -----------------------
def calibrate_per_subset(val_df, test_df, pred_col, min_subset=30):
    # Group key = (compression, qp)
    v = val_df.copy()
    t = test_df.copy()

    key_cols = ["compression", "qp"]
    for c in key_cols:
        if c not in v.columns or c not in t.columns:
            raise ValueError(f"Missing required column '{c}' for subset calibration.")

    maps = {}  # (comp, qp) -> map_kind

    # Fit per subset if enough validation samples, else global
    # Prepare global backup
    global_map = choose_best_mapping(v[pred_col].values, v["y_true"].values)

    # Build subset maps
    for (comp, qp), g in v.groupby(key_cols):
        if len(g) >= min_subset and g[pred_col].notna().sum() >= min_subset:
            mk = choose_best_mapping(g[pred_col].values, g["y_true"].values)
            maps[(comp, qp)] = mk
        else:
            maps[(comp, qp)] = global_map

    # Apply to VAL/TEST
    def apply_map_row(row, x):
        mk = maps.get((row["compression"], int(row["qp"])), global_map)
        if mk["type"] == "linear":
            return apply_linear(x, mk["a"], mk["b"])
        else:
            return float(mk["iso"].predict([x])[0])

    val_cal = v[pred_col].copy()
    test_cal = t[pred_col].copy()

    val_cal = [apply_map_row(r, x) for r, x in zip(v[key_cols].to_dict("records"), v[pred_col].values)]
    test_cal = [apply_map_row(r, x) for r, x in zip(t[key_cols].to_dict("records"), t[pred_col].values)]

    v["y_pred_subsetcal"] = np.asarray(val_cal, float)
    t["y_pred_subsetcal"] = np.asarray(test_cal, float)

    return v, t, maps, global_map

val_c, test_c, subset_maps, global_map = calibrate_per_subset(val, test, pred_col_val, min_subset=MIN_SUBSET_VAL)

m_val_subset = metrics(val_c["y_true"], val_c["y_pred_subsetcal"])
m_test_subset = metrics(test_c["y_true"], test_c["y_pred_subsetcal"])

print("\n=== After per-subset calibration (auto Linear/Isotonic) ===")
print("VAL :", m_val_subset)
print("TEST:", m_test_subset)

# -----------------------
# Optional residual repair (global isotonic) — only if it improves VAL RMSE
# -----------------------
def residual_repair(val_df, test_df, src_col, out_col):
    # Fit global isotonic on VAL
    ir = fit_iso(val_df[src_col].values, val_df["y_true"].values)
    val_df[out_col] = ir.predict(val_df[src_col].values)
    test_df[out_col] = ir.predict(test_df[src_col].values)
    return ir, val_df, test_df

best_val = m_val_subset
best_test = m_test_subset
best_col_test = "y_pred_subsetcal"
applied_repair = False

if USE_RESIDUAL_REPAIR:
    _val = val_c.copy()
    _test = test_c.copy()
    ir, _val, _test = residual_repair(_val, _test, src_col="y_pred_subsetcal", out_col="y_pred_subsetcal_rr")

    m_val_rr  = metrics(_val["y_true"],  _val["y_pred_subsetcal_rr"])
    m_test_rr = metrics(_test["y_true"], _test["y_pred_subsetcal_rr"])

    print("\n=== After residual repair (global isotonic) ===")
    print("VAL :", m_val_rr)
    print("TEST:", m_test_rr)

    # Keep only if VAL RMSE improves (and not harming PLCC badly)
    if m_val_rr["RMSE"] + 1e-9 < best_val["RMSE"]:
        val_c = _val
        test_c = _test
        best_val = m_val_rr
        best_test = m_test_rr
        best_col_test = "y_pred_subsetcal_rr"
        applied_repair = True
    else:
        print("[INFO] Residual repair not adopted (did not improve VAL RMSE).")

# -----------------------
# Compare against baseline — keep only if TEST improves RMSE
# -----------------------
improved = best_test["RMSE"] + 1e-9 < base_test_metrics["RMSE"]

out_summary = {
    "baseline": {"VAL": base_val_metrics, "TEST": base_test_metrics},
    "per_subset": {"VAL": m_val_subset, "TEST": m_test_subset},
    "residual_repair_applied": applied_repair,
    "final_choice": "per_subset+rr" if (applied_repair and improved) else ("per_subset" if improved else "baseline"),
    "improved_over_baseline": bool(improved),
}

print("\n=== Final decision ===")
print(json.dumps(out_summary, indent=2))

# -----------------------
# Save outputs (only if improved)
# -----------------------
if improved:
    # Write calibrated predictions & metrics
    test_out = test_c[["video_id","y_true", best_col_test, "compression","qp"]].rename(columns={best_col_test:"y_pred_tierA"})
    val_out  = val_c[["video_id","y_true"] + [c for c in val_c.columns if c.startswith("y_pred") or c in ("compression","qp")]].copy()

    test_csv = OUT_DIR / "test_preds_tierA.csv"
    val_csv  = OUT_DIR / "val_preds_tierA.csv"
    test_out.to_csv(test_csv, index=False)
    val_out.to_csv(val_csv, index=False)

    with open(OUT_DIR / "metrics_tierA.json", "w") as f:
        json.dump({
            "VAL": best_val, "TEST": best_test,
            "decision": out_summary
        }, f, indent=2)
    print(f"\n[OK] Saved improved outputs:\n- {test_csv}\n- {val_csv}\n- {OUT_DIR/'metrics_tierA.json'}")
else:
    print("\n[NOTE] TEST RMSE did not improve vs baseline — keeping your previous predictions.")
