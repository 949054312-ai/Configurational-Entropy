#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


STRUCT_KEYWORDS = ["r_global", "g_global", "r_first", "g_first"]

ENERGY_TERM_COLS = [
    "LJ-14",
    "Coulomb-14",
    "LJ-(SR)",
    "Disper.-corr.",
    "Coulomb-(SR)",
    "Coul.-recip.",
]


def clean_colname(x: str) -> str:
    # remove BOM, strip spaces, normalize internal whitespace
    s = str(x).replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s


def safe_std(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return np.nan
    return float(np.std(arr, ddof=1))


def safe_mean(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def safe_percentile(arr: np.ndarray, q: float) -> float:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))


def summarize_series(series: pd.Series) -> dict:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mean_val = safe_mean(arr)
    std_val = safe_std(arr)
    rel_std = std_val / abs(mean_val) if (not np.isnan(mean_val) and mean_val != 0 and not np.isnan(std_val)) else np.nan
    return {
        "n": int(np.sum(~np.isnan(arr))),
        "mean": mean_val,
        "std": std_val,
        "rel_std": rel_std,
        "p05": safe_percentile(arr, 5),
        "p95": safe_percentile(arr, 95),
    }


def pick_structural_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        name = str(c)
        if any(k in name for k in STRUCT_KEYWORDS):
            cols.append(c)
    return cols


def find_blend_column(df: pd.DataFrame) -> str:
    # Strategy:
    # 1) if a cleaned column equals "blends", use it
    # 2) otherwise use the first column
    cleaned = {c: clean_colname(c) for c in df.columns}
    for original, cleaned_name in cleaned.items():
        if cleaned_name.lower() == "blends":
            return original
    return df.columns[0]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate per blend and export two formats: "
            "rows (one row per blend) and wide (one column per blend)."
        )
    )
    ap.add_argument("--xlsx", type=str, default="Database-2.xlsx", help="Excel file path")
    ap.add_argument("--sheet", type=str, default=None, help="Sheet name (default: first sheet)")
    ap.add_argument("--keep_pctl", action="store_true", help="Keep 5th and 95th percentiles")
    ap.add_argument("--out_rows", type=str, default="cluster_summary_rows.csv", help="Row format output CSV")
    ap.add_argument("--out_wide", type=str, default="cluster_summary_wide_blends_as_columns.csv", help="Wide format output CSV")
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"File not found: {xlsx_path.resolve()}")

    df = pd.read_excel(xlsx_path, sheet_name=args.sheet if args.sheet else 0)

    # Clean column names in-place for reliable matching
    df.columns = [clean_colname(c) for c in df.columns]

    blend_col = "blends" if "blends" in df.columns else df.columns[0]

    required_cols = [blend_col, "Potential", "Total-Energy", "Entropy"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    # Identify structural columns and available energy terms
    struct_cols = pick_structural_columns(df)
    energy_term_cols = [c for c in ENERGY_TERM_COLS if c in df.columns]

    # Convert relevant columns to numeric (except blend_col)
    numeric_cols = ["Potential", "Total-Energy", "Entropy"] + struct_cols + energy_term_cols
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for blend, g in df.groupby(blend_col, dropna=False):
        # normalize blend label to avoid hidden spaces causing multiple groups
        blend_key = str(blend).strip()

        out = {"blends": blend_key, "n_rows": int(len(g))}

        # Entropy summary (usually constant per blend, but summarized robustly)
        sS = summarize_series(g["Entropy"])
        out["Entropy_mean"] = sS["mean"]
        out["Entropy_std"] = sS["std"]

        # 1) Energy landscape roughness (Potential)
        sU = summarize_series(g["Potential"])
        out["roughness_sigma_Potential"] = sU["std"]
        out["Potential_mean"] = sU["mean"]
        if args.keep_pctl:
            out["Potential_p05"] = sU["p05"]
            out["Potential_p95"] = sU["p95"]

        # 2) Local environment energy disorder proxy (Total-Energy)
        sE = summarize_series(g["Total-Energy"])
        out["proxy_sigma_Etotal"] = sE["std"]
        out["Etotal_mean"] = sE["mean"]
        if args.keep_pctl:
            out["Etotal_p05"] = sE["p05"]
            out["Etotal_p95"] = sE["p95"]

        # Structural descriptors (RDF related)
        for c in struct_cols:
            sc = summarize_series(g[c])
            out[f"{c}__mean"] = sc["mean"]
            out[f"{c}__std"] = sc["std"]
            out[f"{c}__relstd"] = sc["rel_std"]
            if args.keep_pctl:
                out[f"{c}__p05"] = sc["p05"]
                out[f"{c}__p95"] = sc["p95"]

        # Energy term decomposition (optional)
        for c in energy_term_cols:
            sc = summarize_series(g[c])
            out[f"{c}__mean"] = sc["mean"]
            out[f"{c}__std"] = sc["std"]
            out[f"{c}__relstd"] = sc["rel_std"]
            if args.keep_pctl:
                out[f"{c}__p05"] = sc["p05"]
                out[f"{c}__p95"] = sc["p95"]

        rows.append(out)

    summary_rows = pd.DataFrame(rows).sort_values(by="blends", kind="stable")
    summary_rows.to_csv(args.out_rows, index=False)

    # Wide output: metrics as rows, blends as columns
    wide = summary_rows.set_index("blends").T
    wide.to_csv(args.out_wide)

    print("Completed successfully.")
    print(f"Blend column used: {blend_col}")
    print(f"Number of blends: {summary_rows.shape[0]}")
    print(f"Saved row format: {args.out_rows}")
    print(f"Saved wide format: {args.out_wide}")
    print("First 5 blends:")
    print(summary_rows["blends"].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
