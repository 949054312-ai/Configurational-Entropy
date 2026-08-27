# -*- coding: utf-8 -*-
"""
Direct external prediction with the ORIGINAL fitted Ridge Regression +
Correlation-based feature selection model.

Important:
1. This script DOES NOT retrain the model.
2. This script DOES NOT call model.fit().
3. This script DOES NOT redo correlation-based feature selection.
4. It uses the feature_names stored inside the original .pkl, builds those
   blend-level descriptors with the same aggregation rules as the training code,
   then calls model.predict() only.

Expected files in the same folder:
    original_ridge_correlation_model.pkl
    molecular_descriptors_original.xlsx
    New_blends_input_original_model.xlsx

Output:
    Original_Ridge_direct_predictions.csv
"""

import pickle
import numpy as np
import pandas as pd

MODEL_FILE = "original_ridge_correlation_model.pkl"
DESCRIPTOR_FILE = "molecular_descriptors_original.xlsx"
INPUT_FILE = "New_blends_input_original_model.xlsx"
OUTPUT_FILE = "Original_Ridge_direct_predictions.csv"


def parse_molecules(cell):
    if pd.isna(cell):
        return []
    s = str(cell)
    if "\n" in s:
        return [x.strip() for x in s.split("\n") if x.strip()]
    if "<br>" in s:
        return [x.strip() for x in s.split("<br>") if x.strip()]
    return [x.strip() for x in s.split() if x.strip()]


def parse_weights(cell):
    return [float(x) for x in str(cell).split(":")]


def weighted_median_original(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values)
    sorted_idx = np.argsort(values[mask])
    sorted_weights = weights[mask][sorted_idx]
    sorted_values = values[mask][sorted_idx]
    cum_weights = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cum_weights, 0.5)
    if median_idx >= len(sorted_values):
        median_idx = len(sorted_values) - 1
    return float(sorted_values[median_idx])


def create_selected_features(molecules, raw_weights, descriptor_df, feature_names):
    weights = np.asarray(raw_weights, dtype=float)
    weights = weights / weights.sum()

    lookup = descriptor_df.set_index("name")
    out = {}

    for feature in feature_names:
        if feature == "num_molecules":
            out[feature] = len(molecules)
            continue

        if feature.endswith("_weighted_median"):
            desc = feature[:-len("_weighted_median")]
            values = [lookup.loc[m, desc] for m in molecules]
            out[feature] = weighted_median_original(values, weights)

        elif feature.endswith("_weighted_mean"):
            desc = feature[:-len("_weighted_mean")]
            values = np.asarray([lookup.loc[m, desc] for m in molecules], dtype=float)
            mask = np.isfinite(values)
            out[feature] = float(np.average(values[mask], weights=weights[mask]))

        elif feature.endswith("_weighted_std"):
            desc = feature[:-len("_weighted_std")]
            values = np.asarray([lookup.loc[m, desc] for m in molecules], dtype=float)
            mask = np.isfinite(values)
            mean = np.average(values[mask], weights=weights[mask])
            var = np.average((values[mask] - mean) ** 2, weights=weights[mask])
            out[feature] = float(np.sqrt(var))

        else:
            raise ValueError(f"Unexpected stored feature: {feature}")

    return out


def main():
    with open(MODEL_FILE, "rb") as f:
        package = pickle.load(f)

    if package.get("feature_method") != "Correlation-based":
        raise RuntimeError("The loaded pkl is not the required Correlation-based model.")
    if package.get("performance", {}).get("Model") != "Ridge Regression":
        raise RuntimeError("The loaded pkl is not the required Ridge Regression model.")

    model = package["model"]
    feature_names = package["feature_names"]

    descriptors = pd.read_excel(DESCRIPTOR_FILE)
    blends = pd.read_excel(INPUT_FILE, sheet_name="New_Blends_Input")

    feature_rows = []
    for _, row in blends.iterrows():
        molecules = parse_molecules(row["Blend_molecules"])
        weights = parse_weights(row["Weight_ratio"])

        if len(molecules) != len(weights):
            raise ValueError(
                f"{row['Blend_ID']}: molecule count ({len(molecules)}) "
                f"does not equal weight count ({len(weights)})."
            )

        features = create_selected_features(
            molecules, weights, descriptors, feature_names
        )
        feature_rows.append(features)

    X_new = pd.DataFrame(feature_rows, columns=feature_names)

    # This is the ONLY model operation used for external prediction.
    y_pred = model.predict(X_new)

    out = blends.copy()
    out["Predicted_Sconf_mean"] = y_pred
    out["Prediction_Mode"] = "DIRECT original model.predict(), no fit"
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Model:", package["performance"]["Model"])
    print("Feature method:", package["feature_method"])
    print("Original R2:", package["performance"]["R2"])
    print("Original MAE:", package["performance"]["MAE"])
    print("Original RMSE:", package["performance"]["RMSE"])
    print("\nDirect predictions:")
    print(out[["Blend_ID", "Predicted_Sconf_mean"]].to_string(index=False))
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
