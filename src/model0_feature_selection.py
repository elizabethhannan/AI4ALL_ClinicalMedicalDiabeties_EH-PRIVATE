"""
============================================================================
 GE-79 / CDED  -  Group 6C  |  AI4ALL Ignite 2026
 MODEL 0:  FEATURE SELECTION  (Random Forest importance ranking)
 Label stamped on the figure:  GE-79 · Feature Selection
============================================================================
 WHERE THIS FILE GOES (PyCharm):
     AI4ALL_ML-Diabetes_-Ver_1_A/
     └── src/
         └── model0_feature_selection.py      <-- THIS FILE

 IT READS:
     ../data/GE79_MASTER_DATASET_V1.csv   (input dataset)

 IT WRITES (to ../outputs/):
     model0_FINAL_FEATURES.csv                 <-- locked biomarker features Models 1 & 3 reuse
     model0_feature_importance_fullscope.csv   - full ranking of every candidate feature
     model0_fig_feature_selection.png          - feature-selection importance chart
     model0_fig_target_distribution.png        - target class distribution chart

 ----------------------------------------------------------------------------
 RUN ORDER:  RUN THIS FILE FIRST. It creates model0_FINAL_FEATURES.csv, which
             Model 1 (Logistic Regression) and Model 3 (Random Forest) read.
             Model 2 is currently a placeholder/guide.
 ----------------------------------------------------------------------------

 WHAT THIS DOES (Phase 4 — feature selection)
   FULL SCOPE: all six biomarker domains are offered as candidate features.
   A Random Forest scores each feature's importance. Because the dataset is
   small (n=75), a single ranking is noisy, so importance is AVERAGED over
   20 random seeds for stability. A short list of science-based ANCHORS
   (glycemic, BP, vasoreactivity, perfusion, white-matter) is always kept,
   because the CDED literature says they matter. The final set is:
       FINAL_FEATURES = top-12 ranked features UNION science anchors.

 TARGET ENCODING
   0 = No Impairment   (MMSE >= 28, "Normal")
   1 = Impaired        (MMSE 25-27)

 LEAKAGE CONTROL
   - MMSE (defines the label) is NOT a feature -> no target leakage.
   - One row per patient (Visit 2)             -> no repeated-measures leakage.
   - Imputation + scaling happen INSIDE the importance pipeline.
============================================================================
"""

import runpy
import sys
import warnings
from pathlib import Path


def _run_lightweight_streamlit_app_if_needed():
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return

    if get_script_run_ctx(suppress_warning=True) is not None:
        app_dir = Path(__file__).resolve().parents[1] / "app"
        app_dir_str = str(app_dir)
        if app_dir_str in sys.path:
            sys.path.remove(app_dir_str)
        sys.path.insert(0, app_dir_str)
        loaded_helper = sys.modules.get("echarts_components")
        if loaded_helper is not None and not str(getattr(loaded_helper, "__file__", "")).startswith(app_dir_str):
            del sys.modules["echarts_components"]
        app_path = app_dir / "model0_streamlit.py"
        runpy.run_path(str(app_path), run_name="__main__")
        st.stop()


_run_lightweight_streamlit_app_if_needed()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from neumorphic_visualizations import save_feature_selection, save_target_distribution

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "GE79_MASTER_DATASET_V1.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "GE-79"
CLASS_0_LABEL = "No Impairment"
CLASS_1_LABEL = "Impaired"
ENCODING = f"0 = {CLASS_0_LABEL}, 1 = {CLASS_1_LABEL}"

TARGET_COL = "cognitive_status_label"
TARGET_MAP = {"Normal": 0, "Mild Impairment": 1}

# Never-predictors: ids, target, 100%-missing column, duplicate of `group`
DROP_COLS = ["patient_id", "datapatient_id", "visit", "cognitive_status_code",
             "cognitive_status_label", "gait_walk1_distance_m", "dm_status"]

# Keep diabetes_duration (45% missing) WITH a missingness flag
ADD_MISSING_FLAG_FOR = ["diabetes_duration"]

# Science-based anchors — kept regardless of ranking (CDED literature support)
SCIENCE_ANCHORS = ["glucose_mg_dl", "fasting_glucose_mg_dl", "hba1c_percent",
                   "daytime_sbp", "global_vasoreactivity",
                   "perfusion_whole_brain_baseline_whole", "wmh_registered",
                   "diabetes_duration"]

N_TOP_FEATURES = 12          # how many data-ranked features to keep
N_STABILITY_SEEDS = 20       # average importance over this many seeds

TEAL = "#2a9d8f"


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    missing_cols = [col for col in ADD_MISSING_FLAG_FOR if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required column(s): {missing_cols}")
    for col in ADD_MISSING_FLAG_FOR:                 # missingness indicator
        df[f"{col}_missing"] = df[col].isnull().astype(int)
    y = df[TARGET_COL].map(TARGET_MAP)
    X = df.drop(columns=DROP_COLS, errors="ignore")
    print(f"Loaded {DATA_PATH.name}: {len(df)} patients, "
          f"{X.shape[1]} candidate features (full scope)")
    print("Target:", df[TARGET_COL].value_counts().to_dict())
    return X, y


# ---------------------------------------------------------------------------
# 2. PREPROCESSOR (fit inside the importance pipeline -> no leakage)
# ---------------------------------------------------------------------------
def build_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include="object").columns.tolist()
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, numeric),
                              ("cat", cat_pipe, categorical)])


# ---------------------------------------------------------------------------
# 3. FEATURE SELECTION — stability-averaged RF importance + science anchors
# ---------------------------------------------------------------------------
def select_features(X, y):
    pre = build_preprocessor(X)
    runs = []
    for seed in range(N_STABILITY_SEEDS):
        rf = Pipeline([("pre", pre),
                       ("clf", RandomForestClassifier(
                            n_estimators=400, max_depth=6, min_samples_leaf=3,
                            class_weight="balanced", random_state=seed))])
        rf.fit(X, y)
        names = rf.named_steps["pre"].get_feature_names_out()
        runs.append(pd.Series(rf.named_steps["clf"].feature_importances_, index=names))

    imp = pd.concat(runs, axis=1)
    ranking = (pd.DataFrame({"feature": imp.index,
                             "importance_mean": imp.mean(axis=1).values,
                             "importance_std": imp.std(axis=1).values})
               .sort_values("importance_mean", ascending=False))
    ranking["feature"] = (ranking["feature"]
                          .str.replace("num__", "", regex=False)
                          .str.replace("cat__", "", regex=False))
    ranking.to_csv(OUT_DIR / "model0_feature_importance_fullscope.csv", index=False)

    top = list(dict.fromkeys(ranking["feature"].head(N_TOP_FEATURES)))
    final = sorted(set(top) | set(SCIENCE_ANCHORS))
    print(f"\nSelected {len(final)} FINAL_FEATURES "
          f"(top {N_TOP_FEATURES} ranked + {len(SCIENCE_ANCHORS)} science anchors):")
    for f in final:
        print("   -", f)

    save_feature_selection(ranking, OUT_DIR / "model0_fig_feature_selection.png")
    print("Saved model0_fig_feature_selection.png")
    return final


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print(" MODEL 0 — FEATURE SELECTION  (GE-79 · Random Forest importance)")
    print("=" * 64)
    X, y = load_data()
    save_target_distribution(y, OUT_DIR / "model0_fig_target_distribution.png")
    print("Saved model0_fig_target_distribution.png")
    final_features = select_features(X, y)
    pd.Series(final_features, name="final_features").to_csv(
        OUT_DIR / "model0_FINAL_FEATURES.csv", index=False)
    print(f"\nSaved model0_FINAL_FEATURES.csv ({len(final_features)} features).")
    print("Next: run Model 1 (Logistic Regression) and Model 3 (Random Forest).")
    print("Model 2 is currently a placeholder/guide and can be skipped.")


if __name__ == "__main__":
    main()
