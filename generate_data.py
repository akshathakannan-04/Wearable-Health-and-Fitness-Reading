"""
Wearable Health Analytics & Fitness Readiness Dashboard
======================================================
Module: generate_data.py

Generates synthetic wearable health telemetry, performs rolling window
feature engineering with leakage prevention, calculates a multi-factor
Fitness Readiness Score (0-100), and trains/evaluates predictive models
(Random Forest and XGBoost).
"""

import os
import tempfile

# Ensure Matplotlib config directory is writable across environments
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Scikit-learn & Modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional XGBoost import with safe fallback
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ==============================================================================
# 1. SYNTHETIC DATA GENERATION
# ==============================================================================

def generate_synthetic_wearable_data(
    num_users: int = 50,
    num_days: int = 60,
    end_date: str = "2026-10-23",
    seed: int = 42,
    kaggle_fitbit_path: str = None
) -> pd.DataFrame:
    """
    Generates plausible, reproducible wearable health telemetry across 50 users
    over 60 consecutive days (3,000 rows).

    Optionally adapts a Kaggle Fitbit dataset if provided.
    """
    if kaggle_fitbit_path and os.path.exists(kaggle_fitbit_path):
        print(f"Loading external dataset from {kaggle_fitbit_path}...")
        df_ext = pd.read_csv(kaggle_fitbit_path)
        # Validate or map columns if necessary
        return df_ext

    np.random.seed(seed)
    date_range = pd.date_range(end=end_date, periods=num_days, freq="D")

    # Define user personas to simulate realistic biological heterogeneity:
    # 30% Sedentary, 50% Active/Moderate, 20% Athletic/High Activity
    personas = (
        ["sedentary"] * int(num_users * 0.3) +
        ["moderate"] * int(num_users * 0.5) +
        ["athletic"] * (num_users - int(num_users * 0.3) - int(num_users * 0.5))
    )
    np.random.shuffle(personas)

    records = []

    for user_idx in range(1, num_users + 1):
        user_id = f"USER_{user_idx:03d}"
        persona = personas[user_idx - 1]

        # Baseline parameters per persona
        if persona == "sedentary":
            base_steps = np.random.uniform(4000, 6000)
            base_cals = np.random.uniform(200, 350)
            base_rhr = np.random.uniform(66, 75)
            base_sleep = np.random.uniform(6.2, 7.2)
            base_stress = np.random.uniform(45, 70)
        elif persona == "moderate":
            base_steps = np.random.uniform(7500, 10500)
            base_cals = np.random.uniform(400, 600)
            base_rhr = np.random.uniform(58, 66)
            base_sleep = np.random.uniform(7.0, 8.0)
            base_stress = np.random.uniform(30, 55)
        else:  # athletic
            base_steps = np.random.uniform(12000, 16000)
            base_cals = np.random.uniform(650, 950)
            base_rhr = np.random.uniform(50, 58)
            base_sleep = np.random.uniform(7.5, 8.5)
            base_stress = np.random.uniform(20, 45)

        # Generate day-to-day correlated telemetry with seasonal/day-of-week variation
        for d_idx, dt in enumerate(date_range):
            is_weekend = dt.weekday() >= 5
            weekend_step_mult = 1.15 if persona != "sedentary" else 0.90
            weekend_sleep_bonus = 0.5 if is_weekend else 0.0

            # Day-to-day noise
            sleep_duration = np.clip(
                base_sleep + weekend_sleep_bonus + np.random.normal(0, 0.6),
                4.5,
                10.0
            )

            # Restorative sleep distribution (Deep: 15-25%, REM: 20-25%)
            deep_ratio = np.clip(np.random.uniform(0.16, 0.24), 0.10, 0.30)
            rem_ratio = np.clip(np.random.uniform(0.20, 0.26), 0.12, 0.35)
            deep_sleep = round(sleep_duration * deep_ratio, 2)
            rem_sleep = round(sleep_duration * rem_ratio, 2)

            # Poor sleep elevates resting heart rate and stress
            sleep_deficit = max(0.0, 7.5 - sleep_duration)
            rhr_noise = np.random.normal(0, 1.2) + (sleep_deficit * 0.8)
            resting_hr = round(float(np.clip(base_rhr + rhr_noise, 45, 95)), 1)

            # Stress score (0-100) influenced by sleep deficit and fatigue
            stress = int(np.clip(
                base_stress + (sleep_deficit * 6) + np.random.normal(0, 7),
                10,
                95
            ))

            # Daily steps and active calories
            step_noise = np.random.normal(1.0, 0.15)
            steps = int(np.clip(
                base_steps * (weekend_step_mult if is_weekend else 1.0) * step_noise - (sleep_deficit * 400),
                1500,
                28000
            ))

            # Active calories closely correlate with steps and physiological effort
            cals_per_step = base_cals / base_steps
            active_cals = int(np.clip(
                steps * cals_per_step + np.random.normal(0, 30),
                100,
                1800
            ))

            records.append({
                "user_id": user_id,
                "date": dt.strftime("%Y-%m-%d"),
                "total_steps": steps,
                "active_calories": active_cals,
                "sleep_duration_hours": round(sleep_duration, 2),
                "rem_sleep_hours": rem_sleep,
                "deep_sleep_hours": deep_sleep,
                "resting_heart_rate": resting_hr,
                "stress_score": stress
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["user_id", "date"]).reset_index(drop=True)
    return df


# ==============================================================================
# 2. FEATURE ENGINEERING & READINESS SCORE
# ==============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes required rolling window metrics per user while preventing
    future data leakage into historical baselines.

    Calculates:
    - 7-day rolling average of steps.
    - 7-day rolling average of active calories.
    - 5-day sleep consistency using rolling standard deviation.
    - Resting heart rate deviation from the previous 14-day baseline (strictly prior days).
    - Sleep quality ratio using REM and deep sleep.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["user_id", "date"]).reset_index(drop=True)

    # 1. 7-day rolling average of steps
    df["steps_7d_avg"] = (
        df.groupby("user_id")["total_steps"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        .round(1)
    )

    # 2. 7-day rolling average of active calories
    df["calories_7d_avg"] = (
        df.groupby("user_id")["active_calories"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        .round(1)
    )

    # 3. 5-day sleep consistency using rolling standard deviation
    # Higher std = irregular sleep schedule; fill initial window with 0.0
    df["sleep_5d_std"] = (
        df.groupby("user_id")["sleep_duration_hours"]
        .transform(lambda s: s.rolling(window=5, min_periods=2).std())
        .fillna(0.0)
        .round(2)
    )

    # 4. Resting heart rate deviation from PREVIOUS 14-day baseline
    # CRITICAL: We use shift(1) so today's RHR does not leak into its own baseline!
    # For day 1 where shift(1) is NaN, fallback to the day's own value (dev = 0).
    df["rhr_14d_baseline"] = (
        df.groupby("user_id")["resting_heart_rate"]
        .transform(lambda s: s.shift(1).rolling(window=14, min_periods=1).mean())
    )
    # Fill day 1 baseline with current RHR so initial deviation starts at 0
    df["rhr_14d_baseline"] = df["rhr_14d_baseline"].fillna(df["resting_heart_rate"]).round(1)
    df["rhr_14d_baseline_dev"] = (df["resting_heart_rate"] - df["rhr_14d_baseline"]).round(2)

    # 5. Sleep quality ratio: (REM + Deep) / Total Sleep
    df["sleep_quality_ratio"] = (
        (df["rem_sleep_hours"] + df["deep_sleep_hours"]) / df["sleep_duration_hours"]
    ).clip(0.0, 1.0).round(3)

    return df


def calculate_readiness_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a transparent, illustrative Fitness Readiness Score (0-100):
    - Sleep and recovery: 35%
    - Activity balance: 25%
    - Resting heart rate deviation: 25%
    - Stress: 15%

    Note: Educational and illustrative metric only; not a medical diagnostic.
    """
    df = df.copy()

    # --- A. Sleep & Recovery Subscore (0-100, 35% weight) ---
    # Duration component (ideal: 7.5 - 8.5 hrs)
    duration_score = np.clip(100 - np.abs(df["sleep_duration_hours"] - 8.0) * 18, 20, 100)
    # Quality component (ideal restorative ratio ~40-45%)
    quality_score = np.clip((df["sleep_quality_ratio"] / 0.42) * 100, 20, 100)
    # Consistency component (penalize sleep std > 1.0 hr)
    consistency_score = np.clip(100 - df["sleep_5d_std"] * 25, 30, 100)
    sleep_subscore = (0.50 * duration_score + 0.35 * quality_score + 0.15 * consistency_score)

    # --- B. Activity Balance Subscore (0-100, 25% weight) ---
    # Compares acute daily steps against 7-day average (Acute-to-Chronic ratio)
    step_ratio = df["total_steps"] / (df["steps_7d_avg"].replace(0, 1))
    # Sweet spot ratio is ~0.85 - 1.25. Spikes > 1.6 indicate fatigue/overexertion risk.
    activity_subscore = np.clip(100 - np.abs(step_ratio - 1.05) * 45, 20, 100)

    # --- C. Resting Heart Rate Deviation Subscore (0-100, 25% weight) ---
    # Elevated RHR (+3 to +8 bpm above baseline) indicates systemic strain/illness/incomplete recovery.
    # At or below baseline is optimal.
    rhr_dev = df["rhr_14d_baseline_dev"]
    # Penalty for positive deviation (elevated RHR)
    rhr_subscore = np.where(
        rhr_dev <= 0,
        np.clip(95 - rhr_dev * 1.5, 85, 100),  # Well-recovered bonus
        np.clip(95 - (rhr_dev * 10), 10, 95)   # Strain penalty
    )

    # --- D. Stress Subscore (0-100, 15% weight) ---
    # Lower wearable stress score -> higher recovery readiness
    stress_subscore = np.clip(100 - df["stress_score"], 10, 100)

    # Weighted composite score (0-100)
    raw_readiness = (
        0.35 * sleep_subscore +
        0.25 * activity_subscore +
        0.25 * rhr_subscore +
        0.15 * stress_subscore
    )

    df["Readiness_Score"] = np.clip(raw_readiness, 0, 100).round(1)
    return df


# ==============================================================================
# 3. MACHINE LEARNING: TRAINING & EVALUATION
# ==============================================================================

def train_and_evaluate_models(
    df: pd.DataFrame,
    test_days: int = 12,
    output_model_path: str = "trained_model.pkl",
    output_plot_path: str = "feature_importance.png"
):
    """
    Trains and compares Random Forest and XGBoost Regressors using a strict
    chronological train/test split.

    Avoids direct formula component leakage (e.g. subscores) so the model learns
    predictive patterns from raw telemetry and rolling context.
    """
    df_sorted = df.sort_values(by="date").copy()
    dates = df_sorted["date"].drop_duplicates().sort_values()

    cutoff_date = dates.iloc[-test_days]
    print(f"Chronological split cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"Training period: {dates.iloc[0].strftime('%Y-%m-%d')} to {dates.iloc[-test_days-1].strftime('%Y-%m-%d')}")
    print(f"Testing period:  {cutoff_date.strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}")

    # Feature selection: raw metrics + engineered rolling temporal indicators
    # Note: We deliberately exclude the arithmetic formula components to prevent direct leakage
    feature_cols = [
        "total_steps",
        "active_calories",
        "sleep_duration_hours",
        "rem_sleep_hours",
        "deep_sleep_hours",
        "resting_heart_rate",
        "stress_score",
        "steps_7d_avg",
        "calories_7d_avg",
        "sleep_5d_std"
    ]
    target_col = "Readiness_Score"

    train_mask = df_sorted["date"] < cutoff_date
    test_mask = df_sorted["date"] >= cutoff_date

    X_train = df_sorted.loc[train_mask, feature_cols]
    y_train = df_sorted.loc[train_mask, target_col]
    X_test = df_sorted.loc[test_mask, feature_cols]
    y_test = df_sorted.loc[test_mask, target_col]

    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Model 1: Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=9,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS (Chronological Test Set)")
    print("=" * 50)
    print(f"1. Random Forest Regressor:")
    print(f"   - RMSE : {rf_rmse:.3f}")
    print(f"   - MAE  : {rf_mae:.3f}")
    print(f"   - R²   : {rf_r2:.4f}")

    best_model = rf_model
    best_model_name = "Random Forest Regressor"
    best_metrics = {"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2}

    # Model 2: XGBoost Regressor (with fallback check)
    xgb_metrics = None
    if XGBOOST_AVAILABLE:
        xgb_model = XGBRegressor(
            n_estimators=130,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)

        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
        xgb_mae = mean_absolute_error(y_test, xgb_preds)
        xgb_r2 = r2_score(y_test, xgb_preds)

        print(f"\n2. XGBoost Regressor:")
        print(f"   - RMSE : {xgb_rmse:.3f}")
        print(f"   - MAE  : {xgb_mae:.3f}")
        print(f"   - R²   : {xgb_r2:.4f}")

        xgb_metrics = {"rmse": xgb_rmse, "mae": xgb_mae, "r2": xgb_r2}

        # Compare models and select best
        if xgb_r2 > rf_r2:
            best_model = xgb_model
            best_model_name = "XGBoost Regressor"
            best_metrics = xgb_metrics
    else:
        print("\n2. XGBoost is unavailable in current environment; falling back to Random Forest only.")

    print(f"\nSelected Best Model: {best_model_name} (R² = {best_metrics['r2']:.4f})")
    print("=" * 50)

    # 4. Feature Importance Plot
    importances = best_model.feature_importances_
    indices = np.argsort(importances)
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(9, 5.5))
    bars = plt.barh(range(len(sorted_features)), sorted_importances, color="#1f77b4", edgecolor="#0e4f7a")
    plt.yticks(range(len(sorted_features)), sorted_features, fontsize=10)
    plt.xlabel("Relative Feature Importance", fontsize=11, fontweight="bold")
    plt.title(f"Feature Importance ({best_model_name})", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
                 va="center", ha="left", fontsize=9, color="#222")

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved feature importance plot to {output_plot_path}")

    # 5. Save best model and metadata
    model_payload = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_cols,
        "metrics": best_metrics,
        "all_metrics": {
            "Random Forest": {"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2},
            "XGBoost": xgb_metrics
        }
    }
    joblib.dump(model_payload, output_model_path)
    print(f"Saved trained model payload to {output_model_path}")

    return best_model, best_metrics


# ==============================================================================
# 4. MAIN ORCHESTRATION PIPELINE
# ==============================================================================

def run_pipeline(
    csv_path: str = "wearable_data.csv",
    model_path: str = "trained_model.pkl",
    plot_path: str = "feature_importance.png",
    num_users: int = 50,
    num_days: int = 60
):
    """
    End-to-end execution:
    1. Generates 50 users x 60 days synthetic dataset (3,000 rows).
    2. Performs rolling window feature engineering and calculates Readiness_Score.
    3. Saves wearable_data.csv.
    4. Trains Random Forest & XGBoost with chronological train/test split.
    5. Saves feature_importance.png and trained_model.pkl.
    """
    print("=" * 60)
    print("WEARABLE HEALTH ANALYTICS: DATA GENERATION & ML PIPELINE")
    print("=" * 60)

    # 1. Generate Raw Data
    print(f"\n[Step 1] Generating synthetic wearable telemetry ({num_users} users, {num_days} days)...")
    df_raw = generate_synthetic_wearable_data(num_users=num_users, num_days=num_days, seed=42)

    # 2. Engineer Features & Calculate Readiness
    print("[Step 2] Engineering rolling features & calculating Readiness Score...")
    df_features = engineer_features(df_raw)
    df_final = calculate_readiness_score(df_features)

    # 3. Save CSV
    df_final.to_csv(csv_path, index=False)
    print(f"[Step 3] Dataset successfully saved to '{csv_path}' ({len(df_final)} rows, {df_final.shape[1]} columns).")

    # 4. Train & Evaluate Models
    print("\n[Step 4] Training and evaluating Random Forest & XGBoost regressors...")
    best_model, best_metrics = train_and_evaluate_models(
        df_final,
        test_days=12,
        output_model_path=model_path,
        output_plot_path=plot_path
    )

    print("\nPipeline completed successfully!")
    return df_final, best_model


if __name__ == "__main__":
    run_pipeline()

