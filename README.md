Website : https://wearable-health-and-fitness-reading-t5bdweqfxerdjvbv4ydkcn.streamlit.app/

# Wearable Health Analytics & Fitness Readiness Dashboard

An end-to-end Python Data Science and Machine Learning ecosystem project that synthesizes wearable biometric time series, computes leakage-free rolling physiological features, evaluates a transparent multi-component Fitness Readiness Score (0–100), benchmarks regression models (Random Forest vs. XGBoost) using chronological evaluation, and serves an interactive web dashboard using Streamlit and Plotly.

---

## 1. Abstract

Modern consumer wearables (e.g., Apple Watch, Whoop, Oura, Garmin, Fitbit) collect continuous biometric telemetry such as step volume, active energy expenditure, heart rate metrics, and multi-stage sleep architecture. Translating these discrete, high-frequency signals into an actionable daily recovery recommendation requires systematic data engineering and predictive modeling. This project implements a complete wearable analytics pipeline that extracts rolling temporal baselines, mitigates chronological data leakage, calculates an interpretable composite Fitness Readiness Score, trains machine learning models to capture complex non-linear feature interactions, and presents personalized insights through an interactive Streamlit dashboard.

---

## 2. Problem Statement

Raw wearable telemetry is often noisy, fragmented, and disconnected from actionable guidance. Users face two primary dilemmas:
1. **Acute vs. Chronic Fatigue Mismatch:** Exercising heavily after cumulative sleep debt or during elevated resting heart rate (RHR) spikes increases the risk of overtraining syndrome, autonomic fatigue, and injury.
2. **Black-Box Proprietary Scores:** Commercial fitness algorithms rarely reveal their feature weighting, component trade-offs, or rolling baseline calculations, leaving athletes and researchers unable to audit how recovery readiness is derived.

There is a need for a modular, transparent, and reproducible Data Science pipeline that demonstrates how to clean, engineer, model, and visualize wearable telemetry without historical leakage.

---

## 3. Project Objectives

- **Generate Reproducible Telemetry:** Create a plausible synthetic dataset of 50 users across 60 days (3,000 records) representing diverse physiological archetypes (sedentary, moderate, athletic).
- **Leakage-Free Feature Engineering:** Compute rolling step averages, sleep consistency metrics, and resting heart rate baseline deviations using strictly prior historical data (`shift(1)` windows) to prevent target and baseline leakage.
- **Formulate Transparent Readiness Scoring:** Calculate a multi-factor score from 0 to 100 balancing sleep quality/duration (35%), acute-to-chronic workload (25%), resting heart rate deviation (25%), and stress (15%).
- **Chronological Machine Learning:** Train and compare Random Forest Regressor and XGBoost Regressor across chronological train/test splits, evaluating RMSE, MAE, and $R^2$.
- **Interactive Web Interface:** Provide a rich Streamlit dashboard with user selection, date filters, key metric cards, dual-axis Plotly trend graphs, sleep architecture breakdowns, what-if predictive tools, and CSV exports.

---

## 4. Architecture Diagram

```mermaid
flowchart LR
    A[Data Generation\n50 Users x 60 Days] --> B[Data Cleaning &\nValidation]
    B --> C[Feature Engineering\nRolling Windows & Baselines]
    C --> D[Fitness Readiness\nComposite Score 0-100]
    D --> E[Chronological ML Split\nTrain: First 80% | Test: Last 20%]
    E --> F[Model Benchmarking\nRandom Forest vs. XGBoost]
    F --> G[Model Evaluation &\nFeature Importance]
    G --> H[Streamlit Interactive\nDashboard & Plotly]
```

**Pipeline Flow:**
`Data Generation` $\rightarrow$ `Data Cleaning` $\rightarrow$ `Feature Engineering` $\rightarrow$ `Readiness Score` $\rightarrow$ `Machine Learning` $\rightarrow$ `Model Evaluation` $\rightarrow$ `Streamlit Dashboard`

---

## 5. Technologies Used

- **Language:** Python 3.10+
- **Data Manipulation & Math:** Pandas, NumPy
- **Machine Learning & Modeling:** Scikit-learn, XGBoost, Joblib
- **Visualization & Graphics:** Plotly (interactive charts), Matplotlib (static feature importances)
- **Web Application Framework:** Streamlit

---

## 6. Dataset Description

The generated dataset is saved as `wearable_data.csv` containing **3,000 observations** (50 unique users observed across 60 consecutive days).

### Core Telemetry Columns:
| Column Name | Type | Description |
| :--- | :---: | :--- |
| `user_id` | String | Unique user identifier (`USER_001` through `USER_050`) |
| `date` | Date | Record timestamp (60 consecutive days) |
| `total_steps` | Integer | Daily ambulatory step count (1,500 – 28,000) |
| `active_calories` | Integer | Active energy expended (100 – 1,800 kcal) |
| `sleep_duration_hours` | Float | Total nightly sleep time (4.5 – 10.0 hours) |
| `rem_sleep_hours` | Float | Rapid Eye Movement cognitive restorative sleep (0.5 – 2.8 hrs) |
| `deep_sleep_hours` | Float | Slow-wave physical restorative sleep (0.5 – 2.5 hrs) |
| `resting_heart_rate` | Float | Sleeping resting heart rate (45.0 – 95.0 bpm) |
| `stress_score` | Integer | Autonomic nervous system stress metric (10 – 95) |

### Engineered Features:
| Feature Name | Computation Method | Rationale |
| :--- | :--- | :--- |
| `steps_7d_avg` | `groupby('user_id').rolling(7, min_periods=1).mean()` | Chronic physical activity volume |
| `calories_7d_avg` | `groupby('user_id').rolling(7, min_periods=1).mean()` | 7-day rolling energy baseline |
| `sleep_5d_std` | `groupby('user_id').rolling(5, min_periods=2).std().fillna(0)` | Schedule consistency (lower is better) |
| `rhr_14d_baseline` | `shift(1).rolling(14, min_periods=1).mean()` | Leakage-free historical resting heart rate |
| `rhr_14d_baseline_dev` | `resting_heart_rate - rhr_14d_baseline` | Physiological strain deviation |
| `sleep_quality_ratio` | `(rem_sleep_hours + deep_sleep_hours) / sleep_duration_hours` | Proportion of restorative sleep stages |
| `Readiness_Score` | Weighted composite $(0-100)$ | Daily estimated recovery readiness |

*Note: The generator also supports external Kaggle Fitbit CSV files via optional argument if configured, but functions completely self-contained out-of-the-box.*

---

## 7. Methodology

### A. Prevention of Data Leakage
In time series physiological modeling, calculating baselines that include the current observation introduces circular target leakage. To prevent this:
- Resting heart rate baseline is computed using `.shift(1)` before applying the 14-day rolling mean window.
- The train/test split is strictly **chronological**: the first 48 days (80% of history) are reserved for training, and the final 12 days (20% of history) serve as the unseen validation test set across all users.

### B. Readiness Score Formulation
The composite readiness index represents a weighted multi-factor heuristic:
$$\text{Readiness Score} = 0.35 \times S_{\text{sleep}} + 0.25 \times S_{\text{activity}} + 0.25 \times S_{\text{RHR}} + 0.15 \times S_{\text{stress}}$$

- **Sleep & Recovery ($35\%$):** Evaluates total duration against an 8-hour target, penalizes irregular 5-day variance, and rewards high restorative sleep ratios (REM + Deep).
- **Activity Balance ($25\%$):** Assesses the Acute-to-Chronic Workload Ratio (today's steps vs. 7-day rolling average) to identify under-activity or sudden overtraining spikes.
- **RHR Deviation ($25\%$):** Compares today's RHR to the preceding 14-day baseline. Elevated RHR signals fatigue or illness and incurs an escalating penalty.
- **Stress Component ($15\%$):** Inversely scores autonomic stress markers ($100 - \text{stress}$).

---

## 8. Machine Learning Benchmarks & Evaluation

Two non-linear regressors were trained on raw telemetry and rolling indicators without directly exposing formula subcomponents.

### Chronological Test Set Results (Unseen Final 12 Days):

| Model Architecture | RMSE | MAE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | `2.160` | `1.695` | `0.7595` |
| **XGBoost Regressor** | **`1.955`** | **`1.510`** | **`0.8030`** |

- **Best Model:** XGBoost Regressor was selected automatically and saved to `trained_model.pkl`.
- **Top Predictive Features:** Resting heart rate baseline deviation, sleep duration hours, stress score, and 7-day rolling step averages exhibited the strongest feature importances.
- **Methodological Note on Target Evaluation:** Performance on a synthetic or formula-derived target evaluates regression capacity to learn non-linear surrogate functions and interactions; **it does not establish real-world clinical health prediction accuracy.**

---

## 9. Installation & Execution

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Dataset & Train Models
```bash
python generate_data.py
```
*This command creates `wearable_data.csv`, evaluates Random Forest and XGBoost, outputs `feature_importance.png`, and saves `trained_model.pkl`.*

### Step 3: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
*(If run directly, `app.py` will automatically trigger `generate_data.py` if any required files are missing.)*

---

## 10. Dashboard Features

- **Sidebar Filtering:** Select individual user profiles (`USER_001`–`USER_050`) and filter arbitrary observation date windows.
- **Real-Time Biomarker Cards:** Displays current Readiness Score, Resting Heart Rate with baseline delta, Daily Steps with rolling delta, and Sleep Duration with restorative stage percentage.
- **Context-Aware Training Guidance:**
  - *Readiness $< 40$:* Suggests active rest, hydration, light mobility, and sleep prioritization.
  - *Readiness $40–70$:* Suggests steady-state cardio, moderate endurance, or maintenance strength sessions.
  - *Readiness $> 70$:* Suggests readiness for peak workouts, progressive overload, or high-intensity interval training.
- **Dual-Axis Activity & Readiness Chart:** Visualizes acute step volume and 7-day trends alongside readiness thresholds ($40$ and $70$).
- **Sleep Architecture Visualization:** Stacked stage analysis of Deep, REM, and Light sleep against recommended 8-hour targets, accompanied by 5-day schedule consistency graphs.
- **Interactive Machine Learning Simulator:** Test what-if scenarios using real-time sliders to observe model predictions.
- **Data Table & CSV Export:** Inspect recent records and export filtered user subsets directly via a download button.

---

## 11. Limitations & Future Scope

### Limitations
- **Synthetic Physiological Noise:** Synthetic data approximations may not capture rare arrhythmias, sleep apnea events, or seasonal viral infections.
- **Fixed Linear Weighting:** While clinically motivated, static component weights (35/25/25/15) do not adapt dynamically to specialized training phases (e.g., tapering vs. hypertrophy blocks).

### Future Scope
- **Continuous Heart Rate Variability (HRV):** Incorporating rMSSD and high-frequency HRV metrics for deeper autonomic nervous system analysis.
- **Personalized Adaptive Weighting:** Using reinforcement learning or self-reported subjective soreness to adjust factor weights per individual athlete.
- **Wearable Device API Ingestion:** Direct OAuth 2.0 connectors for Garmin Connect, Whoop API, and Apple HealthKit XML exports.

---

## 12. Disclaimer

**Educational & Demonstrative Use Only:** This software application, feature pipeline, and readiness estimation are engineered for educational, analytical, and computer science instructional purposes. They do not constitute medical advice, medical diagnoses, or clinically validated evaluations of autonomic fatigue, cardiovascular health, or overtraining syndrome. Always consult a licensed medical professional for personal health decisions.

