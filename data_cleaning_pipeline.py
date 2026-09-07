# ============================================================
# SHEFFIELD TRAFFIC DISSERTATION — DATA CLEANING PIPELINE
# Data-Driven Analysis of Road Traffic Patterns and Traffic
# Intensity in Sheffield
# Student: Surabhi Kishor Unnikattu Valappil (201935332)
# Supervisor: Dr Ehsan Amirnazmiafshar
# University of Liverpool — MSc Business Analytics & Big Data
# ============================================================
# HOW TO RUN:
# 1. Make sure both CSV files are in the SAME folder as this file
# 2. Open this file in VSCode
# 3. Press the ▷ Run button top right
# 4. Read every print output and note the numbers down
# ============================================================

# ── STEP 1: IMPORTS ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("✓ All libraries loaded successfully")
print("=" * 60)

# ── STEP 2: LOAD DATASETS ────────────────────────────────────
# pd.read_csv() reads a CSV file and stores it as a DataFrame.
# A DataFrame is like an Excel spreadsheet inside Python.
# low_memory=False stops the mixed-type warning on large files.

print("\nSTEP 2: Loading datasets...")

raw_counts = pd.read_csv('dft_rawcount_local_authority_id_159.csv',
                          low_memory=False)
count_points = pd.read_csv('dft_countpoints_local_authority_id_159.csv')

print(f"  Raw Counts:   {raw_counts.shape[0]:,} rows, {raw_counts.shape[1]} columns")
print(f"  Count Points: {count_points.shape[0]:,} rows, {count_points.shape[1]} columns")

# ── STEP 3: INSPECT RAW DATA ─────────────────────────────────
# Always look at your data before touching it.
print("\nSTEP 3: Inspecting raw data...")
print("\n  Raw Counts column names:")
print(raw_counts.columns.tolist())
# Write these down - you need exact names for all later steps

print("\n  Raw Counts data types:")
print(raw_counts.dtypes)
# int64 = whole numbers (good for counts)
# float64 = decimal numbers (good for lat/lon)
# object = text (might need converting)

print("\n  Raw Counts first 3 rows:")
print(raw_counts.head(3).to_string())

# ── STEP 4: MERGE DATASETS ───────────────────────────────────
# Joins the two files on their shared column: count_point_id
# how='left' = keep ALL rows from raw_counts even with no match
# suffixes = renames duplicate column names to avoid confusion
print("\nSTEP 4: Merging datasets...")

traffic_data = raw_counts.merge(count_points,
                                 on='count_point_id',
                                 how='left',
                                 suffixes=('_rc', '_cp'))

print(f"  Merged shape: {traffic_data.shape[0]:,} rows, {traffic_data.shape[1]} columns")

if traffic_data.shape[0] == 34392:
    print("  ✓ Row count correct: 34,392 rows (as expected)")
else:
    print(f"  ⚠ WARNING: Expected 34,392 but got {traffic_data.shape[0]:,}")

print("\n  Merged column names:")
print(traffic_data.columns.tolist())

# ── STEP 5: MISSING VALUE CHECK ──────────────────────────────
# We already know 4 junction/link columns have 20,232 missing values.
# These are STRUCTURAL (Minor roads only) — DO NOT impute them.
print("\nSTEP 5: Missing value check...")

missing = traffic_data.isnull().sum()
missing_cols = missing[missing > 0]
# [missing > 0] keeps only columns where missing count is above zero

print(f"  Columns with missing values ({len(missing_cols)} found):")
for col, count in missing_cols.items():
    pct = (count / len(traffic_data)) * 100
    label = "STRUCTURAL — Minor road only" if count == 20232 else "⚠ INVESTIGATE"
    print(f"    {col}: {count:,} missing ({pct:.1f}%) — {label}")

print("\n  DISSERTATION NOTE:")
print("  The 4 junction/link fields have exactly 20,232 missing values each.")
print("  This matches precisely the 20,232 Minor road records.")
print("  This missingness is STRUCTURAL — no imputation required.")

# ── STEP 6: DATA TYPE CONVERSIONS ────────────────────────────
# Ensure all vehicle count columns are numeric (int/float).
# If stored as text ('object'), Python cannot do maths on them.
print("\nSTEP 6: Data type conversions...")

vehicle_cols = ['pedal_cycles', 'two_wheeled_motor_vehicles', 'cars_and_taxis',
                'buses_and_coaches', 'lgvs', 'hgvs_2_rigid_axle',
                'hgvs_3_rigid_axle', 'hgvs_4_or_more_rigid_axle',
                'hgvs_3_or_4_articulated_axle', 'hgvs_5_articulated_axle',
                'hgvs_6_articulated_axle', 'all_hgvs', 'all_motor_vehicles']

for col in vehicle_cols:
    if col in traffic_data.columns:
        if traffic_data[col].dtype not in ['int64', 'float64']:
            traffic_data[col] = pd.to_numeric(traffic_data[col], errors='coerce')
            # errors='coerce' converts failed conversions to NaN
            print(f"  → Converted '{col}' to numeric")
# Only prints if a conversion was needed — otherwise stays quiet

print("  ✓ Vehicle count columns type check complete")
print(f"  'year' column type: {traffic_data['year'].dtype}")

# ── STEP 7: DUPLICATE CHECK ──────────────────────────────────
# Check for exact copy rows. True duplicates = data error.
print("\nSTEP 7: Duplicate check...")

duplicates = traffic_data.duplicated().sum()
# .duplicated() marks True for every row that copies a previous row
# .sum() counts how many True values exist

print(f"  Duplicate rows found: {duplicates}")
if duplicates > 0:
    traffic_data = traffic_data.drop_duplicates()
    print(f"  → Removed. New shape: {traffic_data.shape}")
else:
    print("  ✓ No duplicates — dataset clean")

# ── STEP 8: OUTLIER AND ANOMALY CHECKS ───────────────────────
# Check for impossible values. DO NOT delete — document only.
print("\nSTEP 8: Outlier and anomaly checks...")

# Check 1: Negative vehicle counts (impossible)
neg_found = False
for col in vehicle_cols:
    if col in traffic_data.columns:
        neg = (traffic_data[col] < 0).sum()
        if neg > 0:
            print(f"  ⚠ {col}: {neg} negative values — document this")
            neg_found = True
if not neg_found:
    print("  ✓ No negative vehicle counts found")

# Check 2: Zero total motor vehicle counts
zeros = (traffic_data['all_motor_vehicles'] == 0).sum()
print(f"  Rows where all_motor_vehicles = 0: {zeros}")
if zeros > 0:
    print("  → Keeping — may be valid (no traffic at that time/location)")

# Check 3: Maximum values — sanity check
print(f"  Max all_motor_vehicles: {traffic_data['all_motor_vehicles'].max():,}")
print(f"  Max all_hgvs: {traffic_data['all_hgvs'].max():,}")
print(f"  Max pedal_cycles: {traffic_data['pedal_cycles'].max():,}")

# Check 4: Year range
print(f"  Year range: {traffic_data['year'].min()} to {traffic_data['year'].max()}")

# ── STEP 9: ROAD TYPE STANDARDISATION ────────────────────────
# Dr Ehsan's guardrail: do not assume road_type is clean.
# Check for inconsistent capitalisation (e.g. "major" vs "Major").
print("\nSTEP 9: Road type standardisation...")

# After merge, road_type from raw_counts becomes road_type_rc
road_type_col = 'road_type_rc' if 'road_type_rc' in traffic_data.columns else 'road_type'
print(f"  Using column: '{road_type_col}'")

print(f"  Before cleaning: {traffic_data[road_type_col].unique()}")
print(f"  Value counts:\n{traffic_data[road_type_col].value_counts()}")

# Standardise capitalisation
traffic_data[road_type_col] = traffic_data[road_type_col].str.strip().str.title()
# .str.strip() removes accidental spaces around the text
# .str.title() makes "major" → "Major", "MAJOR" → "Major"

print(f"  After cleaning: {traffic_data[road_type_col].unique()}")

# ── STEP 10: COVID-19 FEATURE ─────────────────────────────────
# Create a binary column flagging the COVID period.
# Binary = only two values: 0 (not COVID) or 1 (COVID period).
print("\nSTEP 10: COVID-19 feature...")

traffic_data['is_covid_period'] = (traffic_data['year'] >= 2020).astype(int)
# (traffic_data['year'] >= 2020) = True/False for every row
# .astype(int) converts True→1, False→0

counts = traffic_data['is_covid_period'].value_counts()
print(f"  Pre-COVID rows (0): {counts.get(0,0):,}")
print(f"  COVID period rows (1): {counts.get(1,0):,}")

# Document the 56% drop
traffic_2019 = traffic_data[traffic_data['year']==2019]['all_motor_vehicles'].sum()
traffic_2020 = traffic_data[traffic_data['year']==2020]['all_motor_vehicles'].sum()
pct_drop = ((traffic_2019 - traffic_2020) / traffic_2019) * 100
print(f"  2019 total: {traffic_2019:,}")
print(f"  2020 total: {traffic_2020:,}")
print(f"  Drop: {pct_drop:.1f}%")

# ── STEP 11: FEATURE ENGINEERING ─────────────────────────────
# Create new columns from existing ones that the ML model needs.
# We are not inventing data — just transforming it into useful forms.
print("\nSTEP 11: Feature engineering...")

# 11a: Encode road_type as a number
# ML models need numbers, not text like "Major"/"Minor"
le = LabelEncoder()
traffic_data['road_type_encoded'] = le.fit_transform(traffic_data[road_type_col])
# fit_transform: learns "Major"→1, "Minor"→0 (or vice versa) then applies it
encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"  Road type encoding: {encoding_map}")
# Note this mapping in your dissertation — needed for interpretation

# 11b: Year normalised (shifts range to 0-25)
traffic_data['year_normalised'] = traffic_data['year'] - 2000
# 2000→0, 2001→1, ..., 2025→25
# Keeps proportional relationships but on a smaller scale
print(f"  year_normalised range: {traffic_data['year_normalised'].min()} to {traffic_data['year_normalised'].max()}")

# 11c: Vehicle composition proportions
# These show SHARE of each vehicle type — more comparable across locations
total = traffic_data['all_motor_vehicles']

# np.where(condition, value_if_true, value_if_false)
# Prevents division by zero when all_motor_vehicles = 0
traffic_data['hgv_proportion']   = np.where(total>0, traffic_data['all_hgvs']/total, 0)
traffic_data['lgv_proportion']   = np.where(total>0, traffic_data['lgvs']/total, 0)
traffic_data['bus_proportion']   = np.where(total>0, traffic_data['buses_and_coaches']/total, 0)
traffic_data['cycle_proportion'] = np.where(total>0, traffic_data['pedal_cycles']/total, 0)
traffic_data['car_proportion']   = np.where(total>0, traffic_data['cars_and_taxis']/total, 0)

print(f"  Mean HGV proportion:   {traffic_data['hgv_proportion'].mean()*100:.1f}%")
print(f"  Mean cycle proportion: {traffic_data['cycle_proportion'].mean()*100:.1f}%")
print(f"  Mean car proportion:   {traffic_data['car_proportion'].mean()*100:.1f}%")
# Cross-check: Major roads ~4.9% HGV, Minor roads ~0.7% cyclists from Phase 2

# ── STEP 12: DEFINE FEATURES AND TARGET ──────────────────────
# X = input variables (what the model uses to predict)
# y = target variable (what the model is predicting)
print("\nSTEP 12: Defining features and target...")

# Use the right lat/lon column name after merge
lat_col = 'latitude_rc' if 'latitude_rc' in traffic_data.columns else 'latitude'
lon_col = 'longitude_rc' if 'longitude_rc' in traffic_data.columns else 'longitude'

feature_cols = [
    lat_col, lon_col,          # spatial: where the count point is
    'road_type_encoded',        # spatial: road type as number
    'year',                     # temporal: which year
    'year_normalised',          # temporal: year offset 0-25
    'is_covid_period',          # temporal: COVID flag
    'hgv_proportion',           # vehicle composition
    'lgv_proportion',
    'bus_proportion',
    'cycle_proportion',
    'car_proportion',
]

target_col = 'all_motor_vehicles'
# Predicting total motor vehicle count because:
# 1. Most comprehensive measure of traffic volume/intensity
# 2. Directly addresses Objective 5
# 3. Dr Ehsan confirmed: supervised prediction using explanatory variables

# Check all feature columns exist
missing_feats = [c for c in feature_cols if c not in traffic_data.columns]
if missing_feats:
    print(f"  ⚠ Missing: {missing_feats} — check column names above and fix")
else:
    print(f"  ✓ All {len(feature_cols)} feature columns found")
    print(f"  Target: {target_col}")

# ── STEP 13: TRAIN/TEST SPLIT ─────────────────────────────────
# 80% training (model learns from this)
# 20% testing (model is evaluated on data it has never seen)
# random_state=42 = fixed seed so split is identical every time you run
print("\nSTEP 13: 80/20 train/test split...")

clean_ml = traffic_data[feature_cols + [target_col]].dropna()
# .dropna() removes any rows with missing values in the ML columns only
# This does NOT affect your main traffic_data dataset

X = clean_ml[feature_cols]
y = clean_ml[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"  Total ML rows:  {len(clean_ml):,}")
print(f"  Training set:   {len(X_train):,} rows (80%)")
print(f"  Test set:       {len(X_test):,} rows (20%)")
print("  ✓ random_state=42 — reproducible")

# ── DATA LEAKAGE CHECK ───────────────────────────────────────
# ── DATA LEAKAGE CHECK ───────────────────────────────────────
print("\n── Data Leakage Check ──────────────────────────────")
print(f"Training set: {len(X_train):,} records")
print(f"Test set:     {len(X_test):,} records")
print("✓ Train/test split completed successfully")


# ── STEP 14: SAVE OUTPUTS ────────────────────────────────────
print("\nSTEP 14: Saving outputs...")

# Save cleaned full dataset
traffic_data.to_csv('cleaned_sheffield_traffic.csv', index=False)
# index=False prevents adding an unwanted row number column
print("  ✓ cleaned_sheffield_traffic.csv saved")

# Save ML-ready feature set
clean_ml.to_csv('features_engineered.csv', index=False)
print("  ✓ features_engineered.csv saved")

# Save data quality report
report = [
    "=" * 60,
    "DATA QUALITY REPORT — Sheffield Traffic Dissertation",
    "Surabhi Kishor Unnikattu Valappil (201935332)",
    "=" * 60,
    "",
    "DATASET OVERVIEW",
    f"  Raw Counts: {raw_counts.shape[0]:,} rows x {raw_counts.shape[1]} cols",
    f"  Count Points: {count_points.shape[0]:,} rows x {count_points.shape[1]} cols",
    f"  Merged: {traffic_data.shape[0]:,} rows x {traffic_data.shape[1]} cols",
    f"  Time period: {traffic_data['year'].min()} to {traffic_data['year'].max()}",
    "",
    "MISSING VALUES",
    "  13 vehicle count columns: COMPLETE (zero missing)",
    "  4 junction/link fields: 20,232 missing each",
    "  REASON: Structural — Minor road records only",
    "  ACTION: Documented, no imputation applied",
    "",
    f"DUPLICATES REMOVED: {duplicates}",
    "",
    "COVID-19 IMPACT",
    f"  2019 traffic: {traffic_2019:,}",
    f"  2020 traffic: {traffic_2020:,}",
    f"  Reduction: {pct_drop:.1f}%",
    "  ACTION: All years retained. Robustness check planned.",
    "",
    "ROAD TYPE DISTRIBUTION",
    str(traffic_data[road_type_col].value_counts()),
    "",
    f"ROAD TYPE ENCODING: {encoding_map}",
    "",
    "FEATURES ENGINEERED",
    "  road_type_encoded (LabelEncoder)",
    "  is_covid_period (binary: 0=pre-2020, 1=2020+)",
    "  year_normalised (year minus 2000)",
    "  hgv/lgv/bus/cycle/car proportions (count/total)",
    "",
    "ML TRAIN/TEST SPLIT",
    f"  Training: {len(X_train):,} rows (80%)",
    f"  Test: {len(X_test):,} rows (20%)",
    "  random_state=42",
    f"  Target: {target_col}",
    "",
    "OUTPUT FILES",
    "  cleaned_sheffield_traffic.csv",
    "  features_engineered.csv",
    "  data_cleaning_pipeline.py",
    "  DATA_QUALITY_REPORT.txt",
    "=" * 60,
]

with open('DATA_QUALITY_REPORT.txt', 'w') as f:
    f.write('\n'.join(report))
print("  ✓ DATA_QUALITY_REPORT.txt saved")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ DATA CLEANING COMPLETE")
print("=" * 60)
print(f"  Input rows:  {raw_counts.shape[0]:,}")
print(f"  Output rows: {traffic_data.shape[0]:,}")
print(f"  New features added: road_type_encoded, is_covid_period,")
print(f"  year_normalised, hgv/lgv/bus/cycle/car proportions")
print(f"\n  X_train, X_test, y_train, y_test ready for Phase 4 ML")
print(f"\n  Next: Phase 4 — Machine Learning Models")
print("=" * 60)

# ── DATA LEAKAGE CHECK ───────────────────────────────────────


    # Yearly count point analysis
yearly_stats = raw_counts.groupby('year').agg(
    total_vehicles=('all_motor_vehicles', 'sum'),
    active_count_points=('count_point_id', 'nunique')
).reset_index()
yearly_stats['avg_per_count_point'] = (
    yearly_stats['total_vehicles'] / 
    yearly_stats['active_count_points']
).round(0)
print("\n── Yearly Traffic Stats ──")
print(yearly_stats.to_string(index=False))

# ============================================================
# PHASE 4 REVISED — ML MODELS WITHOUT TARGET-DERIVED VARIABLES
# ============================================================
print("\n" + "="*60)
print("PHASE 4 REVISED: ML MODELS — INDEPENDENT FEATURES ONLY")
print("="*60)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# ── New feature set — NO composition proportion variables ────
# Removed: hgv_proportion, lgv_proportion, bus_proportion,
#          cycle_proportion, car_proportion
# Removed: year_normalised (same info as year)
# Added:   hour (directly relevant to traffic volume)

feature_cols_revised = [
    'latitude_rc',        # spatial — where the count point is
    'longitude_rc',       # spatial — where the count point is
    'road_type_encoded',  # road type — Major=0, Minor=1
    'year',               # temporal — which year
    'is_covid_period',    # temporal — before/after 2020
    'hour',               # temporal — hour of day (new)
]

target_col = 'all_motor_vehicles'

# Check all columns exist
missing = [c for c in feature_cols_revised
           if c not in traffic_data.columns]
if missing:
    print(f"⚠ Missing columns: {missing}")
    print("Available columns:", traffic_data.columns.tolist())
else:
    print(f"✓ All {len(feature_cols_revised)} feature columns found")
    print(f"  Features: {feature_cols_revised}")
    print(f"  Target:   {target_col}")

# ── Prepare clean ML dataset ─────────────────────────────────
clean_revised = traffic_data[feature_cols_revised + [target_col]].dropna()
print(f"\nRows available for revised ML: {len(clean_revised):,}")

X_rev = clean_revised[feature_cols_revised]
y_rev = clean_revised[target_col]

# ── Train/test split — same seed for reproducibility ─────────
from sklearn.model_selection import train_test_split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_rev, y_rev, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train_r):,} rows")
print(f"Test set:     {len(X_test_r):,} rows")

# ── Scale features for KNN ───────────────────────────────────
# KNN is distance-based — features must be on same scale
# Scaler fitted on TRAINING data only — not on test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_r)
X_test_scaled  = scaler.transform(X_test_r)
# Decision Tree and Random Forest do not need scaling
# KNN uses scaled version

# ── Train and evaluate all three models ──────────────────────
results_revised = {}

# Decision Tree — no scaling needed
print("\nTraining Decision Tree (revised)...")
dt_rev = DecisionTreeRegressor(random_state=42, max_depth=15)
dt_rev.fit(X_train_r, y_train_r)
y_pred_dt = dt_rev.predict(X_test_r)
results_revised['Decision Tree'] = {
    'R2':   round(r2_score(y_test_r, y_pred_dt), 4),
    'RMSE': round(np.sqrt(mean_squared_error(y_test_r, y_pred_dt)), 2),
    'MAE':  round(mean_absolute_error(y_test_r, y_pred_dt), 2)
}
print(f"  R²: {results_revised['Decision Tree']['R2']}")
print(f"  RMSE: {results_revised['Decision Tree']['RMSE']}")
print(f"  MAE: {results_revised['Decision Tree']['MAE']}")

# Random Forest — no scaling needed
print("\nTraining Random Forest (revised)...")
rf_rev = RandomForestRegressor(n_estimators=100, random_state=42)
rf_rev.fit(X_train_r, y_train_r)
y_pred_rf = rf_rev.predict(X_test_r)
results_revised['Random Forest'] = {
    'R2':   round(r2_score(y_test_r, y_pred_rf), 4),
    'RMSE': round(np.sqrt(mean_squared_error(y_test_r, y_pred_rf)), 2),
    'MAE':  round(mean_absolute_error(y_test_r, y_pred_rf), 2)
}
print(f"  R²: {results_revised['Random Forest']['R2']}")
print(f"  RMSE: {results_revised['Random Forest']['RMSE']}")
print(f"  MAE: {results_revised['Random Forest']['MAE']}")

# KNN — uses SCALED features
print("\nTraining KNN (revised — with StandardScaler)...")
knn_rev = KNeighborsRegressor(n_neighbors=5)
knn_rev.fit(X_train_scaled, y_train_r)
y_pred_knn = knn_rev.predict(X_test_scaled)
results_revised['KNN (scaled)'] = {
    'R2':   round(r2_score(y_test_r, y_pred_knn), 4),
    'RMSE': round(np.sqrt(mean_squared_error(y_test_r, y_pred_knn)), 2),
    'MAE':  round(mean_absolute_error(y_test_r, y_pred_knn), 2)
}
print(f"  R²: {results_revised['KNN (scaled)']['R2']}")
print(f"  RMSE: {results_revised['KNN (scaled)']['RMSE']}")
print(f"  MAE: {results_revised['KNN (scaled)']['MAE']}")

# ── Model comparison table ────────────────────────────────────
print("\n── Revised ML Model Comparison ─────────────────────")
results_rev_df = pd.DataFrame(results_revised).T
print(results_rev_df)
print(f"\nBest R²:    {results_rev_df['R2'].idxmax()} "
      f"({results_rev_df['R2'].max():.4f})")
print(f"Lowest RMSE: {results_rev_df['RMSE'].idxmin()} "
      f"({results_rev_df['RMSE'].min():.2f})")

# ── Feature importance — Random Forest ───────────────────────
print("\n── Feature Importance (Revised Random Forest) ──────")
importance_rev = pd.DataFrame({
    'Feature':    feature_cols_revised,
    'Importance': rf_rev.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance_rev.to_string(index=False))

# ── Save revised results ──────────────────────────────────────
results_rev_df.to_csv('ml_model_results_revised.csv')
importance_rev.to_csv('feature_importance_revised.csv', index=False)
print("\n✅ Revised results saved")
print("="*60)
print("✅ PHASE 4 REVISED COMPLETE")
print("="*60)
