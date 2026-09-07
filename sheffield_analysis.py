# Sheffield Traffic Dissertation — Exploratory Analysis
# Surabhi | MSc Business Analytics & Big Data | University of Liverpool

import pandas as pd
import matplotlib.pyplot as plt

# ── Load the two datasets ──────────────────────────────────────────────────────
raw_counts   = pd.read_csv('dft_rawcount_local_authority_id_159.csv')
count_points = pd.read_csv('dft_countpoints_local_authority_id_159.csv')

# ── Sanity checks ──────────────────────────────────────────────────────────────
print("Raw Counts shape:", raw_counts.shape)
print("Count Points shape:", count_points.shape)

print("\nRaw Counts columns:\n", raw_counts.columns.tolist())
print("\nCount Points columns:\n", count_points.columns.tolist())

# ── Understand the structure ───────────────────────────────────────────────────
print("\n--- RAW COUNTS INFO ---")
raw_counts.info()

print("\nMissing values in Raw Counts:\n", raw_counts.isnull().sum())

# ── Descriptive statistics ─────────────────────────────────────────────────────
print("\n--- DESCRIPTIVE STATS ---")
print(raw_counts.describe())

# ── Run these AFTER checking column names above ────────────────────────────────
# NOTE: column names printed above may differ — update 'year', 'all_motor_vehicles'
#       etc. below if needed once you see the actual names

print("\nUnique years:", raw_counts['year'].unique())
print("\nRoad types:\n", raw_counts['road_type'].value_counts())

# ── Annual trend ───────────────────────────────────────────────────────────────
yearly_trend = raw_counts.groupby('year')['all_motor_vehicles'].sum()
print("\nYearly vehicle totals:\n", yearly_trend)

# ── Vehicle breakdown ──────────────────────────────────────────────────────────
vehicle_breakdown = raw_counts[['cars_and_taxis', 'buses_and_coaches',
                                 'lgvs', 'pedal_cycles']].sum()

# ── Merge datasets ─────────────────────────────────────────────────────────────
merged_df = pd.merge(raw_counts, count_points, on='count_point_id', how='left')
print("\nMerged dataset shape:", merged_df.shape)

# ── Plot 1: Annual trend ───────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(yearly_trend.index, yearly_trend.values,
         marker='o', color='steelblue', linewidth=2)
plt.title('Annual Motor Vehicle Counts in Sheffield', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Total Vehicle Count')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot1_annual_trend.png', dpi=150)
plt.show()

# ── Plot 2: Vehicle types ──────────────────────────────────────────────────────
vehicle_breakdown.plot(kind='bar', color='teal', edgecolor='white', figsize=(10, 5))
plt.title('Total Counts by Vehicle Type — Sheffield')
plt.ylabel('Total Count')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('plot2_vehicle_types.png', dpi=150)
plt.show()

# ── Plot 3: Distribution histogram ────────────────────────────────────────────
plt.figure(figsize=(10, 5))
raw_counts['all_motor_vehicles'].hist(bins=50, color='coral', edgecolor='white')
plt.title('Distribution of Annual Motor Vehicle Counts per Count Point')
plt.xlabel('Vehicle Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plot3_distribution.png', dpi=150)
plt.show()

print("\n✅ Done — check your folder for the 3 saved plots.")


# ============================================================
# PHASE 2 — DEEPER EDA
# ============================================================

# ── Block 1: Traffic summary by road type ───────────────────
vehicle_cols = ['cars_and_taxis', 'all_hgvs', 'lgvs',
                'buses_and_coaches', 'pedal_cycles']

road_type_summary = raw_counts.groupby('road_type')[vehicle_cols].sum()
print("\n── Road Type Summary (raw totals) ──")
print(road_type_summary)

# ── Block 2: Vehicle composition as percentages ──────────────
road_type_pct = road_type_summary.div(
    road_type_summary.sum(axis=1), axis=0) * 100

print("\n── Vehicle Composition by Road Type (%) ──")
print(road_type_pct.round(1))

road_type_pct.plot(kind='bar', stacked=True,
                   figsize=(8, 6), colormap='Set2')
plt.title('Vehicle Composition by Road Type — Sheffield (%)')
plt.ylabel('Percentage of Total Traffic')
plt.xticks(rotation=0)
plt.legend(title='Vehicle Type',
           bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plot4_composition_roadtype.png', dpi=150)
plt.show()
print("plot4 saved")

# ── Block 3: Yearly trend by road type ──────────────────────
yearly_roadtype = (raw_counts
    .groupby(['year', 'road_type'])['all_motor_vehicles']
    .sum()
    .unstack())

print("\n── Yearly Traffic by Road Type ──")
print(yearly_roadtype)

yearly_roadtype.plot(kind='line', marker='o', figsize=(12, 5))
plt.title('Annual Motor Vehicle Counts by Road Type — Sheffield')
plt.xlabel('Year')
plt.ylabel('Total Vehicle Count')
plt.legend(title='Road Type')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot5_yearly_by_roadtype.png', dpi=150)
plt.show()
print("plot5 saved")

# ── Block 4: COVID robustness check ─────────────────────────
yearly_all = raw_counts.groupby('year')['all_motor_vehicles'].sum()

no_covid = raw_counts[~raw_counts['year'].isin([2020, 2021])]
yearly_no_covid = no_covid.groupby('year')['all_motor_vehicles'].sum()

plt.figure(figsize=(12, 5))
plt.plot(yearly_all.index, yearly_all.values,
         color='steelblue', marker='o', linewidth=2,
         label='All years (inc. COVID)')
plt.plot(yearly_no_covid.index, yearly_no_covid.values,
         color='coral', marker='o', linewidth=2,
         linestyle='--', label='Excluding 2020-21')
plt.title('Annual Traffic Trend With and Without COVID Years — Sheffield')
plt.xlabel('Year')
plt.ylabel('Total Vehicle Count')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot6_covid_comparison.png', dpi=150)
plt.show()
print("plot6 saved")

# ── Block 5: Hotspot identification ─────────────────────────
location_totals = merged_df.groupby('count_point_id').agg(
    total_traffic   = ('all_motor_vehicles', 'sum'),
    latitude        = ('latitude_x',         'first'),
    longitude       = ('longitude_x',        'first'),
    road_name       = ('road_name_x',        'first'),
    road_type       = ('road_type_x',        'first')
).reset_index()

top10 = location_totals.sort_values(
    'total_traffic', ascending=False).head(10)

print("\n── Top 10 Highest Traffic Locations in Sheffield ──")
print(top10[['road_name', 'road_type', 'total_traffic']])

# ── Block 6: Bubble map ──────────────────────────────────────
plt.figure(figsize=(9, 8))
plt.scatter(
    location_totals['longitude'],
    location_totals['latitude'],
    s=location_totals['total_traffic'] / 500,
    alpha=0.5,
    color='darkorange',
    edgecolors='black',
    linewidth=0.4
)
plt.title(
    'Traffic Intensity by Location — Sheffield\n'
    '(bubble size = total vehicle count across all years)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('plot7_hotspot_map.png', dpi=150)
plt.show()
print("plot7 saved")

print("\n✅ Phase 2 complete — all plots saved")

import pandas as pd

feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

pd.set_option('display.float_format', '{:.6f}'.format)
print(feature_importance_df)
