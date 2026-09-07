import pandas as pd

df = pd.read_csv('cleaned_sheffield_traffic.csv', low_memory=False)
road_col = 'road_type_rc' if 'road_type_rc' in df.columns else 'road_type'

major = df[df[road_col]=='Major']
minor = df[df[road_col]=='Minor']

print("Method 2 — Sum of vehicle type / Sum of all_motor_vehicles:")
print("\nMAJOR ROADS:")
for col, name in [('cars_and_taxis','Cars'),('all_hgvs','HGVs'),
                  ('lgvs','LGVs'),('buses_and_coaches','Buses'),
                  ('pedal_cycles','Cyclists')]:
    pct = major[col].sum() / major['all_motor_vehicles'].sum() * 100
    print(f"  {name}: {pct:.2f}%")

print("\nMINOR ROADS:")
for col, name in [('cars_and_taxis','Cars'),('all_hgvs','HGVs'),
                  ('lgvs','LGVs'),('buses_and_coaches','Buses'),
                  ('pedal_cycles','Cyclists')]:
    pct = minor[col].sum() / minor['all_motor_vehicles'].sum() * 100
    print(f"  {name}: {pct:.2f}%")

print("\nOVERALL:")
for col, name in [('cars_and_taxis','Cars'),('all_hgvs','HGVs'),
                  ('lgvs','LGVs'),('buses_and_coaches','Buses'),
                  ('pedal_cycles','Cyclists')]:
    pct = df[col].sum() / df['all_motor_vehicles'].sum() * 100
    print(f"  {name}: {pct:.2f}%")