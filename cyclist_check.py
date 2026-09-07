# Save as cyclist_check.py and run it
import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_sheffield_traffic.csv')
road_col = 'road_type_rc' if 'road_type_rc' in df.columns else 'road_type'

major = df[df[road_col]=='Major']
minor = df[df[road_col]=='Minor']

print("Method 1 — Mean of record-level proportions:")
print(f"  Major: {major['cycle_proportion'].mean()*100:.2f}%")
print(f"  Minor: {minor['cycle_proportion'].mean()*100:.2f}%")
print(f"  Overall: {df['cycle_proportion'].mean()*100:.2f}%")

print("\nMethod 2 — Sum cyclists / Sum total vehicles:")
print(f"  Major: {major['pedal_cycles'].sum()/major['all_motor_vehicles'].sum()*100:.2f}%")
print(f"  Minor: {minor['pedal_cycles'].sum()/minor['all_motor_vehicles'].sum()*100:.2f}%")
print(f"  Overall: {df['pedal_cycles'].sum()/df['all_motor_vehicles'].sum()*100:.2f}%")

print("\nChecking if pedal_cycles included in all_motor_vehicles:")
sample = df[['pedal_cycles','cars_and_taxis','all_hgvs','lgvs',
             'buses_and_coaches','all_motor_vehicles']].head(3)
print(sample)
check = df['pedal_cycles'] + df['cars_and_taxis'] + df['all_hgvs'] + \
        df['lgvs'] + df['buses_and_coaches'] + \
        df['two_wheeled_motor_vehicles']
matches = (check == df['all_motor_vehicles']).sum()
print(f"\nRows where sum of types = all_motor_vehicles: {matches:,} of {len(df):,}")