import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('cleaned_sheffield_traffic.csv')
road_col = 'road_type_rc' if 'road_type_rc' in df.columns else 'road_type'
lat_col = 'latitude_rc' if 'latitude_rc' in df.columns else 'latitude'
lon_col = 'longitude_rc' if 'longitude_rc' in df.columns else 'longitude'

locations = df.groupby(['count_point_id', lat_col, lon_col, road_col]).size().reset_index()
major = locations[locations[road_col]=='Major']
minor = locations[locations[road_col]=='Minor']

fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(minor[lon_col], minor[lat_col],
           c='steelblue', s=35, alpha=0.6,
           label=f'Minor roads (n={len(minor)})', zorder=2)
ax.scatter(major[lon_col], major[lat_col],
           c='darkorange', s=55, alpha=0.8,
           label=f'Major roads (n={len(major)})', zorder=3)
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title('Figure X: DfT Count Point Locations — Sheffield\n'
             '(286 count points by road classification)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot8_study_area_map.png', dpi=150)
plt.show()
print("✅ Study area map saved as plot8_study_area_map.png")