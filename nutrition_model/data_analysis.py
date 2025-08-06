import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('combined_growing.csv')

# Get total number of data points
total_points = len(df)
print(f"\nTotal number of data points: {total_points}")

# Get count of each type
type_counts = df['type'].value_counts()
print("\nCount of each type:")
print(type_counts)

# Get statistics for target_weight by type
print("\nTarget weight statistics by type:")
for type_val in df['type'].unique():
    type_data = df[df['type'] == type_val]
    print(f"\nType {type_val}:")
    print(f"Count: {len(type_data)}")
    print(f"Unique target weights: {sorted(type_data['target_weight'].unique())}")
    print(f"Mean target weight: {type_data['target_weight'].mean():.2f}")
    print(f"Min target weight: {type_data['target_weight'].min()}")
    print(f"Max target weight: {type_data['target_weight'].max()}")
