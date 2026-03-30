import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('liver_tumor_metrics.csv')

print(f"Loaded {len(df)} rows from liver_tumor_metrics.csv")
print(f"Found {df['model'].nunique()} unique models")

# Clean metrics: replace extreme outliers with NaN
outlier_threshold = 1e6
df['dice_liver_clean'] = df['dice_liver'].where(df['dice_liver'] < outlier_threshold)
df['dice_tumor_clean'] = df['dice_tumor'].where(df['dice_tumor'] < outlier_threshold)
df['dice_avg_clean'] = df['dice_avg'].where(df['dice_avg'] < outlier_threshold)

# Group by model - compute stats for each metric
liver_stats = df.groupby('model')['dice_liver_clean'].agg(['count', 'mean', 'median', 'std']).add_prefix('liver_')
tumor_stats = df.groupby('model')['dice_tumor_clean'].agg(['count', 'mean', 'median', 'std']).add_prefix('tumor_')
avg_stats = df.groupby('model')['dice_avg_clean'].agg(['count', 'mean', 'median', 'std']).add_prefix('avg_')

# Combine all stats
stats = pd.concat([liver_stats, tumor_stats, avg_stats], axis=1).round(4)

print("\n📊 Dice Statistics by Model (Liver, Tumor, Average):")
print(stats)

# Save full results
stats.to_csv('dice_stats_detailed.csv')
print(f"\n✓ Full results saved to dice_stats_detailed.csv")

# Rankings
print("\n🏆 Best Liver Segmentation (mean):")
print(stats.nlargest(3, 'liver_mean')[['liver_mean', 'liver_std']])

print("\n🎯 Best Tumor Segmentation:")
print(stats.nlargest(3, 'tumor_mean')[['tumor_mean', 'tumor_std']])

print("\n📈 Overall Best (avg_mean):")
print(stats.nlargest(3, 'avg_mean')[['avg_mean', 'avg_std']])
