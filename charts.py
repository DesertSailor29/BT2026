import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load data
df = pd.read_csv('liver_tumor_metrics.csv')
metrics = ['dice_liver', 'dice_tumor', 'dice_avg', 'hd_full_liver', 'hd_full_tumor', 
           'hd95_liver', 'hd95_tumor', 'hd95_avg']
os.makedirs('charts', exist_ok=True)

def plot_metrics(df, title_suffix, folder):
    """Create 4x2 subplot grid of boxplots"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        data = df[[ 'model', metric ]].dropna()
        if data.empty:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(metric)
            continue
            
        sns.boxplot(data=data, x='model', y=metric, hue='model', ax=axes[i], 
                   palette='Set1', legend=False, dodge=False)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(alpha=0.3)
    
    plt.suptitle(f"Liver Tumor Metrics {title_suffix}", fontsize=16, y=0.98)
    plt.tight_layout()
    filename = f"{folder}/metrics{title_suffix.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {filename}")

# 1. ALL DATA
print("📊 All data charts...")
plot_metrics(df, "(All Data)", "charts")

# 2. NO OUTLIERS - Simple robust method
print("📊 Removing outliers...")
df_clean = df.copy()

for metric in metrics:
    # Per-model IQR
    model_groups = df.groupby('model')[metric]
    if len(model_groups) == 0:
        continue
        
    Q1 = model_groups.quantile(0.25)
    Q3 = model_groups.quantile(0.75)
    lower_bound = Q1 - 1.5 * (Q3 - Q1)
    upper_bound = Q3 + 1.5 * (Q3 - Q1)
    
    # Apply bounds row-by-row using model mapping
    lower_map = df['model'].map(lower_bound)
    upper_map = df['model'].map(upper_bound)
    
    # Mark outliers as NaN (vectorized, handles NaNs)
    outlier_mask = ~((df[metric] >= lower_map) & (df[metric] <= upper_map))
    df_clean.loc[outlier_mask & df_clean[metric].notna(), metric] = np.nan

df_no_outliers = df_clean.dropna()

# No outliers charts
os.makedirs('charts_no_outliers', exist_ok=True)
plot_metrics(df_no_outliers, "(No Outliers)", "charts_no_outliers")

print("\n🎉 COMPLETE! Check:")
print("📁 charts/metrics(All_Data).png")
print("📁 charts_no_outliers/metrics(No_Outliers).png")
print("   → 16x20in, 300 DPI, 7 models x 8 metrics each")
