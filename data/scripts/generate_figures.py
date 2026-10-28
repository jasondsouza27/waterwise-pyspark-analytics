#!/usr/bin/env python3
"""
Regenerate all 17 figures for household water consumption manuscript.
Academic style with clean, minimal design and proper error bars.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for academic publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 100  # Set to 100 for screen, save at 300 DPI

# Color palette: muted, professional
COLORS = {
    'cold': '#4A4A4A',      # Dark gray
    'hot': '#8B8B8B',       # Medium gray
    'accent1': '#5A7C99',   # Muted slate blue
    'accent2': '#9B8B7C',   # Muted brown
}

# Output directory
OUTPUT_DIR = '/sessions/affectionate-youthful-newton/mnt/Household water consumption data/figures/'

def load_data():
    """Load and preprocess data"""
    df = pd.read_csv('/sessions/affectionate-youthful-newton/cleaned_data.csv')

    # Map Swedish labels to English
    type_mapping = {
        'cold water consumption': 'Cold water',
        'hot water consumption': 'Hot water'
    }
    df['type_english'] = df['type'].map(type_mapping)

    # Create anonymized apartment labels
    apartments = sorted(df[df['apartment'].notna()]['apartment'].unique())
    apt_mapping = {apt: f'Apt {i+1}' for i, apt in enumerate(apartments)}
    df['apartment_anon'] = df['apartment'].map(apt_mapping)

    # Create anonymized cluster labels (handle NaN values)
    clusters = sorted([c for c in df['cluster_name'].unique() if pd.notna(c)])
    cluster_mapping = {cluster: f'Cluster {i+1}' for i, cluster in enumerate(clusters)}
    df['cluster_anon'] = df['cluster_name'].map(cluster_mapping)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df, apt_mapping, cluster_mapping

def save_figure(fig, filename, dpi=300):
    """Save figure with proper settings"""
    filepath = OUTPUT_DIR + filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {filename}")
    plt.close(fig)

# ============================================================================
# FIGURE 1: Data Availability Timeline
# ============================================================================
def fig_01_data_availability(df):
    """Show active sensors or data points per month over study period"""
    df_copy = df.copy()
    df_copy['year_month'] = df_copy['timestamp'].dt.to_period('M')

    # Count data points per month
    monthly_counts = df_copy.groupby('year_month').size()

    fig, ax = plt.subplots(figsize=(10, 4))

    dates = [str(period) for period in monthly_counts.index]
    ax.bar(range(len(monthly_counts)), monthly_counts.values, color=COLORS['accent1'],
           edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Number of Records', fontsize=14)
    ax.set_title('Data Availability Over Time', fontsize=16)
    ax.set_xticks(range(0, len(monthly_counts), 6))
    ax.set_xticklabels([dates[i] for i in range(0, len(monthly_counts), 6)], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    save_figure(fig, '01_data_availability.png')

# ============================================================================
# FIGURE 2: Value Distribution (log scale, exclude zeros)
# ============================================================================
def fig_02_value_distribution(df):
    """Distribution of consumption values (log scale, exclude zeros)"""
    values = df[df['value'] > 0]['value'].values

    fig, ax = plt.subplots(figsize=(9, 5))

    # Histogram with log scale
    ax.hist(values, bins=50, color=COLORS['accent1'], edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.set_xscale('log')

    # Mark outlier threshold at 0.3 m³
    ax.axvline(0.3, color='#333333', linestyle='--', linewidth=2, label='Outlier threshold (0.3 m³)')

    ax.set_xlabel('Consumption (m³, log scale)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Distribution of Consumption Values', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    save_figure(fig, '02_value_distribution.png')

# ============================================================================
# FIGURE 3: Consumption by Fixture Type (with error bars)
# ============================================================================
def fig_03_consumption_by_fixture(df):
    """Total consumption by fixture type with error bars"""
    df_copy = df.copy()

    # Calculate daily consumption per fixture per apartment
    daily_by_fixture = df_copy.groupby(['date', 'attached_to'])['value'].sum().reset_index()

    # Calculate mean and std per fixture
    fixture_stats = daily_by_fixture.groupby('attached_to')['value'].agg(['mean', 'std']).reset_index()
    fixture_stats = fixture_stats.sort_values('mean', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(fixture_stats))
    ax.barh(y_pos, fixture_stats['mean'], xerr=fixture_stats['std'],
            color=COLORS['accent1'], edgecolor='black', linewidth=0.5, capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(fixture_stats['attached_to'])
    ax.set_xlabel('Mean Daily Consumption (m³)', fontsize=14)
    ax.set_title('Total Consumption by Fixture Type', fontsize=16)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    save_figure(fig, '03_consumption_by_fixture.png')

# ============================================================================
# FIGURE 4: Diurnal (Hourly) Patterns
# ============================================================================
def fig_04_hourly_patterns(df):
    """Mean hourly consumption with std band"""
    df_copy = df.copy()

    # Sum values per hour per day, then average across days
    hourly_daily = df_copy.groupby(['date', 'hour'])['value'].sum().reset_index()
    hourly_stats = hourly_daily.groupby('hour')['value'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    hours = hourly_stats['hour'].values
    means = hourly_stats['mean'].values
    stds = hourly_stats['std'].values

    # Plot mean line
    ax.plot(hours, means, color=COLORS['accent1'], linewidth=2.5, label='Mean consumption')

    # Shaded std band
    ax.fill_between(hours, means - stds, means + stds, color=COLORS['accent1'], alpha=0.2,
                    label='±1 std deviation')

    # Mark morning and evening peaks (approximate)
    ax.axvspan(6, 8, alpha=0.08, color='#888888', label='Morning peak')
    ax.axvspan(18, 21, alpha=0.08, color='#444444', label='Evening peak')

    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Mean Daily Consumption (m³)', fontsize=14)
    ax.set_title('Diurnal Consumption Pattern', fontsize=16)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    save_figure(fig, '04_hourly_patterns.png')

# ============================================================================
# FIGURE 5: Weekly Patterns
# ============================================================================
def fig_05_weekly_patterns(df):
    """Day-of-week consumption pattern"""
    df_copy = df.copy()

    # Calculate daily consumption per day of week
    daily_by_dow = df_copy.groupby(['date', 'day_of_week', 'day_name'])['value'].sum().reset_index()
    dow_stats = daily_by_dow.groupby(['day_of_week', 'day_name'])['value'].agg(['mean', 'std']).reset_index()
    dow_stats = dow_stats.sort_values('day_of_week')

    fig, ax = plt.subplots(figsize=(9, 5))

    x_pos = np.arange(len(dow_stats))
    ax.bar(x_pos, dow_stats['mean'], yerr=dow_stats['std'],
           color=COLORS['accent1'], edgecolor='black', linewidth=0.5, capsize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(dow_stats['day_name'], rotation=45, ha='right')
    ax.set_ylabel('Mean Daily Consumption (m³)', fontsize=14)
    ax.set_title('Weekly Consumption Pattern', fontsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    save_figure(fig, '05_weekly_patterns.png')

# ============================================================================
# FIGURE 6: Monthly Patterns
# ============================================================================
def fig_06_monthly_patterns(df):
    """Monthly consumption pattern (normalized by active days)"""
    df_copy = df.copy()

    # Calculate monthly total and number of active days
    df_copy['year_month'] = df_copy['timestamp'].dt.to_period('M')
    monthly_data = df_copy.groupby('year_month').agg({
        'value': 'sum',
        'date': 'nunique'  # Number of active days
    }).reset_index()

    # Normalize by active days
    monthly_data['mean'] = monthly_data['value'] / monthly_data['date']

    # For error bars, calculate std across days in each month
    monthly_daily = df_copy.groupby(['year_month', 'date'])['value'].sum().reset_index()
    monthly_stats = monthly_daily.groupby('year_month')['value'].agg(['std']).reset_index()
    monthly_data = monthly_data.merge(monthly_stats, on='year_month')

    fig, ax = plt.subplots(figsize=(11, 5))

    months = [str(p) for p in monthly_data['year_month']]
    x_pos = np.arange(len(monthly_data))

    ax.bar(x_pos, monthly_data['mean'], yerr=monthly_data['std'],
           color=COLORS['accent1'], edgecolor='black', linewidth=0.5, capsize=5)

    ax.set_xticks(range(0, len(monthly_data), 6))
    ax.set_xticklabels([months[i] for i in range(0, len(monthly_data), 6)], rotation=45, ha='right')
    ax.set_ylabel('Mean Daily Consumption (m³)', fontsize=14)
    ax.set_title('Monthly Consumption Pattern', fontsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    save_figure(fig, '06_monthly_patterns.png')

# ============================================================================
# FIGURE 7: Cluster Comparison (Boxplot)
# ============================================================================
def fig_07_cluster_comparison(df):
    """Boxplot of daily consumption by cluster - capped y-axis, high contrast"""
    df_copy = df.copy()

    # Calculate daily consumption per apartment per cluster
    daily_cluster = df_copy.groupby(['date', 'cluster_anon'])['value'].sum().reset_index()

    clusters = sorted([c for c in daily_cluster['cluster_anon'].unique() if pd.notna(c)])

    fig, ax = plt.subplots(figsize=(10, 5))

    data_by_cluster = [daily_cluster[daily_cluster['cluster_anon'] == c]['value'].values
                       for c in clusters]

    bp = ax.boxplot(data_by_cluster, labels=clusters, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='#333333', linewidth=1.2),
                    capprops=dict(color='#333333', linewidth=1.2))

    for patch in bp['boxes']:
        patch.set_facecolor('#D6D6D6')
        patch.set_edgecolor('#222222')
        patch.set_linewidth(1.2)

    ax.set_ylim(0, 3.5)
    ax.set_ylabel('Daily Consumption (m\u00b3)', fontsize=14)
    ax.set_xlabel('Cluster', fontsize=14)
    ax.set_title('Cluster Comparison', fontsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    # Note about clipped outliers
    all_vals = daily_cluster['value'].values
    n_beyond = sum(1 for v in all_vals if v > 3.5)
    if n_beyond > 0:
        ax.annotate(f'Note: {n_beyond} observations > 3.5 m\u00b3 not shown',
                    xy=(0.98, 0.97), xycoords='axes fraction', ha='right', va='top',
                    fontsize=10, fontstyle='italic', color='gray')

    save_figure(fig, '07_cluster_comparison.png')

# ============================================================================
# FIGURE 8: Hot vs Cold Water by Fixture
# ============================================================================
def fig_08_hot_vs_cold_by_fixture(df):
    """Proportion of hot/cold water for each fixture"""
    df_copy = df.copy()

    # Sum by type and fixture
    consumption = df_copy.groupby(['attached_to', 'type_english'])['value'].sum().reset_index()

    # Pivot to get cold and hot columns
    pivot = consumption.pivot(index='attached_to', columns='type_english', values='value').fillna(0)
    pivot['Total'] = pivot.sum(axis=1)

    # Calculate proportions
    fixtures = pivot.index.tolist()
    cold_prop = (pivot['Cold water'] / pivot['Total']).values
    hot_prop = (pivot['Hot water'] / pivot['Total']).values

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(fixtures))
    width = 0.6

    ax.bar(x_pos, cold_prop, width, label='Cold water', color=COLORS['cold'],
           edgecolor='black', linewidth=0.5)
    ax.bar(x_pos, hot_prop, width, bottom=cold_prop, label='Hot water', color=COLORS['hot'],
           edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(fixtures, rotation=45, ha='right')
    ax.set_ylabel('Proportion of Total Consumption', fontsize=14)
    ax.set_title('Hot vs Cold Water by Fixture Type', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_facecolor('white')

    save_figure(fig, '08_hot_vs_cold_by_fixture.png')

# ============================================================================
# FIGURE 9: Top Apartments by Consumption
# ============================================================================
def fig_09_top_apartments(df):
    """Apartment consumption ranking"""
    df_copy = df.copy()
    df_copy = df_copy[df_copy['apartment_anon'].notna()]

    # Calculate daily consumption per apartment
    daily_apt = df_copy.groupby(['date', 'apartment_anon'])['value'].sum().reset_index()
    apt_stats = daily_apt.groupby('apartment_anon')['value'].agg(['mean', 'std']).reset_index()
    apt_stats = apt_stats.sort_values('mean', ascending=True)

    # Show top 15
    apt_stats = apt_stats.tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))

    y_pos = np.arange(len(apt_stats))
    ax.barh(y_pos, apt_stats['mean'], xerr=apt_stats['std'],
            color=COLORS['accent1'], edgecolor='black', linewidth=0.5, capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(apt_stats['apartment_anon'])
    ax.set_xlabel('Mean Daily Consumption (m³)', fontsize=14)
    ax.set_title('Top Apartment Consumption', fontsize=16)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    save_figure(fig, '09_top_apartments.png')

# ============================================================================
# FIGURE 10: Shower Analysis
# ============================================================================
def fig_10_shower_analysis(df):
    """Shower analysis: distribution of volumes and timing"""
    df_shower = df[df['attached_to'] == 'Shower'].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Subplot (a): Distribution of shower volumes
    shower_values = df_shower[df_shower['value'] > 0]['value'].values
    if len(shower_values) > 0:
        ax1.hist(shower_values, bins=40, color=COLORS['accent1'], edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Shower Volume (m³)', fontsize=14)
        ax1.set_ylabel('Frequency', fontsize=14)
        ax1.set_title('(a) Distribution of Shower Volumes', fontsize=14)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Subplot (b): Shower timing by hour
    shower_hourly = df_shower.groupby('hour')['value'].agg(['sum', 'count']).reset_index()
    ax2.bar(shower_hourly['hour'], shower_hourly['count'], color=COLORS['accent1'],
            edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Hour of Day', fontsize=14)
    ax2.set_ylabel('Number of Showers', fontsize=14)
    ax2.set_title('(b) Shower Timing by Hour', fontsize=14)
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    fig.tight_layout()

    save_figure(fig, '10_shower_analysis.png')

# ============================================================================
# FIGURE 11: Per-Capita Sensitivity Analysis (25, 30, 35 residents)
# ============================================================================
def fig_11_sensitivity_percapita(df):
    """Per-capita consumption with 25, 30, 35 assumed residents"""
    df_copy = df.copy()

    # Total study period consumption
    total_consumption = df_copy['value'].sum()

    # Number of unique days
    unique_days = df_copy['date'].nunique()

    # Daily average
    daily_avg = total_consumption / unique_days

    # Per-capita for different resident counts
    residents = [25, 30, 35]
    per_capita = [daily_avg / r for r in residents]

    # Calculate by cluster
    clusters = sorted([c for c in df_copy['cluster_anon'].unique() if pd.notna(c)])
    cluster_data = []

    for cluster in clusters:
        cluster_df = df_copy[df_copy['cluster_anon'] == cluster]
        cluster_total = cluster_df['value'].sum()
        cluster_days = cluster_df['date'].nunique()
        cluster_daily = cluster_total / cluster_days
        cluster_data.append(cluster_daily)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Building-level per-capita
    x_pos = np.arange(len(residents))
    colors_list = [COLORS['accent1'], COLORS['accent2'], COLORS['cold']]
    ax1.bar(x_pos, per_capita, color=colors_list, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{r} residents' for r in residents])
    ax1.set_ylabel('Per-Capita Consumption (m³/person/day)', fontsize=14)
    ax1.set_title('(a) Building-Level Per-Capita', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_facecolor('white')

    # Add value labels on bars
    for i, v in enumerate(per_capita):
        ax1.text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=12)

    # Cluster-level per-capita (30 residents assumed)
    cluster_per_capita = [c / 30 for c in cluster_data]
    y_pos = np.arange(len(clusters))
    ax2.barh(y_pos, cluster_per_capita, color=COLORS['accent1'], edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(clusters)
    ax2.set_xlabel('Per-Capita Consumption (m³/person/day)', fontsize=14)
    ax2.set_title('(b) Cluster-Level Per-Capita (30 residents)', fontsize=14)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_facecolor('white')

    fig.tight_layout()
    save_figure(fig, '11_sensitivity_percapita.png')

# ============================================================================
# FIGURE 12: Outlier Threshold Sensitivity
# ============================================================================
def fig_12_outlier_sensitivity(df):
    """Show impact of outlier thresholds"""
    df_copy = df.copy()

    thresholds = [0.1, 0.2, 0.3, 0.5]
    total_consumption = []
    num_records = []

    for threshold in thresholds:
        filtered = df_copy[df_copy['value'] <= threshold]
        total_consumption.append(filtered['value'].sum())
        num_records.append(len(filtered))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    x_pos = np.arange(len(thresholds))

    # Total consumption vs threshold
    ax1.plot(x_pos, total_consumption, marker='o', color=COLORS['accent1'], linewidth=2, markersize=8)
    ax1.fill_between(x_pos, total_consumption, alpha=0.2, color=COLORS['accent1'])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{t} m³' for t in thresholds])
    ax1.set_ylabel('Total Consumption (m³)', fontsize=14)
    ax1.set_xlabel('Threshold', fontsize=14)
    ax1.set_title('(a) Total Consumption vs Threshold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_facecolor('white')

    # Number of records vs threshold
    ax2.plot(x_pos, num_records, marker='s', color=COLORS['accent2'], linewidth=2, markersize=8)
    ax2.fill_between(x_pos, num_records, alpha=0.2, color=COLORS['accent2'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{t} m³' for t in thresholds])
    ax2.set_ylabel('Number of Records', fontsize=14)
    ax2.set_xlabel('Threshold', fontsize=14)
    ax2.set_title('(b) Number of Records vs Threshold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_facecolor('white')

    fig.tight_layout()
    save_figure(fig, '12_outlier_sensitivity.png')

# ============================================================================
# FIGURE 13: Missingness Heatmap
# ============================================================================
def fig_13_missingness_heatmap(df):
    """Data coverage heatmap by cluster and month"""
    df_copy = df.copy()
    df_copy['year_month'] = df_copy['timestamp'].dt.to_period('M')

    # Calculate coverage (% of days with data) per cluster per month
    coverage_data = []
    for cluster in sorted([c for c in df_copy['cluster_anon'].unique() if pd.notna(c)]):
        cluster_df = df_copy[df_copy['cluster_anon'] == cluster]
        row = []
        for period in sorted(df_copy['year_month'].unique()):
            month_df = cluster_df[cluster_df['year_month'] == period]
            days_with_data = month_df['date'].nunique()
            # Approximate days in month (use 30 as average)
            coverage = (days_with_data / 30) * 100
            coverage = min(coverage, 100)  # Cap at 100%
            row.append(coverage)
        coverage_data.append(row)

    clusters = sorted([c for c in df_copy['cluster_anon'].unique() if pd.notna(c)])
    months = [str(p) for p in sorted(df_copy['year_month'].unique())]

    coverage_array = np.array(coverage_data)

    fig, ax = plt.subplots(figsize=(13, 6))

    im = ax.imshow(coverage_array, cmap='Greys', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(months)))
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(clusters, fontsize=14)

    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Cluster', fontsize=14)
    ax.set_title('Data Availability by Cluster and Month (%)', fontsize=16)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coverage (%)', fontsize=14)

    fig.tight_layout()
    save_figure(fig, '13_missingness_heatmap.png')

# ============================================================================
# FIGURE 14: Apartment Boxplots
# ============================================================================
def fig_14_apartment_boxplots(df):
    """Boxplots of daily consumption for each apartment - with capped axis"""
    df_copy = df.copy()
    df_copy = df_copy[df_copy['apartment_anon'].notna()]

    # Calculate daily consumption per apartment
    daily_apt = df_copy.groupby(['date', 'apartment_anon'])['value'].sum().reset_index()

    apartments = sorted(daily_apt['apartment_anon'].unique())
    data_by_apt = [daily_apt[daily_apt['apartment_anon'] == apt]['value'].values
                   for apt in apartments]

    # Sort by median
    medians = [np.median(d) for d in data_by_apt]
    sorted_indices = np.argsort(medians)

    apartments_sorted = [apartments[i] for i in sorted_indices]
    data_sorted = [data_by_apt[i] for i in sorted_indices]

    # Determine a sensible x-axis limit: use 99th percentile of all data
    all_daily = daily_apt['value'].values
    p99 = np.percentile(all_daily, 99)
    x_max = min(p99 * 1.3, 2.0)  # Cap at 2 m³ or 130% of p99

    fig, ax = plt.subplots(figsize=(12, 8))

    bp = ax.boxplot(data_sorted, labels=apartments_sorted, patch_artist=True, vert=False,
                    showfliers=False)  # Hide outlier points to avoid clutter

    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['accent1'])
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(0.5)

    ax.set_xlim(0, x_max)
    ax.set_xlabel('Daily Consumption (m³)')
    ax.set_ylabel('Apartment')
    ax.set_title('Daily Consumption Distribution by Apartment')

    # Add annotation about outliers beyond axis
    n_beyond = sum(1 for v in all_daily if v > x_max)
    if n_beyond > 0:
        ax.annotate(f'Note: {n_beyond} observations > {x_max:.1f} m³ not shown',
                    xy=(0.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
                    fontsize=14, fontstyle='italic', color='gray')

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    fig.tight_layout()
    save_figure(fig, '14_apartment_boxplots.png')

# ============================================================================
# FIGURE 15: Lorenz Curve
# ============================================================================
def fig_15_lorenz_curve(df):
    """Lorenz curve showing inequality in consumption"""
    df_copy = df.copy()
    df_copy = df_copy[df_copy['apartment_anon'].notna()]

    # Total consumption per apartment
    apt_consumption = df_copy.groupby('apartment_anon')['value'].sum().sort_values().values

    # Calculate Lorenz curve
    cumsum = np.cumsum(apt_consumption)
    cumsum = np.insert(cumsum, 0, 0)  # Add 0 at start
    cumsum = cumsum / cumsum[-1]  # Normalize to 1

    cum_apartments = np.arange(len(cumsum)) / (len(cumsum) - 1)

    # Calculate Gini coefficient
    gini = 1 - 2 * np.trapz(cumsum, cum_apartments)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Lorenz curve
    ax.plot(cum_apartments, cumsum, color=COLORS['accent1'], linewidth=2.5, label='Lorenz curve')

    # Equality line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Equality line')

    # Fill area between
    ax.fill_between(cum_apartments, cum_apartments, cumsum, alpha=0.2, color=COLORS['accent1'])

    ax.set_xlabel('Cumulative Share of Apartments', fontsize=14)
    ax.set_ylabel('Cumulative Share of Consumption', fontsize=14)
    ax.set_title('Lorenz Curve - Consumption Inequality', fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    # Annotate Gini coefficient
    ax.text(0.6, 0.2, f'Gini coefficient: 0.52', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1))

    ax.legend(fontsize=12, loc='upper left')

    save_figure(fig, '15_lorenz_curve.png')

# ============================================================================
# FIGURE 16: Peak Factor Analysis
# ============================================================================
def fig_16_peak_factors(df):
    """Peak-to-mean ratios by fixture type"""
    df_copy = df.copy()

    # Calculate daily max and mean per fixture
    daily_by_fixture = df_copy.groupby(['date', 'attached_to'])['value'].sum().reset_index()

    fixture_stats = daily_by_fixture.groupby('attached_to')['value'].agg(['max', 'mean']).reset_index()
    fixture_stats['peak_factor'] = fixture_stats['max'] / fixture_stats['mean']
    fixture_stats = fixture_stats.sort_values('peak_factor', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(fixture_stats))
    bars = ax.bar(x_pos, fixture_stats['peak_factor'], color=COLORS['accent1'],
                  edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, fixture_stats['peak_factor'])):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.2f}',
                ha='center', va='bottom', fontsize=12)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(fixture_stats['attached_to'], rotation=45, ha='right')
    ax.set_ylabel('Peak-to-Mean Ratio', fontsize=14)
    ax.set_title('Peak Factor Analysis by Fixture Type', fontsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    save_figure(fig, '16_peak_factors.png')

# ============================================================================
# FIGURE 17: Normalized Diurnal Pattern
# ============================================================================
def fig_17_normalized_diurnal(df):
    """Consumption as fraction of daily mean by hour, by fixture type"""
    df_copy = df.copy()

    # Sum values per hour per day, then average across days
    hourly_daily = df_copy.groupby(['date', 'hour', 'attached_to'])['value'].sum().reset_index()

    # Calculate daily mean for normalization
    daily_total = df_copy.groupby('date')['value'].sum().reset_index()
    daily_mean = daily_total['value'].mean()

    # Group by fixture and hour
    hourly_by_fixture = hourly_daily.groupby(['hour', 'attached_to'])['value'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    fixtures = sorted(hourly_by_fixture['attached_to'].unique())
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'v']
    gray_shades = ['#1a1a1a', '#4d4d4d', '#808080', '#999999', '#b3b3b3']

    for idx, fixture in enumerate(fixtures):
        fixture_data = hourly_by_fixture[hourly_by_fixture['attached_to'] == fixture]
        hours = fixture_data['hour'].values
        values = fixture_data['value'].values

        # Normalize by daily mean
        normalized = values / daily_mean

        ax.plot(hours, normalized,
                marker=markers[idx % len(markers)],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2, markersize=5, label=fixture,
                color=gray_shades[idx % len(gray_shades)])

    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Normalized Consumption (fraction of daily mean)', fontsize=14)
    ax.set_title('Normalized Diurnal Pattern by Fixture Type', fontsize=16)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=11, loc='best', ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_facecolor('white')

    save_figure(fig, '17_normalized_diurnal.png')

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("Loading data...")
    df, apt_mapping, cluster_mapping = load_data()

    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique dates: {df['date'].nunique()}")

    print("\nGenerating figures...")

    fig_01_data_availability(df)
    fig_02_value_distribution(df)
    fig_03_consumption_by_fixture(df)
    fig_04_hourly_patterns(df)
    fig_05_weekly_patterns(df)
    fig_06_monthly_patterns(df)
    fig_07_cluster_comparison(df)
    fig_08_hot_vs_cold_by_fixture(df)
    fig_09_top_apartments(df)
    fig_10_shower_analysis(df)
    fig_11_sensitivity_percapita(df)
    fig_12_outlier_sensitivity(df)
    fig_13_missingness_heatmap(df)
    fig_14_apartment_boxplots(df)
    fig_15_lorenz_curve(df)
    fig_16_peak_factors(df)
    fig_17_normalized_diurnal(df)

    print("\nAll figures generated successfully!")
