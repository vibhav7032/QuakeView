# plots.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

# ── Global style ──────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
ACCENT  = "#f97316"        # orange
ACCENT2 = "#38bdf8"        # sky blue
ACCENT3 = "#a78bfa"        # violet
TEXT    = "#e6edf3"
GRID    = "#21262d"

CAT_COLORS = {
    'Minor (<2)':       '#94a3b8',
    'Light (2-4)':      '#38bdf8',
    'Moderate (4-6)':   '#facc15',
    'Strong (6-7)':     '#f97316',
    'Major (7+)':       '#ef4444',
}

DEPTH_COLORS = {
    'Shallow (<70km)':          '#38bdf8',
    'Intermediate (70-300km)':  '#a78bfa',
    'Deep (>300km)':            '#f97316',
}

def _apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(BG)
    for ax in axes_list:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(True, which='major', color=GRID, linewidth=0.7, linestyle='--')
        ax.grid(True, which='minor', color=GRID, linewidth=0.3, linestyle=':')

def _make_fig(figsize=(9, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ── 1. Earthquakes per Year (Bar) ─────────────────────────────────────────────
def plot_yearly_counts(yearly_df):
    fig, ax = _make_fig()
    years  = yearly_df['year'].astype(str)
    counts = yearly_df['count']

    bars = ax.bar(years, counts, color=ACCENT, edgecolor=BG, linewidth=0.5, zorder=3)

    # Highlight max year
    max_idx = counts.idxmax()
    bars[max_idx].set_color(ACCENT2)
    bars[max_idx].set_edgecolor(TEXT)
    bars[max_idx].set_linewidth(1.5)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=7.5,
                color=TEXT, fontweight='bold')

    ax.set_title("Earthquakes per Year", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Number of Earthquakes", fontsize=10)
    ax.set_xticklabels(years, rotation=45, ha='right', fontsize=8)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 2. Magnitude Distribution (Histogram) ─────────────────────────────────────
def plot_magnitude_distribution(mag_series):
    fig, ax = _make_fig()

    n, bins, patches = ax.hist(mag_series, bins=50, edgecolor=BG,
                               linewidth=0.4, zorder=3)

    # Colour each bar by magnitude zone
    for patch, left in zip(patches, bins[:-1]):
        if left < 2:    patch.set_facecolor(CAT_COLORS['Minor (<2)'])
        elif left < 4:  patch.set_facecolor(CAT_COLORS['Light (2-4)'])
        elif left < 6:  patch.set_facecolor(CAT_COLORS['Moderate (4-6)'])
        elif left < 7:  patch.set_facecolor(CAT_COLORS['Strong (6-7)'])
        else:           patch.set_facecolor(CAT_COLORS['Major (7+)'])

    # Legend proxy
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=v, label=k) for k, v in CAT_COLORS.items()]
    ax.legend(handles=legend_els, fontsize=7.5, facecolor=PANEL,
              edgecolor=GRID, labelcolor=TEXT, loc='upper right')

    ax.axvline(mag_series.mean(), color=ACCENT2, linestyle='--',
               linewidth=1.2, label=f'Mean: {mag_series.mean():.2f}', zorder=4)
    ax.text(mag_series.mean() + 0.05, ax.get_ylim()[1] * 0.9,
            f'μ={mag_series.mean():.2f}', color=ACCENT2, fontsize=8)

    ax.set_title("Magnitude Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Magnitude", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 3. Depth vs Magnitude (Scatter) ───────────────────────────────────────────
def plot_depth_vs_magnitude(scatter_df):
    fig, ax = _make_fig()

    for cat, grp in scatter_df.groupby('mag_category'):
        color = CAT_COLORS.get(cat, '#ffffff')
        ax.scatter(grp['depth'], grp['mag'],
                   c=color, alpha=0.35, s=8,
                   edgecolors='none', label=cat, zorder=3)

    # Trend line
    z = np.polyfit(scatter_df['depth'], scatter_df['mag'], 1)
    p = np.poly1d(z)
    xs = np.linspace(scatter_df['depth'].min(), scatter_df['depth'].max(), 200)
    ax.plot(xs, p(xs), color=ACCENT2, linewidth=1.5,
            linestyle='--', label='Trend', zorder=4)

    ax.set_title("Depth vs Magnitude", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Depth (km)", fontsize=10)
    ax.set_ylabel("Magnitude", fontsize=10)
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, markerscale=2)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 4. Monthly Frequency Heatmap ──────────────────────────────────────────────
def plot_monthly_heatmap(pivot_df):
    fig, ax = plt.subplots(figsize=(11, 5))

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Reindex columns to months 1-12
    pivot = pivot_df.reindex(columns=range(1, 13), fill_value=0)
    pivot.columns = month_labels

    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', linewidths=0.3,
                linecolor=BG, annot=True, fmt='d',
                annot_kws={'size': 7, 'color': '#111'},
                cbar_kws={'shrink': 0.7})

    ax.set_title("Earthquake Frequency — Year × Month Heatmap",
                 fontsize=13, fontweight='bold', pad=12, color=TEXT)
    ax.set_xlabel("Month", fontsize=10, color=TEXT)
    ax.set_ylabel("Year", fontsize=10, color=TEXT)
    ax.tick_params(colors=TEXT, labelsize=8)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    cbar.ax.yaxis.label.set_color(TEXT)

    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    fig.tight_layout()
    return fig


# ── 5. Magnitude Category Breakdown (Horizontal Bar) ─────────────────────────
def plot_mag_category_counts(cat_df):
    fig, ax = _make_fig(figsize=(9, 4))

    categories = cat_df['category'].tolist()
    counts     = cat_df['count'].tolist()
    colors     = [CAT_COLORS.get(c, ACCENT) for c in categories]

    bars = ax.barh(categories, counts, color=colors,
                   edgecolor=BG, linewidth=0.5, zorder=3, height=0.55)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', ha='left', fontsize=8.5,
                color=TEXT, fontweight='bold')

    ax.set_title("Earthquakes by Magnitude Category",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.invert_yaxis()
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 6. Monthly Trend Line ─────────────────────────────────────────────────────
def plot_monthly_trend(monthly_df):
    fig, ax = _make_fig()

    months = monthly_df['month_name']
    counts = monthly_df['count']

    ax.fill_between(range(len(months)), counts,
                    alpha=0.25, color=ACCENT3, zorder=2)
    ax.plot(range(len(months)), counts,
            color=ACCENT3, linewidth=2.5, marker='o',
            markersize=6, markerfacecolor=ACCENT, zorder=3)

    for i, (m, v) in enumerate(zip(months, counts)):
        ax.text(i, v + max(counts) * 0.015, f'{v:,}',
                ha='center', fontsize=7.5, color=TEXT)

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, fontsize=9)
    ax.set_title("Average Monthly Earthquake Frequency",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Total Count (all years)", fontsize=10)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 7. Top Regions (Horizontal Bar) ──────────────────────────────────────────
def plot_top_regions(regions_df):
    fig, ax = _make_fig(figsize=(9, 5))

    regions = regions_df['region'].tolist()
    counts  = regions_df['count'].tolist()

    cmap   = plt.cm.get_cmap('plasma', len(regions))
    colors = [cmap(i) for i in range(len(regions))]

    bars = ax.barh(regions, counts, color=colors,
                   edgecolor=BG, linewidth=0.4, zorder=3, height=0.6)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', ha='left',
                fontsize=8, color=TEXT, fontweight='bold')

    ax.set_title("Top 10 Most Active Regions",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=8)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig



# def plot_clusters(df):
#     fig, ax = _make_fig()

#     # Load map
#     img = mpimg.imread("world_map.png")

#     # Set map as background
#     ax.imshow(img, extent=[-180, 180, -90, 90])

#     scatter = ax.scatter(
#         df['longitude'], df['latitude'],
#         c=df['cluster'], cmap='viridis',
#         s=5, alpha=0.5
#     )

#     ax.set_title("Earthquake Clusters (Seismic Zones)")
#     ax.set_xlabel("Longitude")
#     ax.set_ylabel("Latitude")

#     handles = [
#     mpatches.Patch(color=plt.cm.viridis(i/4), label=f'Cluster {i}')
#     for i in range(4)
#     ]

#     ax.legend(handles=handles)

#     _apply_dark_style(fig, [ax])
#     fig.tight_layout()
#     return fig



def plot_clusters(df):
    fig, ax = _make_fig(figsize=(9, 5))

    # Load map safely
    try:
        img = mpimg.imread("world_map.png")
        ax.imshow(img, extent=[-180, 180, -90, 90])
    except Exception:
        pass # Failsafes to a blank dark map if the image is missing

    # Theme-matched colors and Geological names
    colors = ['#38bdf8', '#facc15', '#f97316', '#a78bfa'] # Sky, Yellow, Orange, Violet
    # names = [
    #     "Surface Faults", 
    #     "Intermediate Zones", 
    #     "Deep Boundaries", 
    #     "Subduction Zones"
    # ]

    names = [
        "Northern Shallow Activity", 
        "Americas/Atlantic Shallow", 
        "Indo-Pacific Shallow", 
        "Global Subduction Zones (Deep)"
    ]

    # Calculate cluster profiles to prove the ML worked
    profiles = df.groupby('cluster').agg(
        Count=('mag', 'count'), 
        Avg_Depth=('depth', 'mean'), 
        Avg_Mag=('mag', 'mean')
    ).round(2)

    for i in range(4):
        cluster_data = df[df['cluster'] == i]
        ax.scatter(
            cluster_data['longitude'], cluster_data['latitude'],
            c=colors[i], s=8, alpha=0.6, edgecolors='none',
            label=f"{names[i]} (μ Depth: {profiles.loc[i, 'Avg_Depth']}km)"
        )

    ax.set_title("Machine Learning: Discovered Seismic Zones", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, loc='lower left', markerscale=2)

    # Draw the profile table directly onto the bottom right of the map
    stats_text = "Data Story: Cluster Profiles\n" + "-"*32 + "\n"
    for i in range(4):
        stats_text += f"Zone {i}: {int(profiles.loc[i, 'Count']):,} events | μ Mag: {profiles.loc[i, 'Avg_Mag']}\n"
    
    props = dict(boxstyle='round,pad=0.6', facecolor=BG, alpha=0.85, edgecolor=GRID)
    ax.text(0.98, 0.05, stats_text.strip(), transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, color=TEXT)

    _apply_dark_style(fig, [ax])
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    fig.tight_layout()
    return fig
# plots.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
ACCENT  = "#f97316"        # orange
ACCENT2 = "#38bdf8"        # sky blue
ACCENT3 = "#a78bfa"        # violet
TEXT    = "#e6edf3"
GRID    = "#21262d"

CAT_COLORS = {
    'Minor (<2)':       '#94a3b8',
    'Light (2-4)':      '#38bdf8',
    'Moderate (4-6)':   '#facc15',
    'Strong (6-7)':     '#f97316',
    'Major (7+)':       '#ef4444',
}

DEPTH_COLORS = {
    'Shallow (<70km)':          '#38bdf8',
    'Intermediate (70-300km)':  '#a78bfa',
    'Deep (>300km)':            '#f97316',
}

def _apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(BG)
    for ax in axes_list:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(True, which='major', color=GRID, linewidth=0.7, linestyle='--')
        ax.grid(True, which='minor', color=GRID, linewidth=0.3, linestyle=':')

def _make_fig(figsize=(9, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ── 1. Earthquakes per Year (Bar) ─────────────────────────────────────────────
def plot_yearly_counts(yearly_df):
    fig, ax = _make_fig()
    years  = yearly_df['year'].astype(str)
    counts = yearly_df['count']

    bars = ax.bar(years, counts, color=ACCENT, edgecolor=BG, linewidth=0.5, zorder=3)

    # Highlight max year
    max_idx = counts.idxmax()
    bars[max_idx].set_color(ACCENT2)
    bars[max_idx].set_edgecolor(TEXT)
    bars[max_idx].set_linewidth(1.5)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=7.5,
                color=TEXT, fontweight='bold')

    ax.set_title("Earthquakes per Year", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Number of Earthquakes", fontsize=10)
    ax.set_xticklabels(years, rotation=45, ha='right', fontsize=8)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 2. Magnitude Distribution (Histogram) ─────────────────────────────────────
def plot_magnitude_distribution(mag_series):
    fig, ax = _make_fig()

    n, bins, patches = ax.hist(mag_series, bins=50, edgecolor=BG,
                               linewidth=0.4, zorder=3)

    # Colour each bar by magnitude zone
    for patch, left in zip(patches, bins[:-1]):
        if left < 2:    patch.set_facecolor(CAT_COLORS['Minor (<2)'])
        elif left < 4:  patch.set_facecolor(CAT_COLORS['Light (2-4)'])
        elif left < 6:  patch.set_facecolor(CAT_COLORS['Moderate (4-6)'])
        elif left < 7:  patch.set_facecolor(CAT_COLORS['Strong (6-7)'])
        else:           patch.set_facecolor(CAT_COLORS['Major (7+)'])

    # Legend proxy
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=v, label=k) for k, v in CAT_COLORS.items()]
    ax.legend(handles=legend_els, fontsize=7.5, facecolor=PANEL,
              edgecolor=GRID, labelcolor=TEXT, loc='upper right')

    ax.axvline(mag_series.mean(), color=ACCENT2, linestyle='--',
               linewidth=1.2, label=f'Mean: {mag_series.mean():.2f}', zorder=4)
    ax.text(mag_series.mean() + 0.05, ax.get_ylim()[1] * 0.9,
            f'μ={mag_series.mean():.2f}', color=ACCENT2, fontsize=8)

    ax.set_title("Magnitude Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Magnitude", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 3. Depth vs Magnitude (Scatter) ───────────────────────────────────────────
def plot_depth_vs_magnitude(scatter_df):
    fig, ax = _make_fig()

    for cat, grp in scatter_df.groupby('mag_category'):
        color = CAT_COLORS.get(cat, '#ffffff')
        ax.scatter(grp['depth'], grp['mag'],
                   c=color, alpha=0.35, s=8,
                   edgecolors='none', label=cat, zorder=3)

    # Trend line
    z = np.polyfit(scatter_df['depth'], scatter_df['mag'], 1)
    p = np.poly1d(z)
    xs = np.linspace(scatter_df['depth'].min(), scatter_df['depth'].max(), 200)
    ax.plot(xs, p(xs), color=ACCENT2, linewidth=1.5,
            linestyle='--', label='Trend', zorder=4)

    ax.set_title("Depth vs Magnitude", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Depth (km)", fontsize=10)
    ax.set_ylabel("Magnitude", fontsize=10)
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, markerscale=2)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 4. Monthly Frequency Heatmap ──────────────────────────────────────────────
def plot_monthly_heatmap(pivot_df):
    fig, ax = plt.subplots(figsize=(11, 5))

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Reindex columns to months 1-12
    pivot = pivot_df.reindex(columns=range(1, 13), fill_value=0)
    pivot.columns = month_labels

    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', linewidths=0.3,
                linecolor=BG, annot=True, fmt='d',
                annot_kws={'size': 7, 'color': '#111'},
                cbar_kws={'shrink': 0.7})

    ax.set_title("Earthquake Frequency — Year × Month Heatmap",
                 fontsize=13, fontweight='bold', pad=12, color=TEXT)
    ax.set_xlabel("Month", fontsize=10, color=TEXT)
    ax.set_ylabel("Year", fontsize=10, color=TEXT)
    ax.tick_params(colors=TEXT, labelsize=8)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    cbar.ax.yaxis.label.set_color(TEXT)

    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    fig.tight_layout()
    return fig


# ── 5. Magnitude Category Breakdown (Horizontal Bar) ─────────────────────────
def plot_mag_category_counts(cat_df):
    fig, ax = _make_fig(figsize=(9, 4))

    categories = cat_df['category'].tolist()
    counts     = cat_df['count'].tolist()
    colors     = [CAT_COLORS.get(c, ACCENT) for c in categories]

    bars = ax.barh(categories, counts, color=colors,
                   edgecolor=BG, linewidth=0.5, zorder=3, height=0.55)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', ha='left', fontsize=8.5,
                color=TEXT, fontweight='bold')

    ax.set_title("Earthquakes by Magnitude Category",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.invert_yaxis()
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 6. Monthly Trend Line ─────────────────────────────────────────────────────
def plot_monthly_trend(monthly_df):
    fig, ax = _make_fig()

    months = monthly_df['month_name']
    counts = monthly_df['count']

    ax.fill_between(range(len(months)), counts,
                    alpha=0.25, color=ACCENT3, zorder=2)
    ax.plot(range(len(months)), counts,
            color=ACCENT3, linewidth=2.5, marker='o',
            markersize=6, markerfacecolor=ACCENT, zorder=3)

    for i, (m, v) in enumerate(zip(months, counts)):
        ax.text(i, v + max(counts) * 0.015, f'{v:,}',
                ha='center', fontsize=7.5, color=TEXT)

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, fontsize=9)
    ax.set_title("Average Monthly Earthquake Frequency",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Total Count (all years)", fontsize=10)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ── 7. Top Regions (Horizontal Bar) ──────────────────────────────────────────
def plot_top_regions(regions_df):
    fig, ax = _make_fig(figsize=(9, 5))

    regions = regions_df['region'].tolist()
    counts  = regions_df['count'].tolist()

    cmap   = plt.cm.get_cmap('plasma', len(regions))
    colors = [cmap(i) for i in range(len(regions))]

    bars = ax.barh(regions, counts, color=colors,
                   edgecolor=BG, linewidth=0.4, zorder=3, height=0.6)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', ha='left',
                fontsize=8, color=TEXT, fontweight='bold')

    ax.set_title("Top 10 Most Active Regions",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=8)
    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig