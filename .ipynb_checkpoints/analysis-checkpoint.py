# analysis.py
import pandas as pd
import numpy as np

def load_and_clean(filepath="earthquake.csv"):
    df = pd.read_csv(filepath)

    # Parse datetime
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['month_name'] = df['time'].dt.strftime('%b')
    df['date'] = df['time'].dt.date

    # Filter only actual earthquakes (your 'type' column has this)
    df = df[df['type'] == 'earthquake'].copy()

    # Drop rows where magnitude or coordinates are missing
    df = df.dropna(subset=['mag', 'latitude', 'longitude', 'depth'])

    # Magnitude categories
    def categorize_mag(m):
        if m < 2.0:   return 'Minor (<2)'
        elif m < 4.0: return 'Light (2-4)'
        elif m < 6.0: return 'Moderate (4-6)'
        elif m < 7.0: return 'Strong (6-7)'
        else:         return 'Major (7+)'

    df['mag_category'] = df['mag'].apply(categorize_mag)

    # Depth categories
    def categorize_depth(d):
        if d < 70:    return 'Shallow (<70km)'
        elif d < 300: return 'Intermediate (70-300km)'
        else:         return 'Deep (>300km)'

    df['depth_category'] = df['depth'].apply(categorize_depth)

    return df


def get_summary_stats(df):
    return {
        "total_earthquakes": len(df),
        "avg_magnitude":     round(df['mag'].mean(), 2),
        "max_magnitude":     round(df['mag'].max(), 2),
        "min_magnitude":     round(df['mag'].min(), 2),
        "avg_depth_km":      round(df['depth'].mean(), 2),
        "years_covered":     sorted(df['year'].unique().tolist()),
        "most_active_year":  int(df['year'].value_counts().idxmax()),
        "major_quakes":      len(df[df['mag'] >= 7.0]),
    }


def get_yearly_counts(df):
    return df.groupby('year').size().reset_index(name='count')


def get_monthly_counts(df):
    order = ['Jan','Feb','Mar','Apr','May','Jun',
             'Jul','Aug','Sep','Oct','Nov','Dec']
    counts = df.groupby('month_name').size().reindex(order).reset_index(name='count')
    return counts


def get_magnitude_distribution(df):
    return df['mag']   # raw series, histogram in plots.py


def get_depth_vs_mag(df, sample_size=5000):
    # Scatter with too many points is slow — sample it
    return df[['depth', 'mag', 'mag_category']].sample(
        min(sample_size, len(df)), random_state=42
    )


def get_mag_category_counts(df):
    order = ['Minor (<2)', 'Light (2-4)', 'Moderate (4-6)', 'Strong (6-7)', 'Major (7+)']
    counts = df['mag_category'].value_counts().reindex(order).reset_index()
    counts.columns = ['category', 'count']
    return counts


def get_heatmap_data(df):
    # Rows = months, Cols = years — good for a heatmap
    pivot = df.groupby(['year', 'month']).size().unstack(fill_value=0)
    return pivot


def get_top_regions(df, n=10):
    # Extract region from 'place' column (everything after "of ")
    df = df.copy()
    df['region'] = df['place'].str.extract(r'of (.+)$')
    df['region'] = df['region'].fillna(df['place'])
    return df['region'].value_counts().head(n).reset_index(name='count').rename(columns={'index':'region'})


def filter_df(df, year=None, min_mag=None, max_mag=None):
    filtered = df.copy()
    if year and year != "All":
        filtered = filtered[filtered['year'] == int(year)]
    if min_mag is not None:
        filtered = filtered[filtered['mag'] >= min_mag]
    if max_mag is not None:
        filtered = filtered[filtered['mag'] <= max_mag]
    return filtered