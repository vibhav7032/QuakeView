# analysis.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler


def load_and_clean(filepath="earthquake.csv"):
    df = pd.read_csv(filepath)

    # Parse datetime
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['month_name'] = df['time'].dt.strftime('%b')
    df['date'] = df['time'].dt.date

    # Filter only actual earthquakes
    df = df[df['type'] == 'earthquake'].copy()

    # Drop rows where magnitude or coordinates are missing
    df = df.dropna(subset=['mag', 'latitude', 'longitude', 'depth'])

    # Remove clearly bad data
    df = df[df['mag'] > 0]
    df = df[df['depth'] >= 0]

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
        "total_earthquakes": f"{len(df):,}",
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


# def get_monthly_counts(df):
#     order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
#              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#     counts = df.groupby('month_name').size().reindex(order).reset_index(name='count')
#     return counts

def get_monthly_counts(df):
    order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    counts = df.groupby('month_name').size().reindex(order, fill_value=0).reset_index(name='count')
    return counts

def get_magnitude_distribution(df):
    return df['mag']


def get_depth_vs_mag(df, sample_size=5000):
    return df[['depth', 'mag', 'mag_category', 'depth_category']].sample(
        min(sample_size, len(df)), random_state=42
    )


# def get_mag_category_counts(df):
#     order = ['Minor (<2)', 'Light (2-4)', 'Moderate (4-6)', 'Strong (6-7)', 'Major (7+)']
#     counts = df['mag_category'].value_counts().reindex(order).reset_index()
#     counts.columns = ['category', 'count']
#     return counts

def get_mag_category_counts(df):
    order = ['Minor (<2)', 'Light (2-4)', 'Moderate (4-6)', 'Strong (6-7)', 'Major (7+)']
    counts = df['mag_category'].value_counts().reindex(order, fill_value=0).reset_index()
    counts.columns = ['category', 'count']
    counts['count'] = counts['count'].fillna(0).astype(int)
    return counts


def get_depth_category_counts(df):
    order = ['Shallow (<70km)', 'Intermediate (70-300km)', 'Deep (>300km)']
    counts = df['depth_category'].value_counts().reindex(order, fill_value=0).reset_index()
    counts.columns = ['category', 'count']
    counts['count'] = counts['count'].fillna(0).astype(int)
    return counts

def get_heatmap_data(df):
    pivot = df.groupby(['year', 'month']).size().unstack(fill_value=0)
    return pivot


def get_top_regions(df, n=10):
    df = df.copy()
    df['region'] = df['place'].str.extract(r'of (.+)$')
    df['region'] = df['region'].fillna(df['place'])
    result = df['region'].value_counts().head(n).reset_index()
    result.columns = ['region', 'count']
    return result


# def get_depth_category_counts(df):
#     order = ['Shallow (<70km)', 'Intermediate (70-300km)', 'Deep (>300km)']
#     counts = df['depth_category'].value_counts().reindex(order).reset_index()
#     counts.columns = ['category', 'count']
#     return counts


def filter_df(df, year=None, min_mag=0.0, max_mag=10.0):
    filtered = df.copy()
    if year and year != "All":
        filtered = filtered[filtered['year'] == int(year)]
    filtered = filtered[
        (filtered['mag'] >= min_mag) &
        (filtered['mag'] <= max_mag)
    ]
    return filtered


# def get_clusters(df, n_clusters=4):
#     X = df[['latitude', 'longitude', 'depth']]
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     df_clustered = df.copy()
#     df_clustered['cluster'] = kmeans.fit_predict(X_scaled)

#     return df_clustered



def get_clusters(df, n_clusters=4):
    X = df[['latitude', 'longitude', 'depth']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    df_clustered = df.copy()
    raw_clusters = kmeans.fit_predict(X_scaled)
    df_clustered['raw_cluster'] = raw_clusters
    
    # CRITICAL FIX: K-Means assigns random IDs. 
    # We sort them by average depth so Cluster 0 is ALWAYS the shallowest (Surface)
    # and Cluster 3 is ALWAYS the deepest (Subduction Zones).
    cluster_depths = df_clustered.groupby('raw_cluster')['depth'].mean().sort_values()
    depth_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_depths.index)}
    
    df_clustered['cluster'] = df_clustered['raw_cluster'].map(depth_mapping)
    
    return df_clustered