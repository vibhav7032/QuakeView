import analysis
df = analysis.load_and_clean("earthquake.csv")
print("Full:", len(df))
df_f1 = analysis.filter_df(df, year="All", min_mag=0.0, max_mag=10.0)
print("Filter All, 0-10:", len(df_f1))
df_f2 = analysis.filter_df(df, year="2020", min_mag=5.0, max_mag=6.0)
print("Filter 2020, 5-6:", len(df_f2))
