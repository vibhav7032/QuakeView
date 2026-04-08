import tkinter as tk
import main

app = main.EarthquakeDashboard()
app.update()
app._on_data_loaded(app.df_full)   # Wait, df_full might not be populated immediately since it's an async load.

