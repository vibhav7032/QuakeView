import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import threading
import time
import requests
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ==================================================
# THEME — Deep Charcoal + Seismic Red
# ==================================================
COLOR_BG       = "#111318"   # Near-black charcoal
COLOR_SIDEBAR  = "#1a1d26"   # Dark navy-charcoal
COLOR_CARD     = "#1f2333"   # Card background
COLOR_ACCENT   = "#e63946"   # Seismic Red (primary)
COLOR_ACCENT2  = "#f4a261"   # Amber (secondary)
COLOR_TEXT     = "#f0f4f8"   # Off-white text
COLOR_SUBTEXT  = "#8892a4"   # Muted grey text
COLOR_DANGER   = "#ff4d4d"   # Bright red
COLOR_WARNING  = "#f4a261"   # Orange-amber
COLOR_SUCCESS  = "#2dc653"   # Green
COLOR_LOW      = "#4cc9f0"   # Light blue (low mag)

FONT_H1    = ("Segoe UI", 20, "bold")
FONT_H2    = ("Segoe UI", 14, "bold")
FONT_BODY  = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 11)

plt.rcParams.update({
    'figure.facecolor':  COLOR_BG,
    'axes.facecolor':    COLOR_CARD,
    'text.color':        COLOR_TEXT,
    'axes.labelcolor':   COLOR_TEXT,
    'xtick.color':       COLOR_SUBTEXT,
    'ytick.color':       COLOR_SUBTEXT,
    'axes.edgecolor':    '#2e3348',
    'grid.color':        '#2e3348',
    'axes.titlecolor':   COLOR_TEXT,
})


# ==================================================
# DATA ENGINE
# ==================================================
class SeismicBrain:
    def __init__(self):
        self.df = None
        self.model = None
        self.le = LabelEncoder()
        self.accuracy = 0.0
        self.stats = {}
        self.feature_importance = None

    def initialize(self, filepath="earthquake.csv"):
        try:
            raw = pd.read_csv(filepath, low_memory=False)

            # --- Normalise column names (USGS exports sometimes vary) ---
            raw.columns = [c.strip().lower() for c in raw.columns]

            # Rename common USGS column variants
            renames = {}
            for c in raw.columns:
                if c in ('magnitude', 'mag'):
                    renames[c] = 'mag'
                elif c in ('depth', 'deptherror'):
                    if c == 'depth':
                        renames[c] = 'depth'
                elif c in ('time', 'date', 'datetime', 'origintime'):
                    renames[c] = 'time'
                elif c in ('latitude', 'lat'):
                    renames[c] = 'latitude'
                elif c in ('longitude', 'lon', 'long'):
                    renames[c] = 'longitude'
                elif c in ('place', 'location', 'region'):
                    renames[c] = 'place'
                elif c in ('type', 'eventtype'):
                    renames[c] = 'type'
            raw.rename(columns=renames, inplace=True)

            # Ensure required columns exist
            for col in ('mag', 'depth', 'latitude', 'longitude', 'time'):
                if col not in raw.columns:
                    raise ValueError(f"Required column '{col}' not found. "
                                     f"Found: {list(raw.columns)}")

            # Parse time
            raw['time'] = pd.to_datetime(raw['time'], errors='coerce', utc=True)
            raw = raw.dropna(subset=['time', 'mag', 'depth', 'latitude', 'longitude'])
            raw['year']  = raw['time'].dt.year
            raw['month'] = raw['time'].dt.month
            raw['day']   = raw['time'].dt.day

            # Filter to earthquakes only if column exists
            if 'type' in raw.columns:
                raw = raw[raw['type'].str.lower().str.contains('earthquake', na=False)]

            # Drop obvious outliers
            raw = raw[(raw['mag'] >= 0) & (raw['mag'] <= 10)]
            raw = raw[(raw['depth'] >= 0) & (raw['depth'] <= 800)]

            self.df = raw.reset_index(drop=True)

            # ---------- stats ----------
            self.stats['total']   = len(self.df)
            self.stats['avg_mag'] = round(self.df['mag'].mean(), 2)
            self.stats['max_mag'] = round(self.df['mag'].max(), 2)
            self.stats['avg_dep'] = round(self.df['depth'].mean(), 1)
            self.stats['year_min'] = int(self.df['year'].min())
            self.stats['year_max'] = int(self.df['year'].max())

            # ---------- ML: classify alert level (low/moderate/high) ----------
            def mag_label(m):
                if m < 4.0:   return 'Low'
                elif m < 6.0: return 'Moderate'
                else:          return 'High'

            self.df['alert'] = self.df['mag'].apply(mag_label)

            sample = self.df.sample(frac=0.25, random_state=42)
            X = sample[['depth', 'latitude', 'longitude', 'year', 'month']]
            y = self.le.fit_transform(sample['alert'])

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = RandomForestClassifier(n_estimators=60, max_depth=10, n_jobs=-1, random_state=42)
            self.model.fit(X_tr, y_tr)
            self.accuracy = round(self.model.score(X_te, y_te) * 100, 2)

            self.feature_importance = dict(
                zip(['Depth', 'Latitude', 'Longitude', 'Year', 'Month'],
                    self.model.feature_importances_)
            )

            return True

        except Exception as e:
            print(f"[SeismicBrain ERROR] {e}")
            return False, str(e)


brain = SeismicBrain()


# ==================================================
# MAIN APPLICATION
# ==================================================
class SeismicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("S.E.I.S  |  Seismic Earth Intelligence System")
        self.root.geometry("1500x950")
        self.root.configure(bg=COLOR_BG)
        self.root.minsize(1200, 750)

        self.is_tracking = False
        self._iss_marker = None

        # ---- Load data ----
        result = brain.initialize()
        if result is not True:
            messagebox.showerror(
                "Data Load Error",
                "Could not load earthquake.csv.\n\n"
                "Make sure the file is in the same folder as this script.\n\n"
                "Tip: USGS CSV columns needed: time, latitude, longitude, depth, mag"
            )
            root.destroy()
            return

        self._build_layout()
        self.show_dashboard()

    # --------------------------------------------------
    # LAYOUT SKELETON
    # --------------------------------------------------
    def _build_layout(self):
        # SIDEBAR
        self.sidebar = tk.Frame(self.root, bg=COLOR_SIDEBAR, width=270)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo area
        logo_frame = tk.Frame(self.sidebar, bg=COLOR_SIDEBAR)
        logo_frame.pack(fill="x", padx=20, pady=(30, 10))
        tk.Label(logo_frame, text="S.E.I.S", bg=COLOR_SIDEBAR, fg=COLOR_ACCENT,
                 font=("Segoe UI", 28, "bold")).pack(anchor="w")
        tk.Label(logo_frame, text="Seismic Earth Intelligence System",
                 bg=COLOR_SIDEBAR, fg=COLOR_SUBTEXT, font=FONT_SMALL,
                 wraplength=220, justify="left").pack(anchor="w", pady=(0, 20))

        tk.Frame(self.sidebar, bg="#2e3348", height=1).pack(fill="x", padx=20, pady=5)

        # Nav buttons
        self._nav_buttons = {}
        nav_items = [
            ("📊  Dashboard",        self.show_dashboard),
            ("📈  Deep Analysis",    self.show_analysis),
            ("🤖  Predictive AI",    self.show_prediction),
            ("🛰️  Live USGS Feed",   self.show_live_feed),
        ]
        for label, cmd in nav_items:
            btn = tk.Button(
                self.sidebar, text=label, bg=COLOR_SIDEBAR, fg=COLOR_TEXT,
                font=("Segoe UI", 11), bd=0, anchor="w", padx=24, pady=13,
                activebackground=COLOR_ACCENT, activeforeground="white",
                cursor="hand2", command=cmd, relief="flat"
            )
            btn.pack(fill="x", pady=1)
            self._nav_buttons[label] = btn

        # Bottom status panel
        status = tk.Frame(self.sidebar, bg=COLOR_BG, padx=14, pady=14)
        status.pack(side="bottom", fill="x", padx=16, pady=16)
        tk.Label(status, text="SYSTEM STATUS", bg=COLOR_BG, fg="#444",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(status, text="● ONLINE", bg=COLOR_BG, fg=COLOR_SUCCESS,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(status, text=f"AI Accuracy: {brain.accuracy}%",
                 bg=COLOR_BG, fg=COLOR_SUBTEXT, font=FONT_BODY).pack(anchor="w")
        tk.Label(status,
                 text=f"Dataset: {brain.stats['year_min']}–{brain.stats['year_max']}",
                 bg=COLOR_BG, fg=COLOR_SUBTEXT, font=FONT_BODY).pack(anchor="w")

        # MAIN CONTENT
        self.main = tk.Frame(self.root, bg=COLOR_BG)
        self.main.pack(side="left", fill="both", expand=True)

    def _nav_btn(self, text, command):
        return tk.Button(
            self.sidebar, text=text, bg=COLOR_SIDEBAR, fg=COLOR_TEXT,
            font=("Segoe UI", 11), bd=0, anchor="w", padx=24, pady=13,
            activebackground=COLOR_ACCENT, activeforeground="white",
            cursor="hand2", command=command, relief="flat"
        )

    def _clear(self):
        self.is_tracking = False
        for w in self.main.winfo_children():
            w.destroy()

    def _page_header(self, title, subtitle=""):
        hdr = tk.Frame(self.main, bg=COLOR_BG)
        hdr.pack(fill="x", padx=30, pady=(24, 0))
        tk.Label(hdr, text=title, bg=COLOR_BG, fg=COLOR_TEXT, font=FONT_H1).pack(anchor="w")
        if subtitle:
            tk.Label(hdr, text=subtitle, bg=COLOR_BG, fg=COLOR_SUBTEXT,
                     font=FONT_BODY).pack(anchor="w", pady=(2, 0))
        tk.Frame(self.main, bg="#2e3348", height=1).pack(fill="x", padx=30, pady=10)

    # --------------------------------------------------
    # STAT CARD HELPER
    # --------------------------------------------------
    def _stat_card(self, parent, title, value, sub, accent):
        card = tk.Frame(parent, bg=COLOR_CARD, padx=18, pady=14)
        card.pack(side="left", fill="both", expand=True, padx=6)
        tk.Frame(card, bg=accent, height=3, width=60).pack(anchor="w", pady=(0, 8))
        tk.Label(card, text=title, bg=COLOR_CARD, fg=COLOR_SUBTEXT,
                 font=FONT_SMALL).pack(anchor="w")
        tk.Label(card, text=value, bg=COLOR_CARD, fg=accent,
                 font=("Segoe UI", 26, "bold")).pack(anchor="w")
        tk.Label(card, text=sub, bg=COLOR_CARD, fg="#444",
                 font=FONT_SMALL).pack(anchor="w", pady=(2, 0))

    # ==================================================
    # PAGE 1 — EXECUTIVE DASHBOARD
    # ==================================================
    def show_dashboard(self):
        self._clear()
        self._page_header("Executive Dashboard",
                          f"Global seismic activity · {brain.stats['year_min']}–{brain.stats['year_max']}")

        # Stat cards row
        cards_row = tk.Frame(self.main, bg=COLOR_BG)
        cards_row.pack(fill="x", padx=24, pady=(0, 16))
        self._stat_card(cards_row, "TOTAL EVENTS",
                        f"{brain.stats['total']:,}", "Earthquake records", COLOR_ACCENT)
        self._stat_card(cards_row, "MAX MAGNITUDE",
                        f"{brain.stats['max_mag']}", "Most powerful event", COLOR_DANGER)
        self._stat_card(cards_row, "AVG MAGNITUDE",
                        f"{brain.stats['avg_mag']}", "Global average", COLOR_WARNING)
        self._stat_card(cards_row, "AVG DEPTH",
                        f"{brain.stats['avg_dep']} km", "Below surface", COLOR_SUCCESS)
        self._stat_card(cards_row, "AI ACCURACY",
                        f"{brain.accuracy}%", "Random Forest R²", "#4cc9f0")

        # Bottom: two-panel (map left, quick graphs right)
        bottom = tk.Frame(self.main, bg=COLOR_BG)
        bottom.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        # --- Map panel ---
        map_lf = tk.LabelFrame(bottom, text="  Geospatial Distribution  ",
                               bg=COLOR_CARD, fg=COLOR_SUBTEXT,
                               font=FONT_SMALL, bd=0)
        map_lf.pack(side="left", fill="both", expand=True, padx=(0, 8))

        try:
            import tkintermapview
            self.dash_map = tkintermapview.TkinterMapView(map_lf, corner_radius=0)
            self.dash_map.pack(fill="both", expand=True)
            self.dash_map.set_tile_server(
                "https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
            self.dash_map.set_position(20, 0)
            self.dash_map.set_zoom(2)

            # Sample markers coloured by magnitude
            sample = brain.df.sample(n=min(120, len(brain.df)), random_state=7)
            for _, row in sample.iterrows():
                c = "red" if row['mag'] >= 6 else ("orange" if row['mag'] >= 4 else "blue")
                self.dash_map.set_marker(row['latitude'], row['longitude'],
                                         marker_color_circle=c,
                                         marker_color_outside=c)
        except ImportError:
            tk.Label(map_lf,
                     text="Install tkintermapview for map:\npip install tkintermapview",
                     bg=COLOR_CARD, fg=COLOR_SUBTEXT, font=FONT_BODY).pack(expand=True)

        # --- Quick charts panel ---
        charts_col = tk.Frame(bottom, bg=COLOR_BG, width=380)
        charts_col.pack(side="right", fill="y", padx=(8, 0))
        charts_col.pack_propagate(False)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 7), dpi=90)
        fig.patch.set_facecolor(COLOR_BG)
        fig.subplots_adjust(hspace=0.5)

        # Graph 1 — earthquakes per year (bar)
        yr_counts = brain.df.groupby('year').size()
        ax1.bar(yr_counts.index, yr_counts.values, color=COLOR_ACCENT, width=0.7)
        ax1.set_title("Events per Year", pad=8, fontsize=11)
        ax1.set_xlabel("Year", fontsize=9)
        ax1.set_ylabel("Count", fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # Graph 2 — magnitude distribution (histogram)
        ax2.hist(brain.df['mag'], bins=30, color=COLOR_ACCENT2, edgecolor=COLOR_BG)
        ax2.set_title("Magnitude Distribution", pad=8, fontsize=11)
        ax2.set_xlabel("Magnitude", fontsize=9)
        ax2.set_ylabel("Frequency", fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=charts_col)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==================================================
    # PAGE 2 — DEEP ANALYSIS (5 graph types)
    # ==================================================
    def show_analysis(self):
        self._clear()
        self._page_header("Deep Analysis", "Explore multiple dimensions of seismic data")

        # ---- TOOLBAR ----
        toolbar = tk.Frame(self.main, bg=COLOR_CARD, pady=8)
        toolbar.pack(fill="x", padx=24, pady=(0, 12))

        tk.Label(toolbar, text="Visualization:", bg=COLOR_CARD, fg=COLOR_SUBTEXT,
                 font=FONT_BODY).pack(side="left", padx=(16, 8))

        graphs = [
            ("📉  Trend",         "trend"),
            ("🗂️  Depth Profile", "depth"),
            ("🔵  Scatter",       "scatter"),
            ("🌡️  Heatmap",       "heatmap"),
            ("🍩  Donut",         "donut"),
            ("📊  Feature Imp.",  "feat"),
        ]
        for label, mode in graphs:
            tk.Button(toolbar, text=label, bg="#2a2f45", fg=COLOR_TEXT,
                      font=("Segoe UI", 10), bd=0, padx=12, pady=6,
                      activebackground=COLOR_ACCENT, activeforeground="white",
                      cursor="hand2",
                      command=lambda m=mode: self._render_graph(m)).pack(
                side="left", padx=4)

        # Year filter widgets (Combobox + Scale)
        tk.Frame(toolbar, bg=COLOR_CARD, width=1).pack(side="left", padx=16)
        tk.Label(toolbar, text="Filter year:", bg=COLOR_CARD, fg=COLOR_SUBTEXT,
                 font=FONT_BODY).pack(side="left")

        years = sorted(brain.df['year'].unique().tolist())
        self.year_var = tk.StringVar(value="All")
        year_combo = ttk.Combobox(toolbar, textvariable=self.year_var,
                                  values=["All"] + [str(y) for y in years],
                                  width=8, state="readonly")
        year_combo.pack(side="left", padx=8)
        year_combo.bind("<<ComboboxSelected>>",
                        lambda e: self._render_graph(self._current_mode))

        # Min magnitude scale slider
        tk.Label(toolbar, text="Min mag:", bg=COLOR_CARD, fg=COLOR_SUBTEXT,
                 font=FONT_BODY).pack(side="left", padx=(12, 4))
        self.min_mag_var = tk.DoubleVar(value=0.0)
        mag_scale = tk.Scale(toolbar, from_=0, to=9, resolution=0.5,
                             orient="horizontal", variable=self.min_mag_var,
                             bg=COLOR_CARD, fg=COLOR_TEXT, troughcolor="#2a2f45",
                             highlightthickness=0, length=110,
                             command=lambda v: self._render_graph(self._current_mode))
        mag_scale.pack(side="left", padx=4)

        # Canvas area
        self._graph_frame = tk.Frame(self.main, bg=COLOR_BG)
        self._graph_frame.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        self._current_mode = "trend"
        self._render_graph("trend")

    def _filtered_df(self):
        df = brain.df.copy()
        yr = self.year_var.get() if hasattr(self, 'year_var') else "All"
        if yr != "All":
            df = df[df['year'] == int(yr)]
        min_m = self.min_mag_var.get() if hasattr(self, 'min_mag_var') else 0
        df = df[df['mag'] >= min_m]
        return df

    def _render_graph(self, mode):
        self._current_mode = mode
        for w in self._graph_frame.winfo_children():
            w.destroy()

        df = self._filtered_df()

        if df.empty:
            tk.Label(self._graph_frame,
                     text="No data for selected filters.",
                     bg=COLOR_BG, fg=COLOR_SUBTEXT, font=FONT_H2).pack(expand=True)
            return

        fig = plt.Figure(figsize=(11, 6), dpi=95)

        if mode == "trend":
            ax = fig.add_subplot(111)
            trend = df.groupby('year')['mag'].agg(['mean', 'max', 'count'])
            ax.plot(trend.index, trend['mean'], color=COLOR_ACCENT,
                    marker='o', linewidth=2.5, label='Avg Magnitude')
            ax2 = ax.twinx()
            ax2.bar(trend.index, trend['count'], alpha=0.25,
                    color=COLOR_ACCENT2, label='Event Count')
            ax2.set_ylabel("Event Count", color=COLOR_ACCENT2, fontsize=9)
            ax2.tick_params(colors=COLOR_ACCENT2)
            ax.set_title("Yearly Seismic Activity Trend", fontsize=13, pad=12)
            ax.set_xlabel("Year")
            ax.set_ylabel("Avg Magnitude")
            ax.legend(loc='upper left')
            ax.grid(alpha=0.3)

        elif mode == "depth":
            ax = fig.add_subplot(111)
            bins_depth = [0, 70, 300, 800]
            labels_d = ['Shallow (0–70 km)', 'Intermediate (70–300 km)', 'Deep (300–800 km)']
            df['depth_cat'] = pd.cut(df['depth'], bins=bins_depth, labels=labels_d)
            depth_mag = df.groupby('depth_cat')['mag'].mean()
            colors_d = [COLOR_ACCENT2, COLOR_ACCENT, "#9b5de5"]
            bars = ax.bar(depth_mag.index, depth_mag.values,
                          color=colors_d[:len(depth_mag)], edgecolor=COLOR_BG, width=0.5)
            ax.bar_label(bars, fmt='%.2f', color=COLOR_TEXT, fontsize=10, padding=4)
            ax.set_title("Average Magnitude by Depth Category", fontsize=13, pad=12)
            ax.set_ylabel("Avg Magnitude")
            ax.set_xlabel("Depth Category")
            ax.grid(axis='y', alpha=0.3)

        elif mode == "scatter":
            ax = fig.add_subplot(111)
            s = df.sample(min(3000, len(df)), random_state=42)
            sc = ax.scatter(s['longitude'], s['latitude'], c=s['mag'],
                            cmap='hot', s=4, alpha=0.6)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Magnitude", color=COLOR_TEXT)
            cbar.ax.yaxis.set_tick_params(color=COLOR_TEXT)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLOR_TEXT)
            ax.set_title("Geospatial Scatter — Magnitude Intensity", fontsize=13, pad=12)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(alpha=0.2)

        elif mode == "heatmap":
            ax = fig.add_subplot(111)
            cols = ['mag', 'depth', 'latitude', 'longitude']
            available = [c for c in cols if c in df.columns]
            corr = df[available].corr()
            mask = np.zeros_like(corr, dtype=bool)
            np.fill_diagonal(mask, True)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlBu_r',
                        ax=ax, linewidths=0.5, linecolor='#111',
                        annot_kws={"size": 11, "color": "white"})
            ax.set_title("Feature Correlation Matrix", fontsize=13, pad=12)
            ax.tick_params(colors=COLOR_TEXT, labelsize=10)

        elif mode == "donut":
            ax = fig.add_subplot(111)
            def mag_cat(m):
                if m < 2.0:   return "Minor (<2)"
                elif m < 4.0: return "Light (2–4)"
                elif m < 6.0: return "Moderate (4–6)"
                elif m < 7.0: return "Strong (6–7)"
                else:          return "Major (7+)"
            counts = df['mag'].apply(mag_cat).value_counts()
            colors_pie = [COLOR_LOW, COLOR_SUCCESS, COLOR_WARNING, COLOR_ACCENT, COLOR_DANGER]
            wedges, texts, autotexts = ax.pie(
                counts, labels=counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors_pie[:len(counts)], pctdistance=0.80)
            for t in texts:    t.set_color(COLOR_TEXT)
            for t in autotexts: t.set_color("white"); t.set_fontsize(9)
            # Draw donut hole
            ax.add_artist(plt.Circle((0, 0), 0.60, fc=COLOR_BG))
            ax.text(0, 0, f"{len(df):,}\nevents", ha='center', va='center',
                    color=COLOR_TEXT, fontsize=11, fontweight='bold')
            ax.set_title("Earthquake Magnitude Composition", fontsize=13, pad=12)

        elif mode == "feat":
            ax = fig.add_subplot(111)
            if brain.feature_importance:
                fi = brain.feature_importance
                feats = list(fi.keys())
                vals  = list(fi.values())
                colors_f = [COLOR_ACCENT if v == max(vals) else COLOR_ACCENT2 for v in vals]
                bars = ax.barh(feats, vals, color=colors_f, edgecolor=COLOR_BG, height=0.5)
                ax.bar_label(bars, fmt='%.3f', color=COLOR_TEXT, padding=4, fontsize=10)
                ax.set_title("AI Feature Importance (Random Forest)", fontsize=13, pad=12)
                ax.set_xlabel("Importance Score")
                ax.grid(axis='x', alpha=0.3)
            else:
                ax.text(0.5, 0.5, "Feature importance not available",
                        ha='center', va='center', transform=ax.transAxes, color=COLOR_SUBTEXT)

        canvas = FigureCanvasTkAgg(fig, master=self._graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==================================================
    # PAGE 3 — PREDICTIVE AI
    # ==================================================
    def show_prediction(self):
        self._clear()
        self._page_header("Predictive AI",
                          "Random Forest classifier — predict earthquake alert level")

        # Two columns
        content = tk.Frame(self.main, bg=COLOR_BG)
        content.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        # --- LEFT: Input Panel ---
        left = tk.Frame(content, bg=COLOR_CARD, width=380)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)

        tk.Label(left, text="Simulation Inputs", bg=COLOR_CARD, fg=COLOR_TEXT,
                 font=FONT_H2, pady=20).pack()

        tk.Frame(left, bg="#2e3348", height=1).pack(fill="x", padx=20, pady=(0, 16))

        self._pred_entries = {}
        fields = [
            ("Latitude",            "35.0",  -90,  90),
            ("Longitude",           "-118.0", -180, 180),
            ("Depth (km)",          "10.0",    0,   700),
            ("Year",                "2024",  2010, 2024),
            ("Month",               "6",       1,   12),
        ]

        for lbl, default, lo, hi in fields:
            row = tk.Frame(left, bg=COLOR_CARD)
            row.pack(fill="x", padx=20, pady=6)
            tk.Label(row, text=lbl, bg=COLOR_CARD, fg=COLOR_SUBTEXT,
                     font=FONT_BODY, width=16, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=float(default))
            # Entry box
            ent = tk.Entry(row, textvariable=var, bg="#0f1219", fg=COLOR_TEXT,
                           font=FONT_BODY, relief="flat", insertbackground="white", width=10)
            ent.pack(side="left", padx=4)
            # Slider
            sl = tk.Scale(row, from_=lo, to=hi, resolution=0.5 if '.' in default else 1,
                          orient="horizontal", variable=var,
                          bg=COLOR_CARD, fg=COLOR_TEXT, troughcolor="#2a2f45",
                          highlightthickness=0, showvalue=False, length=100,
                          command=lambda v, e=ent, vv=var: e.delete(0, 'end') or e.insert(0, f"{float(v):.1f}"))
            sl.pack(side="right")
            self._pred_entries[lbl] = var

        tk.Frame(left, bg="#2e3348", height=1).pack(fill="x", padx=20, pady=16)

        # Predict button
        tk.Button(left, text="⚡  Run Prediction", bg=COLOR_ACCENT, fg="white",
                  font=("Segoe UI", 12, "bold"), bd=0, pady=12,
                  cursor="hand2", command=self._run_prediction).pack(
            fill="x", padx=20, pady=(0, 20))

        # --- RIGHT: Result Panel ---
        self._result_panel = tk.Frame(content, bg=COLOR_BG)
        self._result_panel.pack(side="right", fill="both", expand=True)

        tk.Label(self._result_panel,
                 text="Configure inputs and click\n'Run Prediction'",
                 bg=COLOR_BG, fg=COLOR_SUBTEXT, font=FONT_H2).place(relx=0.5, rely=0.5, anchor="center")

    def _run_prediction(self):
        try:
            vals = {k: float(v.get()) for k, v in self._pred_entries.items()}
        except Exception:
            messagebox.showerror("Input Error", "All fields must be numeric.")
            return

        X = pd.DataFrame([{
            'depth':     vals['Depth (km)'],
            'latitude':  vals['Latitude'],
            'longitude': vals['Longitude'],
            'year':      vals['Year'],
            'month':     vals['Month'],
        }])

        proba = brain.model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred_label = brain.le.inverse_transform([pred_idx])[0]
        classes = list(brain.le.inverse_transform(range(len(proba))))
        confidence = round(float(max(proba)) * 100, 1)

        # Clear and rebuild result panel
        for w in self._result_panel.winfo_children():
            w.destroy()

        # Alert badge
        colors_map = {"Low": COLOR_SUCCESS, "Moderate": COLOR_WARNING, "High": COLOR_DANGER}
        alert_color = colors_map.get(pred_label, COLOR_ACCENT)

        tk.Label(self._result_panel, text="PREDICTED ALERT LEVEL",
                 bg=COLOR_BG, fg=COLOR_SUBTEXT, font=FONT_SMALL).pack(pady=(30, 4))
        tk.Label(self._result_panel, text=pred_label.upper(),
                 bg=COLOR_BG, fg=alert_color,
                 font=("Segoe UI", 52, "bold")).pack()
        tk.Label(self._result_panel, text=f"Confidence: {confidence}%",
                 bg=COLOR_BG, fg=COLOR_TEXT, font=FONT_H2).pack(pady=(4, 20))

        # Probability bar chart
        fig = plt.Figure(figsize=(6, 3), dpi=90)
        ax = fig.add_subplot(111)
        bar_colors = [colors_map.get(c, COLOR_ACCENT) for c in classes]
        bars = ax.bar(classes, [p * 100 for p in proba], color=bar_colors,
                      edgecolor=COLOR_BG, width=0.5)
        ax.bar_label(bars, fmt='%.1f%%', color=COLOR_TEXT, padding=4, fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_title("Class Probability Distribution", fontsize=11, pad=8)
        ax.set_ylabel("Probability (%)")
        ax.grid(axis='y', alpha=0.3)

        cv = FigureCanvasTkAgg(fig, master=self._result_panel)
        cv.draw()
        cv.get_tk_widget().pack(fill="x", padx=20)

        # Data table (Treeview widget)
        tk.Label(self._result_panel, text="Input Summary",
                 bg=COLOR_BG, fg=COLOR_SUBTEXT, font=FONT_SMALL).pack(pady=(16, 4))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Seismic.Treeview",
                         background=COLOR_CARD, foreground=COLOR_TEXT,
                         fieldbackground=COLOR_CARD, rowheight=26,
                         font=FONT_BODY)
        style.configure("Seismic.Treeview.Heading",
                         background="#2a2f45", foreground=COLOR_TEXT,
                         font=("Segoe UI", 10, "bold"))
        style.map("Seismic.Treeview", background=[('selected', COLOR_ACCENT)])

        tree = ttk.Treeview(self._result_panel, columns=("param", "value"),
                            show="headings", height=5, style="Seismic.Treeview")
        tree.heading("param", text="Parameter")
        tree.heading("value", text="Value")
        tree.column("param", width=160, anchor="w")
        tree.column("value", width=120, anchor="center")
        for k, v in vals.items():
            tree.insert("", "end", values=(k, f"{v:.1f}"))
        tree.pack(padx=20, pady=(0, 20))

    # ==================================================
    # PAGE 4 — LIVE USGS FEED
    # ==================================================
    def show_live_feed(self):
        self._clear()
        self.is_tracking = True
        self._page_header("Live USGS Earthquake Feed",
                          "Real-time data from earthquake.usgs.gov/fdsnws")

        # Controls row
        ctrl = tk.Frame(self.main, bg=COLOR_CARD, pady=10)
        ctrl.pack(fill="x", padx=24, pady=(0, 12))

        tk.Label(ctrl, text="Min Mag:", bg=COLOR_CARD, fg=COLOR_SUBTEXT,
                 font=FONT_BODY).pack(side="left", padx=(16, 4))
        self._live_mag = tk.StringVar(value="4.5")
        ttk.Combobox(ctrl, textvariable=self._live_mag,
                     values=["2.5", "4.5", "6.0"], width=6, state="readonly").pack(
            side="left", padx=4)

        tk.Label(ctrl, text="Period:", bg=COLOR_CARD, fg=COLOR_SUBTEXT,
                 font=FONT_BODY).pack(side="left", padx=(16, 4))
        self._live_period = tk.StringVar(value="week")
        ttk.Combobox(ctrl, textvariable=self._live_period,
                     values=["day", "week", "month"], width=8, state="readonly").pack(
            side="left", padx=4)

        tk.Button(ctrl, text="🔄  Refresh", bg=COLOR_ACCENT, fg="white",
                  font=("Segoe UI", 10, "bold"), bd=0, padx=12, pady=5,
                  cursor="hand2", command=self._fetch_live).pack(side="left", padx=16)

        # Status label
        self._live_status = tk.Label(ctrl, text="Press Refresh to load…",
                                     bg=COLOR_CARD, fg=COLOR_SUBTEXT, font=FONT_SMALL)
        self._live_status.pack(side="left")

        # Split: table left, map right
        split = tk.Frame(self.main, bg=COLOR_BG)
        split.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        # Treeview table
        table_frame = tk.Frame(split, bg=COLOR_CARD, width=520)
        table_frame.pack(side="left", fill="both", padx=(0, 10))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Live.Treeview",
                         background=COLOR_CARD, foreground=COLOR_TEXT,
                         fieldbackground=COLOR_CARD, rowheight=24, font=FONT_SMALL)
        style.configure("Live.Treeview.Heading",
                         background="#2a2f45", foreground=COLOR_TEXT,
                         font=("Segoe UI", 9, "bold"))
        style.map("Live.Treeview", background=[('selected', COLOR_ACCENT)])

        cols = ("time", "mag", "depth", "place")
        self._live_tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                        style="Live.Treeview")
        for c, w in zip(cols, (150, 60, 70, 230)):
            self._live_tree.heading(c, text=c.capitalize())
            self._live_tree.column(c, width=w, anchor="center" if c in ("mag","depth") else "w")

        sb = ttk.Scrollbar(table_frame, orient="vertical",
                           command=self._live_tree.yview)
        self._live_tree.configure(yscroll=sb.set)
        self._live_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Map
        try:
            import tkintermapview
            self._live_map = tkintermapview.TkinterMapView(split, corner_radius=0)
            self._live_map.pack(side="right", fill="both", expand=True)
            self._live_map.set_position(20, 0)
            self._live_map.set_zoom(2)
        except ImportError:
            tk.Label(split, text="Map requires:\npip install tkintermapview",
                     bg=COLOR_BG, fg=COLOR_SUBTEXT, font=FONT_BODY).pack(
                side="right", expand=True)
            self._live_map = None

    def _fetch_live(self):
        self._live_status.config(text="Fetching…", fg=COLOR_WARNING)
        threading.Thread(target=self._do_fetch, daemon=True).start()

    def _do_fetch(self):
        try:
            mag    = self._live_mag.get()
            period = self._live_period.get()
            url = (f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
                   f"{mag}_{period}.geojson")
            r = requests.get(url, timeout=10)
            features = r.json()['features']

            # Clear table
            for item in self._live_tree.get_children():
                self._live_tree.delete(item)

            if self._live_map:
                self._live_map.delete_all_marker()

            for f in features[:100]:
                props = f['properties']
                coords = f['geometry']['coordinates']
                t = datetime.utcfromtimestamp(props['time'] / 1000).strftime("%Y-%m-%d %H:%M")
                mag_v = props.get('mag', '?')
                dep   = round(coords[2], 1)
                place = (props.get('place') or 'Unknown')[:40]
                self._live_tree.insert("", "end", values=(t, mag_v, dep, place))

                if self._live_map:
                    lat, lon = coords[1], coords[0]
                    c = "red" if (mag_v or 0) >= 6 else ("orange" if (mag_v or 0) >= 4 else "blue")
                    try:
                        self._live_map.set_marker(lat, lon,
                                                   text=f"M{mag_v}",
                                                   marker_color_circle=c,
                                                   marker_color_outside=c)
                    except Exception:
                        pass

            self._live_status.config(
                text=f"Loaded {len(features)} events · {datetime.utcnow().strftime('%H:%M UTC')}",
                fg=COLOR_SUCCESS)
        except Exception as e:
            self._live_status.config(text=f"Error: {e}", fg=COLOR_DANGER)


# ==================================================
# ENTRY POINT
# ==================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SeismicApp(root)
    root.mainloop()