# main.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import analysis
import plots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# -- Colour tokens (dark theme) ------------------------------------------------
BG_DARK = "#0d1117"
BG_MID = "#161b22"
BG_PANEL = "#1c2128"
ACCENT = "#f97316"
ACCENT2 = "#38bdf8"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
BORDER = "#30363d"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"


class EarthquakeDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌍  QuakeView  |  Earthquake Analysis Dashboard")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self.configure(bg=BG_DARK)

        self.df_full = None
        self.df_filtered = None

        self._apply_ttk_theme()
        self._build_ui()
        self._start_loading()

    def _apply_ttk_theme(self):
        style = ttk.Style(self)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        forest_tcl = os.path.join(base_dir, "forest-dark.tcl")
        forest_assets = os.path.join(base_dir, "forest-dark")
        awdark_tcl = os.path.join(base_dir, "awdark.tcl")
        fallback_clam = False

        try:
            has_forest_assets = os.path.isdir(forest_assets) and any(
                name.endswith(".png") for name in os.listdir(forest_assets)
            )
            if has_forest_assets:
                self.tk.call("source", forest_tcl)
                style.theme_use("forest-dark")
            else:
                self.tk.call("source", awdark_tcl)
                style.theme_use("awdark")
        except Exception as e:
            print(f"Theme load fallback to clam: {e}")
            style.theme_use("clam")
            fallback_clam = True

        if not fallback_clam:
            return

        style.configure(
            ".",
            background=BG_DARK,
            foreground=TEXT,
            fieldbackground=BG_MID,
            bordercolor=BORDER,
            troughcolor=BG_MID,
            selectbackground=ACCENT,
            selectforeground="white",
            font=("Segoe UI", 10),
        )

        style.configure("TNotebook", background=BG_DARK, borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure(
            "TNotebook.Tab",
            background=BG_MID,
            foreground=TEXT_DIM,
            padding=[14, 6],
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
        )
        style.map("TNotebook.Tab", background=[("selected", BG_PANEL)], foreground=[("selected", ACCENT)])

        style.configure("TFrame", background=BG_DARK)
        style.configure("TLabel", background=BG_DARK, foreground=TEXT)
        style.configure(
            "TButton",
            background=ACCENT,
            foreground="white",
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
            padding=6,
        )
        style.map("TButton", background=[("active", "#ea6c10")])

        style.configure(
            "Outline.TButton",
            background=BG_MID,
            foreground=ACCENT,
            font=("Segoe UI", 9, "bold"),
            borderwidth=1,
            relief="solid",
            padding=6,
        )
        style.map("Outline.TButton", background=[("active", BG_PANEL)])

        style.configure(
            "TCombobox",
            background=BG_MID,
            foreground=TEXT,
            selectbackground=BG_MID,
            selectforeground=TEXT,
            arrowcolor=ACCENT,
            borderwidth=1,
        )

        style.configure("Horizontal.TScale", background=BG_DARK, troughcolor=BG_MID, sliderthickness=16, sliderrelief="flat")
        style.configure("TCheckbutton", background=BG_DARK, foreground=TEXT, focuscolor=BG_DARK)
        style.configure("TProgressbar", troughcolor=BG_MID, background=ACCENT, borderwidth=0, thickness=6)

        style.configure(
            "Treeview",
            background=BG_MID,
            foreground=TEXT,
            fieldbackground=BG_MID,
            borderwidth=0,
            rowheight=26,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=BG_PANEL,
            foreground=ACCENT,
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
        )
        style.map("Treeview", background=[("selected", ACCENT)])

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG_MID, height=52)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(
            hdr,
            text="🌍 QUAKEVIEW | Earthquake Analytics Dashboard",
            bg=BG_MID,
            fg=ACCENT,
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left", padx=18, pady=10)

        tk.Label(
            hdr,
            text="USGS Global Dataset  ·  2015 – 2026",
            bg=BG_MID,
            fg=TEXT_DIM,
            font=("Segoe UI", 9),
        ).pack(side="left", padx=4, pady=10)

        self.status_label = tk.Label(
            hdr, text="⏳  Loading data…", bg=BG_MID, fg=YELLOW, font=("Segoe UI", 9)
        )
        self.status_label.pack(side="right", padx=18)

        self.progress = ttk.Progressbar(self, mode="indeterminate", style="TProgressbar", length=200)
        self.progress.pack(fill="x", padx=0, pady=0)

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)

        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_yearly = ttk.Frame(self.notebook)
        self.tab_dist = ttk.Frame(self.notebook)
        self.tab_scatter = ttk.Frame(self.notebook)
        self.tab_heatmap = ttk.Frame(self.notebook)
        self.tab_monthly = ttk.Frame(self.notebook)
        self.tab_regions = ttk.Frame(self.notebook)
        self.tab_table = ttk.Frame(self.notebook)
        self.tab_clusters = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_overview, text="  📋 Overview  ")
        self.notebook.add(self.tab_yearly, text="  📊 Yearly Trend  ")
        self.notebook.add(self.tab_dist, text="  🔢 Magnitude Dist  ")
        self.notebook.add(self.tab_scatter, text="  🔵 Depth vs Mag  ")
        self.notebook.add(self.tab_heatmap, text="  🗓 Heatmap  ")
        self.notebook.add(self.tab_monthly, text="  📈 Monthly  ")
        self.notebook.add(self.tab_regions, text="  🗺 Top Regions  ")
        self.notebook.add(self.tab_table, text="  🗃 Data Table  ")
        self.notebook.add(self.tab_clusters, text="  🧩 Clusters  ")

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG_MID, width=230)
        sb.pack(side="left", fill="y", padx=(10, 0), pady=10)
        sb.pack_propagate(False)

        def section(text):
            tk.Label(sb, text=text, bg=BG_MID, fg=ACCENT, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=14, pady=(14, 2))
            ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=10)

        section("SUMMARY")
        self.kpi_frame = tk.Frame(sb, bg=BG_MID)
        self.kpi_frame.pack(fill="x", padx=10, pady=6)

        self.kpi_vars = {}
        kpis = [
            ("Total Events", "total_earthquakes", GREEN),
            ("Avg Magnitude", "avg_magnitude", ACCENT2),
            ("Max Magnitude", "max_magnitude", RED),
            ("Major (M≥7)", "major_quakes", YELLOW),
            ("Avg Depth (km)", "avg_depth_km", ACCENT),
        ]
        for label, key, color in kpis:
            card = tk.Frame(self.kpi_frame, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
            card.pack(fill="x", pady=3)
            tk.Label(card, text=label, bg=BG_PANEL, fg=TEXT_DIM, font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=8, pady=(5, 0))
            var = tk.StringVar(value="—")
            self.kpi_vars[key] = var
            tk.Label(card, textvariable=var, bg=BG_PANEL, fg=color, font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=8, pady=(0, 5))

        section("FILTERS")
        tk.Label(sb, text="Year", bg=BG_MID, fg=TEXT_DIM, font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(8, 2))
        self.year_var = tk.StringVar(value="All")
        self.year_combo = ttk.Combobox(sb, textvariable=self.year_var, state="readonly", width=20)
        self.year_combo["values"] = ["All"]
        self.year_combo.pack(padx=14, pady=2)
        self.year_combo.bind("<<ComboboxSelected>>", self._on_year_change)

        tk.Label(sb, text="Min Magnitude", bg=BG_MID, fg=TEXT_DIM, font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(10, 0))
        self.min_mag_var = tk.DoubleVar(value=0.0)
        self.min_mag_label = tk.Label(sb, text="0.0", bg=BG_MID, fg=ACCENT, font=("Segoe UI", 9, "bold"))
        self.min_mag_label.pack(anchor="w", padx=14)
        min_scale = ttk.Scale(sb, from_=0, to=9, variable=self.min_mag_var, orient="horizontal", command=self._on_min_mag_change)
        min_scale.pack(fill="x", padx=14, pady=2)

        tk.Label(sb, text="Max Magnitude", bg=BG_MID, fg=TEXT_DIM, font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(8, 0))
        self.max_mag_var = tk.DoubleVar(value=10.0)
        self.max_mag_label = tk.Label(sb, text="10.0", bg=BG_MID, fg=ACCENT, font=("Segoe UI", 9, "bold"))
        self.max_mag_label.pack(anchor="w", padx=14)
        max_scale = ttk.Scale(sb, from_=0, to=10, variable=self.max_mag_var, orient="horizontal", command=self._on_max_mag_change)
        max_scale.pack(fill="x", padx=14, pady=2)

        section("OPTIONS")
        self.show_trend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sb, text=" Show trend line", variable=self.show_trend_var).pack(anchor="w", padx=14, pady=4)
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sb, text=" Show grid lines", variable=self.show_grid_var).pack(anchor="w", padx=14, pady=2)

        section("ACTIONS")
        ttk.Button(sb, text="▶  Apply Filters", command=self._apply_filters).pack(fill="x", padx=14, pady=(8, 4))
        ttk.Button(sb, text="↺  Reset Filters", style="Outline.TButton", command=self._reset_filters).pack(fill="x", padx=14, pady=4)
        ttk.Button(sb, text="⟳  Reload Data", style="Outline.TButton", command=self._start_loading).pack(fill="x", padx=14, pady=4)

        tk.Label(sb, text="Data: USGS Earthquake Catalog", bg=BG_MID, fg=TEXT_DIM, font=("Segoe UI", 7)).pack(side="bottom", pady=8)

    def _start_loading(self):
        self.progress.start(12)
        self.status_label.config(text="⏳  Loading data…", fg=YELLOW)
        t = threading.Thread(target=self._load_data, daemon=True)
        t.start()

    def _load_data(self):
        try:
            df = analysis.load_and_clean("earthquake.csv")
            self.after(0, self._on_data_loaded, df)
        except Exception as e:
            self.after(0, self._on_load_error, str(e))

    def _on_load_error(self, msg):
        self.progress.stop()
        self.progress.pack_forget()
        messagebox.showerror(
            "Load Error",
            f"Could not load earthquake.csv:\n\n{msg}\n\nMake sure earthquake.csv is in the same folder as main.py.",
        )
        self.status_label.config(text="❌  Load failed", fg=RED)

    def _on_data_loaded(self, df):
        self.df_full = df
        self.df_filtered = df.copy()
        self.progress.stop()
        self.progress.pack_forget()

        stats = analysis.get_summary_stats(df)
        for key, var in self.kpi_vars.items():
            var.set(str(stats.get(key, "—")))

        years = ["All"] + [str(y) for y in stats["years_covered"]]
        self.year_combo["values"] = years
        self.status_label.config(text=f"✅  {stats['total_earthquakes']} records loaded", fg=GREEN)
        self._render_all_tabs(df)

    def _render_all_tabs(self, df):
        self._render_overview(df)
        self._render_chart(self.tab_yearly, plots.plot_yearly_counts, analysis.get_yearly_counts(df))
        self._render_chart(self.tab_dist, plots.plot_magnitude_distribution, analysis.get_magnitude_distribution(df))
        self._render_chart(self.tab_scatter, plots.plot_depth_vs_magnitude, analysis.get_depth_vs_mag(df))
        self._render_chart(self.tab_heatmap, plots.plot_monthly_heatmap, analysis.get_heatmap_data(df))
        self._render_chart(self.tab_monthly, plots.plot_monthly_trend, analysis.get_monthly_counts(df))
        self._render_chart(self.tab_regions, plots.plot_top_regions, analysis.get_top_regions(df))
        clustered_df = analysis.get_clusters(df, n_clusters=4)
        self._render_chart(self.tab_clusters, plots.plot_clusters, clustered_df)
        self._render_table(df)

    def _clear_tab(self, tab):
        for w in tab.winfo_children():
            w.destroy()

    def _render_chart(self, tab, plot_fn, data):
        self._clear_tab(tab)
        try:
            fig = plot_fn(data)
        except Exception as e:
            tk.Label(tab, text=f"Chart error:\n{e}", bg=BG_DARK, fg=RED, font=("Segoe UI", 10), wraplength=600).pack(expand=True)
            return

        frame = tk.Frame(tab, bg=BG_DARK)
        frame.pack(fill="both", expand=True)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar_frame = tk.Frame(frame, bg="#161b22")
        toolbar_frame.pack(side="bottom", fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.config(background="#161b22")
        for btn in toolbar.winfo_children():
            try:
                btn.config(background="#161b22", foreground=TEXT)
            except Exception:
                pass
        toolbar.update()

        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas.draw()

    def _render_overview(self, df):
        self._clear_tab(self.tab_overview)
        stats = analysis.get_summary_stats(df)

        grid = tk.Frame(self.tab_overview, bg=BG_DARK)
        grid.pack(fill="x", padx=20, pady=(16, 8))

        kpi_defs = [
            ("🌐 Total Earthquakes", stats["total_earthquakes"], GREEN),
            ("📏 Avg Magnitude", stats["avg_magnitude"], ACCENT2),
            ("🔺 Max Magnitude", stats["max_magnitude"], RED),
            ("💡 Min Magnitude", stats["min_magnitude"], TEXT_DIM),
            ("🏔 Avg Depth", f"{stats['avg_depth_km']} km", ACCENT),
            ("📅 Most Active Year", stats["most_active_year"], YELLOW),
            ("⚠️ Major Events (M≥7)", stats["major_quakes"], RED),
            ("📆 Years Covered", len(stats["years_covered"]), ACCENT2),
        ]

        for i, (label, value, color) in enumerate(kpi_defs):
            col = i % 4
            row = i // 4
            card = tk.Frame(grid, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            grid.columnconfigure(col, weight=1)
            tk.Label(card, text=label, bg=BG_PANEL, fg=TEXT_DIM, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
            tk.Label(card, text=str(value), bg=BG_PANEL, fg=color, font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=12, pady=(0, 10))

        section_lbl = tk.Label(self.tab_overview, text="Magnitude Category Breakdown", bg=BG_DARK, fg=ACCENT, font=("Segoe UI", 11, "bold"))
        section_lbl.pack(anchor="w", padx=28, pady=(10, 4))

        cat_df = analysis.get_mag_category_counts(df)
        cols = ("Category", "Count", "% of Total")
        tree = ttk.Treeview(self.tab_overview, columns=cols, show="headings", height=6)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=200, anchor="center")

        total = cat_df["count"].sum()
        for _, row in cat_df.iterrows():
            pct = f"{row['count'] / total * 100:.1f}%"
            tree.insert("", "end", values=(row["category"], f"{int(row['count']):,}", pct))
        tree.pack(fill="x", padx=28, pady=4)

        depth_lbl = tk.Label(self.tab_overview, text="Depth Category Breakdown", bg=BG_DARK, fg=ACCENT, font=("Segoe UI", 11, "bold"))
        depth_lbl.pack(anchor="w", padx=28, pady=(14, 4))

        depth_df = analysis.get_depth_category_counts(df)
        tree2 = ttk.Treeview(self.tab_overview, columns=cols, show="headings", height=4)
        for col in cols:
            tree2.heading(col, text=col)
            tree2.column(col, width=200, anchor="center")

        for _, row in depth_df.iterrows():
            pct = f"{row['count'] / total * 100:.1f}%"
            tree2.insert("", "end", values=(row["category"], f"{int(row['count']):,}", pct))
        tree2.pack(fill="x", padx=28, pady=4)

    def _render_table(self, df):
        self._clear_tab(self.tab_table)

        top = tk.Frame(self.tab_table, bg=BG_DARK)
        top.pack(fill="x", padx=12, pady=8)
        tk.Label(top, text=f"Showing top 500 rows of {len(df):,}", bg=BG_DARK, fg=TEXT_DIM, font=("Segoe UI", 9)).pack(side="left")

        display_cols = ["time", "place", "mag", "mag_category", "depth", "depth_category", "latitude", "longitude"]
        display_df = df[display_cols].head(500)

        container = tk.Frame(self.tab_table, bg=BG_DARK)
        container.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        tree = ttk.Treeview(container, columns=display_cols, show="headings")
        widths = [165, 240, 60, 120, 70, 170, 80, 80]
        for col, w in zip(display_cols, widths):
            tree.heading(col, text=col.replace("_", " ").title())
            tree.column(col, width=w, anchor="center")

        tree.tag_configure("odd", background=BG_MID)
        tree.tag_configure("even", background=BG_PANEL)

        for i, (_, row) in enumerate(display_df.iterrows()):
            tag = "odd" if i % 2 else "even"
            t_str = str(row["time"])[:19]
            tree.insert(
                "",
                "end",
                tag=tag,
                values=(
                    t_str,
                    row["place"],
                    row["mag"],
                    row["mag_category"],
                    row["depth"],
                    row["depth_category"],
                    round(row["latitude"], 3),
                    round(row["longitude"], 3),
                ),
            )

        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(side="left", fill="both", expand=True)

    def _on_year_change(self, _event=None):
        if self.df_full is not None:
            self._apply_filters()

    def _on_min_mag_change(self, val):
        min_val = float(val)
        max_val = float(self.max_mag_var.get())
        if min_val > max_val:
            self.max_mag_var.set(min_val)
            self.max_mag_label.config(text=f"{min_val:.1f}")
        self.min_mag_label.config(text=f"{min_val:.1f}")
        if self.df_full is not None:
            self._apply_filters()

    def _on_max_mag_change(self, val):
        max_val = float(val)
        min_val = float(self.min_mag_var.get())
        if max_val < min_val:
            self.min_mag_var.set(max_val)
            self.min_mag_label.config(text=f"{max_val:.1f}")
        self.max_mag_label.config(text=f"{max_val:.1f}")
        if self.df_full is not None:
            self._apply_filters()

    def _apply_filters(self):
        if self.df_full is None:
            return
        self.status_label.config(text="⏳  Filtering…", fg=YELLOW)
        self.update_idletasks()

        df = analysis.filter_df(
            self.df_full,
            year=self.year_var.get(),
            min_mag=round(self.min_mag_var.get(), 1),
            max_mag=round(self.max_mag_var.get(), 1),
        )
        self.df_filtered = df

        if df.empty:
            for _, var in self.kpi_vars.items():
                var.set("—")
            self.status_label.config(text="⚠️  No records for selected filters", fg=YELLOW)
            for tab in [
                self.tab_overview,
                self.tab_yearly,
                self.tab_dist,
                self.tab_scatter,
                self.tab_heatmap,
                self.tab_monthly,
                self.tab_regions,
                self.tab_table,
                self.tab_clusters,
            ]:
                self._clear_tab(tab)
                tk.Label(tab, text="No data for selected filters.", bg=BG_DARK, fg=TEXT_DIM, font=("Segoe UI", 11)).pack(expand=True)
            return

        stats = analysis.get_summary_stats(df)
        for key, var in self.kpi_vars.items():
            var.set(str(stats.get(key, "—")))

        self._render_all_tabs(df)
        self.status_label.config(text=f"✅  {len(df):,} records (filtered)", fg=GREEN)

    def _reset_filters(self):
        self.year_var.set("All")
        self.min_mag_var.set(0.0)
        self.max_mag_var.set(10.0)
        self.min_mag_label.config(text="0.0")
        self.max_mag_label.config(text="10.0")
        if self.df_full is not None:
            self._apply_filters()


if __name__ == "__main__":
    app = EarthquakeDashboard()
    app.mainloop()
