# 🌍 QuakeView

### *An Interactive Earthquake Data Analysis & Visualization System*

---

## 📌 Overview

**QuakeView** is a Python-based desktop application that analyzes and visualizes global earthquake data using an interactive dashboard.
It combines **data analysis**, **visualization**, and **machine learning** to uncover patterns in seismic activity.

The project uses real-world earthquake data from the **USGS (United States Geological Survey)** and presents insights through an intuitive Tkinter-based interface.

---

## 🎯 Features

### 📊 Data Analysis

* Year-wise earthquake trends
* Monthly distribution analysis
* Magnitude and depth categorization
* Top earthquake-prone regions

### 📈 Visualizations

* Bar charts (yearly trends)
* Histograms (magnitude distribution)
* Scatter plots (depth vs magnitude)
* Heatmaps (year vs month activity)
* Region-based analysis

### 🤖 Machine Learning

* **K-Means Clustering** to identify seismic zones
* Groups earthquakes based on:

  * Latitude
  * Longitude
  * Depth

### 🖥️ Interactive Dashboard

* Built using **Tkinter**
* Dynamic filtering:

  * Year selection
  * Magnitude range sliders
* Real-time chart updates

---

## 🧠 Key Insights

* Earthquakes are **not randomly distributed**
* Most occur along **tectonic plate boundaries**
* **Shallow earthquakes** are more dangerous than deep ones
* High-magnitude earthquakes are **rare but impactful**
* Clustering reveals **distinct seismic zones**

---

## 🗂️ Project Structure

```
quakeview/
│
├── main.py        # Tkinter dashboard (UI)
├── analysis.py    # Data cleaning & analysis
├── plots.py       # Visualization functions
├── earthquake.csv # Dataset (USGS data)
└── README.md
```

---

## ⚙️ Technologies Used

* **Python**
* **Pandas** – Data processing
* **NumPy** – Numerical operations
* **Matplotlib / Seaborn** – Visualization
* **Scikit-learn** – Machine Learning (K-Means)
* **Tkinter** – GUI development

---

## 🚀 How to Run

1. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Ensure `earthquake.csv` is in the same folder

3. Run the application:

```bash
python main.py
```

---

## 📊 Dataset

* Source: **USGS Earthquake Catalog**
* Contains:

  * Time
  * Latitude & Longitude
  * Depth
  * Magnitude
  * Location

---

## ⚠️ Note on Machine Learning

This project uses machine learning for **pattern discovery**, not prediction.
Due to the complex nature of seismic activity, exact prediction of earthquakes is not feasible.

Clustering is used to:

* Identify seismic zones
* Understand spatial patterns

---

## 🏆 Project Highlights

* Real-world scientific dataset
* Clean and structured data pipeline
* Interactive dashboard with filters
* Meaningful ML integration (not forced)
* Strong visual storytelling

---

## 📌 Future Improvements

* Map-based visualization (interactive world map)
* Real-time data integration via API
* Advanced clustering techniques
* Exportable reports

---

## 👨‍💻 Authors

* **Vibhav Kumar Misra** — https://github.com/vibhav7032
* **Himanshu Rawat** — https://github.com/himansyou

---

## ⭐ Acknowledgment

Data provided by **USGS (United States Geological Survey)**

---

> *QuakeView helps transform raw seismic data into meaningful insights through visualization and machine learning.*
