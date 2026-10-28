# WaterWise: Large-Scale Water Consumption Pattern Analysis and Anomaly Detection Using PySpark

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 📌 Executive Overview
**WaterWise** is an end-to-end Big Data Analytics mini-project engineered to process, aggregate, and detect behavioral anomalies across **652,315 real household water-meter telemetry events** collected from the **HSB Living Lab** (Chalmers University of Technology, Gothenburg, Sweden).

The project implements a distributed data processing engine using **Apache PySpark (local mode)** and **Spark SQL**, exports partitioned columnar **Parquet** representations, performs **fixture-stratified statistical anomaly detection (Interquartile Range - IQR)**, and delivers insights through an interactive **Streamlit** dashboard styled with a custom navy/teal hydrodynamic theme.

---

## 🏛️ System Architecture

```
+-----------------------------------------------------------------------------------------+
|                                1. DATA SOURCE & INGESTION                               |
|   HSB Living Lab Telemetry CSV: 652,315 records, 214 pulse-counting smart meters (1 L) |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                             2. PYSPARK VALIDATION & CLEANING                            |
|   * Strict UTC Timestamp validation & casting to TimestampType                          |
|   * Zero & negative value rejection (value > 0)                                         |
|   * Physical telemetry threshold gating (value <= 0.3 m3 / 300 L per 10-min interval)    |
|   * Intelligent Composite Deduplication: True duplicate = (timestamp, sensor_id)       |
|   * Shared Facility Imputation: Missing apartment/cluster mapped to 'Shared Facility'   |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                        3. FEATURE ENGINEERING & PARQUET PERSISTENCE                     |
|   * Derived Features: hour (0-23), day_of_week, day_name, month, year, is_weekend      |
|   * Dual Units: volume_m3 (cubic meters) and volume_liters (m3 * 1,000.0)              |
|   * Partitioned Snappy Parquet export for fast analytical queries                      |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                      4. DISTRIBUTED SPARK SQL ANALYTICS & IQR ENGINE                    |
|   * Diurnal 24-hr load curves, weekly patterns, fixture shares, top residential units   |
|   * Group-Stratified IQR: Q1 (25%), Q3 (75%), IQR = Q3 - Q1, Cutoff = Q3 + (1.5 * IQR)  |
|   * Zero-IQR Fallback handling: Q3 + (2.5 * stddev) for discrete pulse meters           |
|   * Distributed Inner Join: Flags anomalies with volume, threshold, excess & ratio      |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                           5. PRESENTATION & FORECASTING LAYER                           |
|   * Streamlit Interactive UI with Navy/Teal Water Theme & Responsive Plotly Visuals     |
|   * Optional Enhancement: Chronological Ridge Regressor for daily demand forecasting    |
|   * Out-of-sample evaluation: MAE & RMSE tracking without data leakage                  |
|   * Interactive College Viva Technical Defense Guide built directly into the UI         |
+-----------------------------------------------------------------------------------------+
```

---

## 📂 Repository Structure

```
WaterWise/
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # Python dependencies (PySpark, Streamlit, Plotly, etc.)
├── README.md                   # Complete documentation and viva defense guide
├── hadoop_home/                # Local Windows Hadoop binaries (winutils.exe, hadoop.dll)
│   └── bin/
├── data/                       # Dataset directory
│   ├── HSB_Living_Lab_Water_Consumption_Anonymized.csv  # 85MB real CSV dataset
│   ├── DATA_DICTIONARY.txt     # Data dictionary from study authors
│   ├── README.txt              # Dataset publication notes
│   ├── README.md               # Dataset documentation and schema mapping
│   └── scripts/                # Original cleaning & figure generation scripts
├── src/                        # Modular source code
│   ├── __init__.py             # Package initializer
│   ├── spark_session.py        # Centralized SparkSession & environment configuration
│   ├── data_loader.py          # Real CSV loading, existence checks, missing stats
│   ├── preprocessing.py        # PySpark cleaning pipeline & Parquet storage
│   ├── analytics.py            # Spark SQL aggregations & KPI queries
│   ├── anomaly_detection.py    # Group-stratified IQR anomaly detection
│   └── forecasting.py          # Chronological daily demand forecasting
├── tests/                      # Automated test suite
│   ├── test_pipeline.py        # End-to-end pipeline verification on real dataset
│   └── test_edge_cases.py      # Unit & edge case tests (missing files, zero IQR, etc.)
└── outputs/                    # Output directory
    ├── README.md               # Output artifacts documentation
    └── parquet/                # Parquet storage for cleaned dataset
```

---

## 🛠️ Technology Stack & Prerequisites

* **Operating System**: Windows 10/11, macOS, or Linux.
* **Java Runtime**: Java 17 or 21 (Eclipse Temurin JDK 17 is pre-configured and verified).
* **Python**: Python 3.11, 3.12, or 3.13.
* **Apache PySpark**: 4.2.0 (Local cluster execution mode).
* **Streamlit**: 1.30+ (Web presentation layer).
* **Plotly**: 5.18+ (Interactive visual telemetry).
* **Pandas & NumPy**: For localized small result-set chart conversions.
* **Scikit-Learn**: For the optional chronological forecasting module.

---

## 🚀 Setup & Execution Guide

### 1. Clone or Open Workspace
```powershell
cd "c:\Users\Jason Dsouza\Desktop\bda project"
```

### 2. Install Required Python Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Verify Dataset Placement
Ensure that `HSB_Living_Lab_Water_Consumption_Anonymized.csv` is present in `data/`:
```powershell
ls data/HSB_Living_Lab_Water_Consumption_Anonymized.csv
```
*Note*: If the file is missing, download it from Zenodo:
[https://zenodo.org/records/18971107](https://zenodo.org/records/18971107) (File: `HSB_Living_Lab_Data_Package.zip`) and extract it into `data/`. The application will explicitly raise an error rather than fabricating mock data.

### 4. Run Automated Verification Suite
To verify that PySpark, schema extraction, IQR calculations, and forecasting pass on your local machine:
```powershell
python tests/test_pipeline.py
python -m pytest tests/test_edge_cases.py -v
```

### 5. Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🔬 Dataset Schema & Internal Mapping

The raw dataset uses semicolon (`;`) delimiters.

| CSV Column | Raw Format | Spark DataType | Application Target | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | `TimestampType` | `timestamp` | UTC 10-minute rounded meter reading |
| `date` | `YYYY-MM-DD` | `DateType` | `date` | Calendar date |
| `hour` | `0` to `23` | `IntegerType` | `hour` | Diurnal hour of day |
| `day_of_week` | `0` to `6` | `IntegerType` | `day_of_week` | Day index (1=Sun, 2=Mon...7=Sat) |
| `day_name` | String | `StringType` | `day_name` | Name of day (Monday through Sunday) |
| `sensor_id` | String (`S0001`...) | `StringType` | `sensor_id` | Unique pulse water meter identifier |
| `apartment` | String (`APT_001`...) | `StringType` | `apartment` | Residential unit (Imputed to `'Shared Facility'` if blank) |
| `cluster_name`| String (`CL_A`...`CL_I`) | `StringType` | `cluster_name` | Building cluster identifier |
| `room_type` | String | `StringType` | `room_type` | Bathroom, Kitchen, Living room, Shared laundry |
| `type` | String | `StringType` | `water_type` | `cold water consumption` or `hot water consumption` |
| `attached_to` | String | `StringType` | `fixture` | Shower, Toilet, Kitchen tap, Washbasin, Washing machine |
| `value` | Float (m³) | `DoubleType` | `volume_m3` | Volume in m³; derived: `volume_liters = value * 1,000.0` |
| `aggregated_value` | Float (m³) | `DoubleType` | `cumulative_m3`| Cumulative index reading on meter |

---

## 💡 Statistical Anomaly Detection Methodology

### Why Stratify by Fixture?
Water end-use fixtures have inherently distinct hydraulic behaviors:
* **Toilet**: Discrete flushes typically draw **3 to 9 Liters**.
* **Washbasin**: Handwashing events typically draw **1 to 5 Liters**.
* **Shower**: Standard showers typically consume **30 to 80 Liters**.

Applying a single global threshold across the entire building would lead to severe statistical flaws:
1. High False Positives: Normal showers would be misidentified as leaks.
2. High False Negatives: Massive running washbasin taps (e.g. 25 Liters) would go unnoticed because they are smaller than a normal shower.

### Mathematical Formulation
For each fixture category $f \in \{\text{Toilet}, \text{Kitchen tap}, \text{Washbasin}, \text{Shower}, \text{Washing machine}\}$:
$$\text{IQR}_f = Q_{3, f} - Q_{1, f}$$
$$\text{Threshold}_f = Q_{3, f} + 1.5 \times \text{IQR}_f$$

* **Zero-IQR Robustness**: When pulse meters produce identical discrete readings causing $Q_1 = Q_3$ ($\text{IQR} = 0$), the pipeline automatically falls back to:
$$\text{Threshold}_f = Q_{3, f} + 2.5 \times \sigma_f$$

### Interpretation Disclaimer
Readings flagged as anomalies represent **statistical outliers** exceeding the group threshold. They **do not prove a water pipe burst or mechanical leak**. Real-world causes include extended showers, filling large water containers or aquariums, intensive deep cleaning, or concurrent fixture usage.

---

## 🔮 Forecasting Enhancement (Decoupled Module)

* **Objective**: Predict aggregate daily water demand (Liters) for proactive municipal or building reservoir planning.
* **Validation Strategy**: Strict **chronological train/test split (80% train / 20% test)** to guarantee zero future data leakage.
* **Features**: Lag 1 ($y_{t-1}$), Lag 7 ($y_{t-7}$), 7-day moving average, day of week, and month.
* **Evaluation Metrics**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) reported on unseen test data.
* **Decoupling**: Built in `src/forecasting.py`, entirely independent of the core PySpark pipeline.

---

## 🎓 College Viva Presentation & Technical Defense Guide

### 5-Minute Demonstration Walkthrough for Examiners
1. **Show the Problem & Dataset (1 min)**:
   - Introduce the HSB Living Lab dataset: 652,315 real readings across 28 units, 9 clusters, and 214 smart pulse meters over 3 years.
   - Point out that city/building water telemetry cannot be processed in single-threaded Excel or basic Pandas without running out of memory.
2. **Show the PySpark Cleaning Pipeline (1 min)**:
   - Navigate to the **Data Quality & Pipeline** tab.
   - Show how raw records are ingested, invalid timestamps are filtered out, zero/negative readings are dropped, and true duplicate readings are removed using `(timestamp, sensor_id)`.
3. **Show Consumption Dynamics (1 min)**:
   - Navigate to **Executive Summary** and **Consumption Trends**.
   - Show the 24-hour diurnal profile (morning and evening hygiene/cooking peaks).
   - Show the weekly pattern: weekend consumption peaks due to higher home occupancy.
4. **Demonstrate Stratified IQR Anomaly Detection (1 min)**:
   - Navigate to **Anomaly Detection (IQR)**.
   - Explain why thresholds are calculated per fixture rather than globally.
   - Adjust the **Sensitivity Slider** (1.0x to 3.0x IQR) in the sidebar to show dynamic recalculation in real-time.
   - Export the anomaly CSV.
5. **Show Forecasting & Viva Technical Defense (1 min)**:
   - Navigate to **Forecasting (Enhancement)** and show the chronological train/test split.
   - Navigate to **College Viva & Architecture** to explain lazy evaluation and Catalyst optimizations.

---

### Key Viva Questions & Accurate Technical Answers

#### Q1: Why did you choose Apache PySpark instead of pure Pandas for this project?
> **Answer**: The HSB Living Lab dataset contains over 652,000 high-frequency sensor readings spanning 3 years. Pure Pandas loads the entire dataset eagerly into single-threaded RAM. If this application were scaled to an entire city or multi-building municipal network with millions of readings daily, Pandas would crash with out-of-memory (OOM) errors. PySpark distributes data across partitions, uses lazy evaluation with Catalyst query optimization, and manages memory spillover to disk. We use Pandas only for rendering small, aggregated chart data in Plotly.

#### Q2: What is Spark's Lazy Evaluation, and where is it visible in your code?
> **Answer**: In PySpark, transformations (such as `.filter()`, `.withColumn()`, `.dropDuplicates()`) do not execute immediately when declared. Instead, Spark constructs a Directed Acyclic Graph (DAG) of logical operations. Physical computation only begins when an **Action** is invoked (such as `.count()`, `.collect()`, or `.write.parquet()`). In `src/preprocessing.py`, all parsing and filtering stages are composed lazily so Catalyst can optimize filter pushdown before triggering physical execution.

#### Q3: Why is grouping by fixture essential for IQR anomaly detection in water analytics?
> **Answer**: Water fixtures have fundamentally different hydrodynamic capacities. A standard dual-flush toilet draws 3 to 6 Liters, while a 10-minute shower consumes 40 to 80 Liters. If we computed a single threshold across all fixtures, normal showers would be falsely flagged as pipe bursts (high false-positive rate), while a toilet running continuously at 20 Liters would go undetected (high false-negative rate). Computing IQR per fixture category ensures explainable, fair statistical thresholds.

#### Q4: How did you distinguish between true duplicates and concurrent legitimate readings?
> **Answer**: In smart building telemetry, dozens of meters report at the exact same 10-minute interval (e.g. Unit 1 toilet and Unit 2 kitchen tap both reporting at 08:00:00). Blindly dropping rows with matching timestamps would delete genuine consumption events. We defined a composite primary key: `(timestamp, sensor_id)`. Rows sharing both the exact timestamp AND the exact physical meter ID are true telemetry duplicates and were dropped.

#### Q5: Why store intermediate outputs in Parquet format instead of CSV?
> **Answer**: Parquet is a columnar storage format with Snappy compression and embedded dictionary encoding. When analytical queries request aggregations on `volume_liters` and `fixture`, Spark reads only those specific columns from disk rather than parsing entire text rows as in CSV. This reduces disk I/O by over 70% and accelerates downstream analytical queries by 5x to 10x.

#### Q6: How did you avoid data leakage in the time-series forecasting model?
> **Answer**: We applied a strictly chronological train/test split where the first 80% of days were used for training and the final 20% were held out for testing. We never used random k-fold cross-validation or shuffled splits, because doing so allows future temporal information to leak into past predictions, producing unrealistically optimistic error scores.

---

## ⚖️ Limitations & Practical Considerations

1. **Statistical Outlier vs. True Physical Leak**: An anomaly flagged by IQR indicates that the volume was statistically rare for that fixture type. It does not definitively prove a mechanical leak.
2. **Discrete Pulse Meter Granularity**: Meters record pulses at 1-Liter resolution every 10 minutes. Short events under 1 Liter may be delayed until the pulse bucket trips.
3. **Local Spark Cluster**: This project runs in `local[*]` mode on a single machine. In an enterprise deployment, the same PySpark DataFrame code runs unchanged on multi-node Kubernetes, Amazon EMR, or Databricks clusters.

---

## 📜 License & Citation
* Dataset: Knutsson, J. (2026). *Household water consumption patterns revealed through high-resolution fixture-level monitoring in a Swedish living lab*. Journal of Cleaner Production. [Zenodo Record 18971107](https://zenodo.org/records/18971107).
* License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
