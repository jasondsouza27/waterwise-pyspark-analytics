# WaterWise — Dataset Documentation

## Dataset Source
* **Origin**: HSB Living Lab (Habitation Lab), Chalmers University of Technology, Gothenburg, Sweden
* **Zenodo Record**: [https://zenodo.org/records/18971107](https://zenodo.org/records/18971107)
* **Citation**: Knutsson, J. (2026). *Household water consumption patterns revealed through high-resolution fixture-level monitoring in a Swedish living lab*. Journal of Cleaner Production.
* **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Dataset Description
The dataset contains high-resolution, fixture-level water consumption records collected across **28 residential apartments** grouped into **9 residential clusters** over the period **December 2019 to January 2023** (946 days with valid telemetry, 83.9% temporal coverage). Measurements are captured at **10-minute intervals** by pulse-counting water meters with a resolution of 1 Liter (0.001 m³).

---

## Directory Setup & File Placement
To run WaterWise with the real dataset:
1. Download the archive `HSB_Living_Lab_Data_Package.zip` from Zenodo:
   `https://zenodo.org/records/18971107/files/HSB_Living_Lab_Data_Package.zip?download=1`
2. Extract the archive contents into this `data/` folder.
3. Ensure the extracted file `HSB_Living_Lab_Water_Consumption_Anonymized.csv` is present directly in `data/`:
   ```
   data/
   ├── HSB_Living_Lab_Water_Consumption_Anonymized.csv
   ├── DATA_DICTIONARY.txt
   ├── README.txt
   ├── scripts/
   │   ├── anonymize_data.py
   │   ├── generate_figures.py
   │   └── water_analysis.py
   └── README.md
   ```

*Note*: If the CSV file is missing, the application will cleanly fail with an explicit error guiding you to place the dataset here. Synthetic or random data will **never** be silently generated.

---

## Real CSV Schema & Internal Schema Mapping

The raw CSV uses semicolon (`;`) delimiters with UTF-8 encoding.

| CSV Column Name | Raw Type | Description | Internal Spark Schema & Mapping |
| :--- | :--- | :--- | :--- |
| `timestamp` | String (YYYY-MM-DD HH:MM:SS) | Event timestamp (UTC, 10-min rounded) | `TimestampType` (`timestamp`) |
| `date` | String (YYYY-MM-DD) | Calendar date | `DateType` (`date`) |
| `hour` | Integer (0-23) | Hour of day (UTC) | `IntegerType` (`hour`) |
| `day_of_week` | Integer (0-6) | Day of week (0=Mon, 6=Sun) | `IntegerType` (`day_of_week`) |
| `day_name` | String | Day name (Monday-Sunday) | `StringType` (`day_name`) |
| `month` | Integer (1-12) | Calendar month | `IntegerType` (`month`) |
| `year` | Integer | Calendar year | `IntegerType` (`year`) |
| `sensor_id` | String (S0001...) | Anonymized water meter ID | `StringType` (`sensor_id`) |
| `apartment` | String (APT_001...) | Anonymized unit ID (null for shared) | `StringType` (`apartment`) |
| `cluster_name` | String (CL_A...CL_I) | Residential cluster ID | `StringType` (`cluster_name`) |
| `room_number` | String (R001...) | Anonymized room ID | `StringType` (`room_number`) |
| `room_type` | String | Room type (Bathroom, Living room, Kitchen, etc.) | `StringType` (`room_type`) |
| `type` | String | `cold water consumption` or `hot water consumption` | `StringType` (`water_type`) |
| `attached_to` | String | Fixture: Toilet, Kitchen tap, Washbasin, Shower, Washing machine | `StringType` (`fixture`) |
| `value` | Float | Volume in m³ for this 10-min interval (0.001 m³ = 1 L) | `DoubleType` (`volume_m3`) -> Derived: `volume_liters` |
| `aggregated_value` | Float | Cumulative volume in m³ | `DoubleType` (`cumulative_m3`) |

---

## Data Cleaning & Validation Rules
1. **Timestamp Validation**: Cast to `TimestampType`. Rows with unparseable timestamps or outside 2019-2023 are rejected.
2. **Value Validation**: Measurements must be strictly positive (`value > 0`) and below the physical sensor threshold (`value <= 0.3` m³, corresponding to 300 Liters in a 10-minute window).
3. **Duplicate Detection**: De-duplication is performed strictly on composite primary keys `(timestamp, sensor_id)` to preserve legitimate concurrent readings from different meters.
4. **Shared Facility Handling**: Records with null `apartment` or `cluster_name` represent shared facility meters (e.g. shared laundry, common bathrooms). These are retained for overall building water analytics and categorized as `'Shared Facility'`.
