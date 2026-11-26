# ⚡ Campus Energy Usage Profiling Using Machine Learning

## Project Overview

This project uses Machine Learning to analyze electricity consumption across college campus buildings and group them according to their 24-hour energy usage patterns. It uses real university campus meter data from the Building Data Genome Project 2 (BDG2) and compares unsupervised clustering algorithms through an interactive Streamlit dashboard.

## Deployed Link :
https://campustrack.streamlit.app/

## Objectives

- Analyze daily and hourly electricity consumption across campus buildings.
- Group buildings with similar load profiles using unsupervised Machine Learning.
- Visualize high-dimensional hourly consumption patterns in 2D space using PCA.
- Provide rule-based energy-saving recommendations for high-load and off-hours consumption.
- Build an interactive, easy-to-explain energy analytics dashboard.

## Dataset

**Source:** [Building Data Genome Project 2 (BDG2)](https://github.com/buds-lab/building-data-genome-project-2) (`Wolf` Education Site)

The default dataset (`data/campus_energy_data.csv`) contains **36 real university campus buildings** across 7 categories (`College Classroom`, `Office`, `Retail`, `Fitness Center`, `Data Centre`, `School`, and `Library`), with `building_id`, `building_name`, `building_type`, `floor_area` (sq. m.), and 24-hour average electricity consumption (`00:00` to `23:00` in kWh).

- Supports **custom CSV upload** directly from the sidebar with schema validation.
- Includes a **synthetic fallback generator** (`data_generator.py`) if `data/campus_energy_data.csv` is not present.

## Machine Learning Models

1. **K-Means Clustering** -- partitions buildings into *k* distinct consumption clusters (2 to 5) by minimizing within-cluster variance.
2. **Gaussian Mixture Model (GMM)** -- models building consumption profiles as a mixture of Gaussian distributions for probabilistic clustering.
3. **Principal Component Analysis (PCA)** -- reduces the 24 hourly consumption features into 2 principal components for interactive 2D cluster visualization.

## Methodology

1. Load the campus energy dataset (or validate an uploaded CSV).
2. Extract the 24 hourly consumption features (`00:00`--`23:00`).
3. Standardize features using `StandardScaler`.
4. Apply K-Means or Gaussian Mixture Model clustering (2--5 clusters).
5. Dynamically assign descriptive cluster names (e.g., *Low Consumption*, *Medium Consumption*, *High Consumption*) based on each cluster's average daily usage.
6. Evaluate cluster separation using the Silhouette Score.
7. Project standardized features into 2D using PCA and generate rule-based energy-saving recommendations.

## Evaluation Metrics

- Silhouette Score (cluster separation quality)
- PCA Explained Variance Ratio (PC1, PC2, and total variance retained)
- Cluster-level average daily consumption and building counts

The application also generates interactive building-level bar charts, hourly load curves, building-type donut charts, and 2D PCA scatter plots.

## Technologies Used

Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Matplotlib, and the Building Data Genome Project 2 (BDG2) dataset.

## Project Structure

```text
CampusTrack/
├── .streamlit/
│   └── config.toml
├── data/
│   └── campus_energy_data.csv
├── app.py
├── data_generator.py
├── ml_model.py
├── requirements.txt
└── README.md
```

## Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/your-username/CampusTrack.git
cd CampusTrack
```

**2. Create and activate a virtual environment (Windows)**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Future Scope

- Integration of weather and occupancy data for deeper energy correlation.
- Time-series forecasting for peak demand prediction.
- Multi-campus comparison across additional BDG2 university sites.
- Automated PDF/CSV energy audit report exports.

## Disclaimer

This project is developed for academic and educational purposes only. Recommendations are rule-based estimates derived from historical hourly averages and should not replace a formal facility engineering audit.

## Author
Shreyas Ghate
**Campus Energy Usage Profiling**  
Machine Learning | Python | Streamlit
