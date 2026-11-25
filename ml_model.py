"""Machine learning module for campus energy clustering, PCA, and recommendations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from data_generator import HOUR_COLUMNS

CLUSTER_NAME_TIERS: Dict[int, List[str]] = {
    2: ["Low Consumption", "High Consumption"],
    3: ["Low Consumption", "Medium Consumption", "High Consumption"],
    4: [
        "Very Low Consumption",
        "Low Consumption",
        "Medium Consumption",
        "High Consumption",
    ],
    5: [
        "Very Low Consumption",
        "Low Consumption",
        "Medium Consumption",
        "High Consumption",
        "Very High Consumption",
    ],
}

NIGHT_HOURS = ["22:00", "23:00", "00:00", "01:00", "02:00", "03:00", "04:00", "05:00"]
DAY_HOURS = [f"{h:02d}:00" for h in range(8, 19)]


def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, pd.Series]:
    """Extract 24 hourly features, standardize them, and compute daily totals."""
    hourly_data = df[HOUR_COLUMNS].astype(float)
    daily_consumption = hourly_data.sum(axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hourly_data.values)
    return X_scaled, scaler, daily_consumption


def _assign_dynamic_cluster_names(
    labels: np.ndarray,
    daily_consumption: pd.Series,
    requested_clusters: int,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Map raw cluster IDs to consumption-ordered names based on average consumption."""
    unique_labels = np.unique(labels)
    cluster_means = {
        int(cid): float(daily_consumption[labels == cid].mean())
        for cid in unique_labels
    }
    sorted_cluster_ids = sorted(cluster_means.keys(), key=lambda cid: cluster_means[cid])

    tier_names = CLUSTER_NAME_TIERS.get(len(sorted_cluster_ids))
    if tier_names is None:
        tier_names = [f"Cluster {i + 1}" for i in range(len(sorted_cluster_ids))]

    id_to_name = {
        cid: tier_names[idx] for idx, cid in enumerate(sorted_cluster_ids)
    }
    named_labels = np.array([id_to_name[int(cid)] for cid in labels])
    return named_labels, id_to_name


def run_clustering(
    df: pd.DataFrame,
    algorithm: str = "K-Means",
    n_clusters: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run K-Means or Gaussian Mixture Model clustering on standardized hourly data."""
    X_scaled, _, daily_consumption = preprocess_features(df)

    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        raw_labels = model.fit_predict(X_scaled)
    elif algorithm == "Gaussian Mixture Model":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        raw_labels = model.fit_predict(X_scaled)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    cluster_names, id_to_name = _assign_dynamic_cluster_names(
        raw_labels, daily_consumption, n_clusters
    )

    unique_count = len(np.unique(raw_labels))
    sil_score: Optional[float] = None
    if unique_count >= 2 and len(df) > unique_count:
        sil_score = float(silhouette_score(X_scaled, raw_labels))

    result_df = pd.DataFrame({
        "building_id": df["building_id"].values,
        "building_name": df["building_name"].values,
        "building_type": df["building_type"].values,
        "floor_area": df["floor_area"].values,
        "daily_consumption": np.round(daily_consumption.values, 2),
        "cluster_id": raw_labels,
        "cluster_name": cluster_names,
    })

    # Order summary by average daily consumption ascending
    ordered_names = [
        id_to_name[cid]
        for cid in sorted(id_to_name.keys(), key=lambda c: result_df.loc[result_df["cluster_id"] == c, "daily_consumption"].mean())
    ]

    summary_rows = []
    for cname in ordered_names:
        subset = result_df[result_df["cluster_name"] == cname]
        summary_rows.append({
            "Cluster": cname,
            "Number of Buildings": int(len(subset)),
            "Average Daily Consumption (kWh)": round(float(subset["daily_consumption"].mean()), 2),
        })
    cluster_summary = pd.DataFrame(summary_rows)

    return {
        "algorithm": algorithm,
        "n_clusters_requested": n_clusters,
        "n_clusters_found": unique_count,
        "result_df": result_df,
        "cluster_summary": cluster_summary,
        "silhouette_score": sil_score,
        "X_scaled": X_scaled,
    }


def run_pca(X_scaled: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce standardized 24-hour consumption features into 2 principal components."""
    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(X_scaled)
    explained_variance_pct = pca.explained_variance_ratio_ * 100.0
    return components, explained_variance_pct


def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate rule-based energy-saving recommendations from actual consumption data."""
    hourly_df = df[HOUR_COLUMNS].astype(float)
    daily_totals = hourly_df.sum(axis=1)
    q25 = float(daily_totals.quantile(0.25))
    q75 = float(daily_totals.quantile(0.75))
    campus_avg = float(daily_totals.mean())

    records = []
    for idx, row in df.iterrows():
        hourly_vals = hourly_df.loc[idx]
        daily_total = float(hourly_vals.sum())
        hourly_avg = float(hourly_vals.mean())
        peak_hour = str(hourly_vals.idxmax())
        peak_val = float(hourly_vals.max())

        night_avg = float(hourly_vals[NIGHT_HOURS].mean())
        day_avg = float(hourly_vals[DAY_HOURS].mean())
        night_share = float(hourly_vals[NIGHT_HOURS].sum()) / daily_total if daily_total > 0 else 0.0

        patterns: List[str] = []
        suggestions: List[str] = []

        if daily_total >= q75 or daily_total >= 1.25 * campus_avg:
            patterns.append("Consistently High Energy Consumption")
            suggestions.append(
                "Inspect lighting, air conditioning, and electrical equipment for efficiency upgrades and automated shutdown schedules."
            )

        if (day_avg > 0 and night_avg >= 0.45 * day_avg) or night_share >= 0.28:
            patterns.append("Unusually High Nighttime Consumption")
            suggestions.append(
                "Check whether unnecessary electrical equipment, computers, or HVAC systems are left running after working hours."
            )

        if hourly_avg > 0 and peak_val >= 1.85 * hourly_avg:
            patterns.append(f"Sharp Consumption Peak ({peak_hour})")
            suggestions.append(
                f"Review peak-hour electricity usage around {peak_hour} ({peak_val:.1f} kWh) and schedule high-energy activities more evenly."
            )

        if not patterns and daily_total <= q25:
            patterns.append("Relatively Low Consumption")
            suggestions.append(
                "Positive energy-efficiency observation: energy usage is well below the campus average. Maintain current conservation habits."
            )

        if not patterns:
            patterns.append("Moderate & Balanced Consumption")
            suggestions.append(
                "Energy usage is within normal campus levels. Continue routine equipment maintenance and periodic energy audits."
            )

        records.append({
            "Building": row["building_name"],
            "Building Type": row["building_type"],
            "Daily Consumption (kWh)": round(daily_total, 2),
            "Peak Hour": peak_hour,
            "Identified Consumption Pattern": "; ".join(patterns),
            "Suggested Improvement": " ".join(suggestions),
        })

    return pd.DataFrame(records)
