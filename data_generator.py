"""Synthetic campus electricity consumption dataset generator and CSV validator."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

HOUR_COLUMNS = [f"{hour:02d}:00" for hour in range(24)]
METADATA_COLUMNS = ["building_id", "building_name", "building_type", "floor_area"]
REQUIRED_COLUMNS = METADATA_COLUMNS + HOUR_COLUMNS

DEFAULT_DATA_PATH = os.path.join("data", "campus_energy_data.csv")

# 30 realistic campus buildings across all 8 required categories
BUILDING_CATALOG = [
    # Libraries (4)
    ("B001", "Central Library", "Library", 4200, 1.10),
    ("B002", "Science & Tech Library", "Library", 3100, 0.95),
    ("B003", "Humanities Reading Hall", "Library", 2400, 0.85),
    ("B004", "Law & Management Library", "Library", 2600, 0.90),
    # Classrooms (6)
    ("B005", "Academic Block A", "Classroom", 3500, 1.05),
    ("B006", "Academic Block B", "Classroom", 3200, 0.95),
    ("B007", "Academic Block C", "Classroom", 2900, 0.88),
    ("B008", "Freshman Lecture Hall", "Classroom", 2700, 0.85),
    ("B009", "Engineering Annex", "Classroom", 3100, 1.00),
    ("B010", "Seminar Complex", "Classroom", 2100, 0.78),
    # Computer Laboratories (4)
    ("B011", "CS Computer Lab 1", "Computer Laboratory", 2200, 1.05),
    ("B012", "CS Computer Lab 2", "Computer Laboratory", 2000, 0.95),
    ("B013", "AI & Data Science Lab", "Computer Laboratory", 2500, 1.15),
    ("B014", "Networking & Systems Lab", "Computer Laboratory", 1900, 0.90),
    # Administrative Buildings (3)
    ("B015", "Main Admin Block", "Administrative Building", 2800, 1.05),
    ("B016", "Admissions & Records Office", "Administrative Building", 1800, 0.82),
    ("B017", "Finance & Accounts Wing", "Administrative Building", 1600, 0.78),
    # Auditoriums (2)
    ("B018", "Main Convocation Auditorium", "Auditorium", 4500, 1.15),
    ("B019", "Cultural & Lecture Auditorium", "Auditorium", 3000, 0.88),
    # Canteens (3)
    ("B020", "Central Student Canteen", "Canteen", 2200, 1.10),
    ("B021", "North Campus Food Court", "Canteen", 1900, 0.95),
    ("B022", "Faculty & Staff Dining Hall", "Canteen", 1400, 0.80),
    # Research Laboratories (4)
    ("B023", "Robotics & Automation Lab", "Research Laboratory", 2600, 1.08),
    ("B024", "Biotechnology Research Center", "Research Laboratory", 3000, 1.18),
    ("B025", "Materials Testing Lab", "Research Laboratory", 2300, 0.95),
    ("B026", "Renewable Energy Lab", "Research Laboratory", 2400, 1.00),
    # Sports Complexes (4)
    ("B027", "Indoor Sports Arena", "Sports Complex", 4800, 1.12),
    ("B028", "Campus Gymnasium", "Sports Complex", 2500, 0.92),
    ("B029", "Aquatic Center", "Sports Complex", 3600, 1.05),
    ("B030", "Outdoor Stadium Pavilion", "Sports Complex", 3200, 0.88),
]


def _base_profile_for_type(building_type: str) -> np.ndarray:
    """Return a 24-hour baseline consumption curve (kWh) for a building type."""
    profiles = {
        "Library": np.array([
            14, 13, 12, 12, 12, 14, 18, 28,
            48, 58, 64, 68, 66, 65, 68, 70,
            72, 74, 72, 68, 62, 50, 32, 18,
        ], dtype=float),
        "Classroom": np.array([
            6, 5, 5, 5, 5, 6, 10, 18,
            34, 42, 46, 48, 42, 44, 46, 42,
            32, 20, 12, 9, 8, 7, 6, 6,
        ], dtype=float),
        "Computer Laboratory": np.array([
            28, 27, 26, 26, 26, 28, 34, 52,
            82, 102, 112, 116, 108, 114, 118, 115,
            106, 90, 66, 52, 44, 38, 32, 30,
        ], dtype=float),
        "Administrative Building": np.array([
            7, 7, 6, 6, 6, 7, 10, 18,
            32, 44, 48, 50, 46, 48, 48, 46,
            40, 26, 14, 10, 8, 8, 7, 7,
        ], dtype=float),
        "Auditorium": np.array([
            8, 7, 7, 7, 7, 8, 10, 14,
            22, 28, 32, 34, 30, 38, 54, 64,
            68, 66, 58, 46, 28, 16, 10, 8,
        ], dtype=float),
        "Canteen": np.array([
            12, 11, 11, 11, 12, 16, 24, 38,
            52, 46, 36, 48, 74, 78, 62, 38,
            36, 46, 58, 54, 40, 26, 16, 13,
        ], dtype=float),
        "Research Laboratory": np.array([
            52, 50, 49, 48, 48, 50, 56, 68,
            88, 106, 118, 124, 120, 122, 126, 124,
            116, 102, 86, 74, 66, 60, 56, 54,
        ], dtype=float),
        "Sports Complex": np.array([
            14, 13, 12, 12, 13, 20, 38, 44,
            32, 26, 24, 24, 26, 28, 32, 42,
            62, 82, 92, 94, 88, 72, 42, 20,
        ], dtype=float),
    }
    return profiles[building_type]


def generate_campus_energy_data(
    output_path: str = DEFAULT_DATA_PATH,
    seed: int = 42,
    save_to_disk: bool = True,
) -> pd.DataFrame:
    """Generate a reproducible synthetic campus electricity dataset for 30 buildings."""
    rng = np.random.default_rng(seed)
    rows = []

    for building_id, building_name, building_type, floor_area, scale in BUILDING_CATALOG:
        base_curve = _base_profile_for_type(building_type) * scale
        # Add realistic hourly variation (~6% std dev) and slight building-level offset
        noise = rng.normal(loc=0.0, scale=0.06 * base_curve)
        hourly_values = np.clip(base_curve + noise, a_min=2.0, a_max=None)
        hourly_values = np.round(hourly_values, 2)

        row = {
            "building_id": building_id,
            "building_name": building_name,
            "building_type": building_type,
            "floor_area": floor_area,
        }
        for hour_col, val in zip(HOUR_COLUMNS, hourly_values):
            row[hour_col] = float(val)
        rows.append(row)

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

    if save_to_disk:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def load_or_generate_dataset(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load dataset from disk if it exists and is valid; otherwise generate it."""
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        is_valid, _ = validate_campus_csv(df)
        if is_valid:
            return df
    return generate_campus_energy_data(output_path=data_path, save_to_disk=True)


def validate_campus_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that an uploaded or loaded DataFrame has required columns and valid data."""
    if df is None or df.empty:
        return False, "The CSV file is empty."

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        preview = ", ".join(missing_cols[:6])
        suffix = "..." if len(missing_cols) > 6 else ""
        return False, f"Missing required columns: {preview}{suffix}"

    if df[METADATA_COLUMNS].isnull().any().any():
        return False, "Metadata columns (building_id, building_name, building_type, floor_area) contain missing values."

    for col in ["floor_area"] + HOUR_COLUMNS:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if numeric_series.isnull().any():
            return False, f"Column '{col}' contains non-numeric or missing values."
        if (numeric_series < 0).any():
            return False, f"Column '{col}' contains negative energy or area values."

    if len(df) < 5:
        return False, "Dataset must contain at least 5 buildings for meaningful clustering."

    return True, ""


if __name__ == "__main__":
    generated_df = generate_campus_energy_data()
    print(f"Generated {len(generated_df)} buildings at '{DEFAULT_DATA_PATH}'.")
