"""
Utilities for working with pre-filtered player datasets (e.g. FIFA 21 subset).
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from django.conf import settings

from .feature_engineering import REQUIRED_FEATURES
from .ml_model import predict


FIFA21_DATASET_FILENAME = "fifa21_players_for_frontend.csv"

# In-memory cache so we only build/load the FIFA 21 dataset once
_FIFA21_DF: Optional[pd.DataFrame] = None


def _get_fifa21_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask selecting FIFA 21 players.

    The `train.csv` file contains a `fifa_version` column. We treat version 21
    (either as 21 or 21.0) as FIFA 21.
    """
    if "fifa_version" not in df.columns:
        raise ValueError("Expected 'fifa_version' column in player dataset.")

    # Handle float/int representations like 21 and 21.0
    return df["fifa_version"].astype(float).round() == 21.0


def build_fifa21_dataset(precompute_predictions: bool = False) -> pd.DataFrame:
    """
    Build a FIFA 21-only dataset with required model features and metadata.

    The resulting DataFrame includes:
      - player_id
      - short_name
      - long_name
      - nationality_name
      - the 27 REQUIRED_FEATURES used by the model
      - (optional) will_be_superstar and probability fields if precompute_predictions
        is True.

    The dataset is also written to DATA_DIR / FIFA21_DATASET_FILENAME.
    """
    # Load directly from the raw train.csv to avoid circular imports with views.
    csv_path = settings.DATA_DIR / "train.csv"

    # Columns for metadata + features
    # We include extra fields that are useful for filtering and display
    # (age, positions, club, overall, potential, base stats).
    meta_cols = [
        "player_id",
        "short_name",
        "long_name",
        "nationality_name",
        "age",
        "player_positions",
        "club_name",
        "overall",
        "potential",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physic",
    ]

    base_columns = meta_cols + list(REQUIRED_FEATURES) + ["fifa_version"]
    # Deduplicate in case of overlaps
    base_columns = list(dict.fromkeys(base_columns))

    df = pd.read_csv(csv_path, usecols=base_columns, low_memory=False)

    mask = _get_fifa21_mask(df)
    fifa21_df = df.loc[mask].copy()

    missing_features = [f for f in REQUIRED_FEATURES if f not in fifa21_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    # Keep only columns that actually exist in the dataframe
    existing_meta_cols = [c for c in meta_cols if c in fifa21_df.columns]
    selected_cols = existing_meta_cols + REQUIRED_FEATURES
    fifa21_selected = fifa21_df[selected_cols].copy()

    if precompute_predictions:
        probabilities = []
        superstar_flags = []

        for _, row in fifa21_selected.iterrows():
            features = [row[feat] for feat in REQUIRED_FEATURES]
            result = predict(features)
            prob = float(result["probability"])
            prediction_class = int(result["prediction"])

            probabilities.append(prob)
            superstar_flags.append(prediction_class == 1)

        fifa21_selected["probability_class_1"] = probabilities
        fifa21_selected["will_be_superstar"] = superstar_flags

    # Cache in memory for this process so subsequent calls are fast
    global _FIFA21_DF
    _FIFA21_DF = fifa21_selected

    # Persist to CSV for later reuse when filesystem is writable.
    # In some environments (e.g. Docker read-only volumes) this may fail,
    # so we ignore OSError and just return the in-memory DataFrame.
    output_path: Path = settings.DATA_DIR / FIFA21_DATASET_FILENAME
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fifa21_selected.to_csv(output_path, index=False)
    except OSError:
        # Read-only filesystem or similar issue; continue without caching to disk.
        pass

    return fifa21_selected


def load_fifa21_dataset() -> Optional[pd.DataFrame]:
    """
    Load the pre-built FIFA 21 dataset if it exists, otherwise return None.
    """
    global _FIFA21_DF

    # Prefer the in-memory cache if already built/loaded
    if _FIFA21_DF is not None:
        return _FIFA21_DF

    # Fallback to CSV on disk if available
    csv_path = settings.DATA_DIR / FIFA21_DATASET_FILENAME
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    _FIFA21_DF = df
    return df

