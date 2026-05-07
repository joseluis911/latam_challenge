"""Flight delay prediction model for SCL airport.

Predicts whether a flight at SCL will be delayed (>15 minutes).
Uses Logistic Regression with class balancing, trained on the top-10 features
identified by the Data Scientist's notebook.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Top-10 features chosen by XGBoost feature importance in the DS notebook.
# These are the one-hot encoded columns of OPERA, TIPOVUELO, and MES.
TOP_10_FEATURES: list[str] = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

DEFAULT_TARGET: str = "delay"
DELAY_THRESHOLD_MINUTES: int = 15
RANDOM_STATE: int = 1
DATA_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "data.csv"


class DelayModel:
    """Logistic Regression with `class_weight='balanced'` over the top-10 features."""

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Build the top-10 one-hot features and (optionally) the target column.

        Args:
            data: Raw flight rows. Must contain `OPERA`, `TIPOVUELO`, `MES`. If a
                target is requested and not present, must also contain
                `Fecha-I` and `Fecha-O` so it can be derived.
            target_column: Name of the target column. When set, the target
                DataFrame is also returned.

        Returns:
            `features` only when no target is requested, otherwise
            `(features, target)`.
        """
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        ).reindex(columns=TOP_10_FEATURES, fill_value=0)

        if target_column is None:
            return features

        if target_column in data.columns:
            target = data[[target_column]].copy()
        else:
            min_diff = data.apply(self._compute_min_diff, axis=1)
            target = (min_diff > DELAY_THRESHOLD_MINUTES).astype(int).to_frame(name=target_column)

        return features, target

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train a balanced Logistic Regression on the provided features and target."""
        y = target.iloc[:, 0] if isinstance(target, pd.DataFrame) else target
        self._model = LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )
        self._model.fit(features, y)

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Predict delays for the given features. Auto-trains on bundled data if
        the model has not been explicitly fitted yet."""
        if self._model is None:
            self._bootstrap_from_disk()
        if self._model is None:
            return [0] * len(features)
        return [int(p) for p in self._model.predict(features)]

    # ------------------------------------------------------------------ helpers

    def _bootstrap_from_disk(self) -> None:
        """Train on `data/data.csv` if available; stay unfitted otherwise."""
        if not DATA_PATH.exists():
            return
        raw = pd.read_csv(DATA_PATH, low_memory=False)
        features, target = self.preprocess(raw, target_column=DEFAULT_TARGET)
        self.fit(features, target)

    @staticmethod
    def _compute_min_diff(row: pd.Series) -> float:
        """Minutes between the operated and scheduled flight times."""
        scheduled = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        operated = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        return (operated - scheduled).total_seconds() / 60.0
