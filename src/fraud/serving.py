"""MLflow pyfunc wrapper: what Model Serving loads. Needs the [databricks] extra."""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd

from .pipeline import FraudModel
from .state import score_with_state


class FraudPyfunc(mlflow.pyfunc.PythonModel):
    """Takes transaction columns + the card-state columns the feature lookup joins in."""

    def __init__(self, model: FraudModel):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        proba = score_with_state(self.model, model_input)
        return pd.DataFrame({"fraud_probability": proba, "flagged": proba >= self.model.threshold})
