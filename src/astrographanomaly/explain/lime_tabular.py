from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

def explain_with_lime_regression(X_scaled: np.ndarray, scores: np.ndarray, feature_names: List[str], idx: int, num_features: int = 8, seed: int = 42) -> Dict[str, Any]:
    """LIME for anomaly score as regression target (works for any engine)."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=seed,
    )

    # local predictor: return scalar score
    def predict_fn(Z):
        # nearest-neighbor style proxy: we don't have model.predict in generic form.
        # LIME only needs a function mapping samples->target.
        # Here we approximate by training a small linear surrogate on-the-fly using the provided scores.
        # This remains descriptive: mapping is purely operational for explainability.
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0, random_state=seed)
        reg.fit(X_scaled, scores)
        return reg.predict(Z)

    exp = explainer.explain_instance(X_scaled[idx], predict_fn, num_features=num_features)
    weights = exp.as_list()  # list of (feature_bin, weight)
    return {
        "idx": int(idx),
        "local_score": float(scores[idx]),
        "weights": [{"feature": f, "weight": float(w)} for f, w in weights],
    }
