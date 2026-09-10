import pandas as pd
import joblib
import typing
import numpy as np
import os
from typing import Tuple, List



def scale_probs(X, model_path = '../models/xgbmodel1.pkl' ):
    """Load a classifier and return predicted default probabilities for rows in X.

    The function attempts to select the model's expected feature columns if available
    (`feature_names_in_`), otherwise it will try to use numeric columns from X.
    Returns a 1D numpy array of probabilities for the positive/default class.
    """
    model = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', os.path.basename(model_path)))
    # find feature columns
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    else:
        # fallback: use numeric columns excluding identifier/date-like columns
        features = X.select_dtypes(include=[np.number]).columns.tolist()
        # if 'funded_amnt' present, keep it too (but numeric already)
    if not features:
        raise ValueError('No feature columns found for scoring')
    # ensure features exist in X
    missing = [c for c in features if c not in X.columns]
    if missing:
        # try to reduce to intersection
        features = [c for c in features if c in X.columns]
    if not features:
        raise ValueError('No model feature columns present in input dataframe')
    Xf = X[features].fillna(0)
    # predict_proba may exist (sklearn/XGBoost wrapper)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(Xf)
        # assume positive class is column 1
        if probs.ndim == 2 and probs.shape[1] > 1:
            return np.asarray(probs[:, 1], dtype=float)
        # fallback: if single-column, return it
        return np.asarray(probs.ravel(), dtype=float)
    # fallback: some models expose predict which returns scores; try to use it
    if hasattr(model, 'predict'):
        preds = model.predict(Xf)
        return np.asarray(preds, dtype=float)
    raise ValueError('Model has no predict_proba or predict')
    
def expected_loss(X):
    """Compute expected loss given a DataFrame with either a `scaled_probs` column
    or compute probabilities from a default model if not present.

    EL is computed as probability_of_default * exposure (funded_amnt).
    Returns total expected loss and the per-loan EL series.
    """
    df = X.copy()
    if 'scaled_probs' not in df.columns:
        # try to score using default model in ../models/xgbmodel1.pkl
        try:
            df['scaled_probs'] = scale_probs(df)
        except Exception:
            df['scaled_probs'] = 0.0
    probs = df['scaled_probs'].astype(float).fillna(0.0)
    exposure = df.get('funded_amnt', pd.Series([0]*len(df))).astype(float).fillna(0.0)
    el = probs * exposure
    return float(el.sum()), el

def npv(X):
    return None

def survival(X):
    # select columns that matter and values to be displayed 
    # in survival analysis for selected period or cohort
    df = X.copy()
    return df


def simulate_portfolio_losses(df: pd.DataFrame, nsim: int = 10000, seed: typing.Optional[int] = None) -> np.ndarray:
    """Simulate portfolio loss distribution using Bernoulli defaults per loan.

    Returns an array of simulated total losses (same currency as `funded_amnt`).
    """
    rnd = np.random.RandomState(seed)
    if 'scaled_probs' not in df.columns:
        try:
            df['scaled_probs'] = scale_probs(df)
        except Exception:
            df['scaled_probs'] = 0.0
    probs = np.clip(df['scaled_probs'].astype(float).fillna(0.0).values, 0.0, 1.0)
    exposure = df.get('funded_amnt', pd.Series([0]*len(df))).astype(float).fillna(0.0).values
    # shape (nsim, n_loans): draw bernoulli for each loan
    # to save memory, simulate in chunks if needed
    n = len(probs)
    if n == 0:
        return np.array([])
    # vectorized simulation: generate (nsim, n) uniform randoms
    U = rnd.rand(nsim, n)
    defaults = (U < probs).astype(float)
    losses = defaults * exposure
    # sum across loans for each simulation
    port_losses = losses.sum(axis=1)
    return port_losses


def var_from_simulations(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Return portfolio VaR at given confidence level from simulated losses array."""
    if losses is None or len(losses) == 0:
        return 0.0
    return float(np.quantile(losses, confidence))

