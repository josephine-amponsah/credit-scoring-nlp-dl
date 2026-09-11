import pandas as pd
import typing
import numpy as np
from typing import Tuple, List



# Worker-based scoring removed. The pipeline expects `prob_default` to be present
# in input DataFrames. We min-max scale raw probabilities into `scaled_probs`
# for downstream metric computations.
    
def expected_loss(X):
    """Compute expected loss given a DataFrame with either a `scaled_probs` column
    or compute probabilities from a default model if not present.

    EL is computed as probability_of_default * exposure (funded_amnt).
    Returns total expected loss and the per-loan EL series.
    """
    df = X.copy()
    # Use dataset-provided raw PDs (`prob_default`). If absent, assume zeros.
    raw_probs = df['prob_default'].astype(float).fillna(0.0) if 'prob_default' in df.columns else pd.Series(0.0, index=df.index)

    # min-max scale into `scaled_probs` for stability in metrics and simulation
    a = raw_probs.astype(float).fillna(0.0)
    lo = a.min()
    hi = a.max()
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        scaled = (a - lo) / (hi - lo)
    else:
        scaled = a.clip(0.0, 1.0)
    df['scaled_probs'] = scaled

    exposure = df.get('funded_amnt', pd.Series([0]*len(df))).astype(float).fillna(0.0)
    el = df['scaled_probs'] * exposure
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
    # use provided raw PDs and min-max scale into [0,1]
    raw = df['prob_default'].astype(float).fillna(0.0) if 'prob_default' in df.columns else pd.Series(0.0, index=df.index)
    lo = raw.min()
    hi = raw.max()
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        probs = np.clip(((raw - lo) / (hi - lo)).values, 0.0, 1.0)
    else:
        probs = np.clip(raw.values, 0.0, 1.0)
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

