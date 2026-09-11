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
    raw_probs = df['prob_default'].astype(float).fillna(0.0) if 'prob_default' in df.columns else pd.Series(0.0, index=df.index)
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


def _default_event_flag(status: typing.Any) -> int:
    """Return 1 if the loan status indicates a default-like event, else 0."""
    if pd.isna(status):
        return 0
    value = str(status).strip()
    if not value:
        return 0
    value_l = value.lower()
    default_like = (
        'charged off' in value_l or
        'default' in value_l or
        'late (' in value_l or
        'late' in value_l or
        'grace' in value_l or
        'does not meet' in value_l
    )
    return 1 if default_like else 0


def survival(X):
    """Return a compact Kaplan-Meier style survival curve for the selected cohort.

    The dataset does not include a clear default timestamp, so we approximate each
    loan's observation age in months from `issue_d` and treat default-like status
    values as event observations. This keeps the metric simple and safe for the
    dashboard UI without needing additional modelling work.
    """
    if X is None or X.empty:
        return {
            'months': [],
            'survival': [],
            'events': [],
            'at_risk': [],
            'median_months': None,
            'n_loans': 0,
        }

    df = X.copy()
    if 'issue_d' not in df.columns:
        return {
            'months': [],
            'survival': [],
            'events': [],
            'at_risk': [],
            'median_months': None,
            'n_loans': 0,
        }

    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    df = df.dropna(subset=['issue_d']).copy()
    if df.empty:
        return {
            'months': [],
            'survival': [],
            'events': [],
            'at_risk': [],
            'median_months': None,
            'n_loans': 0,
        }

    # Approximate each loan's follow-up time in months based on its issue date.
    today = pd.Timestamp.utcnow().normalize()
    df['months_since_issue'] = ((today - df['issue_d']).dt.days / 30.4375).clip(lower=0)
    df['event_flag'] = df['loan_status'].apply(_default_event_flag)

    # Use the observation month as the discrete time index for a simple Kaplan-Meier.
    month_map = np.ceil(df['months_since_issue']).astype(int)
    df['month_index'] = month_map

    # Build the curve in ascending month order.
    months = sorted(df['month_index'].unique().tolist())
    survival_curve = []
    events_list = []
    at_risk_list = []
    running_survival = 1.0

    for month in months:
        at_risk = int((df['month_index'] >= month).sum())
        events = int(((df['month_index'] == month) & (df['event_flag'] == 1)).sum())
        if at_risk > 0:
            if events > 0:
                running_survival *= (1 - (events / at_risk))
            survival_curve.append(float(running_survival))
            events_list.append(events)
            at_risk_list.append(at_risk)
        else:
            survival_curve.append(float(running_survival))
            events_list.append(0)
            at_risk_list.append(0)

    if not survival_curve:
        return {
            'months': [],
            'survival': [],
            'events': [],
            'at_risk': [],
            'median_months': None,
            'n_loans': int(len(df)),
        }

    median_months = None
    for month, value in zip(months, survival_curve):
        if value <= 0.5:
            median_months = float(month)
            break

    return {
        'months': [int(m) for m in months],
        'survival': [float(v) for v in survival_curve],
        'events': [int(v) for v in events_list],
        'at_risk': [int(v) for v in at_risk_list],
        'median_months': median_months,
        'n_loans': int(len(df)),
    }


def simulate_portfolio_losses(df: pd.DataFrame, nsim: int = 10000, seed: typing.Optional[int] = None) -> np.ndarray:
    """Simulate portfolio loss distribution using Bernoulli defaults per loan.

    Returns an array of simulated total losses (same currency as `funded_amnt`).
    """
    rnd = np.random.RandomState(seed)
    raw = df['prob_default'].astype(float).fillna(0.0) if 'prob_default' in df.columns else pd.Series(0.0, index=df.index)
    lo = raw.min()
    hi = raw.max()
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        probs = np.clip(((raw - lo) / (hi - lo)).values, 0.0, 1.0)
    else:
        probs = np.clip(raw.values, 0.0, 1.0)
    exposure = df.get('funded_amnt', pd.Series([0]*len(df))).astype(float).fillna(0.0).values
    n = len(probs)
    if n == 0:
        return np.array([])
    U = rnd.rand(nsim, n)
    defaults = (U < probs).astype(float)
    losses = defaults * exposure
    port_losses = losses.sum(axis=1)
    return port_losses


def var_from_simulations(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Return portfolio VaR at given confidence level from simulated losses array."""
    if losses is None or len(losses) == 0:
        return 0.0
    return float(np.quantile(losses, confidence))

