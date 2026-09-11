import pandas as pd
import joblib
import typing
import numpy as np
import os
import subprocess
import tempfile
import json
import sys
from typing import Tuple, List



def scale_probs(X, model_path='../models/xgbmodel1.pkl'):
    """Predict probabilities by running a separate worker process.

    This isolates model unpickling/prediction into a subprocess to avoid
    crashing the main API process if a native extension misbehaves.
    """
    # prepare input CSV
    with tempfile.TemporaryDirectory() as td:
        in_csv = os.path.join(td, 'input.csv')
        out_json = os.path.join(td, 'out.json')
        # write numeric columns to CSV (worker will select features)
        X.to_csv(in_csv, index=False)
        # model file absolute path
        model_abspath = os.path.join(os.path.dirname(__file__), '..', 'models', os.path.basename(model_path))
        worker = os.path.join(os.path.dirname(__file__), 'predict_worker.py')
        cmd = [sys.executable, worker, '--model', model_abspath, '--input', in_csv, '--output', out_json]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, timeout=60)
        except Exception as e:
            raise RuntimeError(f'prediction subprocess failed: {e}')
        # if worker crashed (non-zero) try to surface readable error
        if proc.returncode != 0:
            stderr = proc.stderr.decode('utf-8', errors='replace') if proc.stderr else ''
            stdout = proc.stdout.decode('utf-8', errors='replace') if proc.stdout else ''
            # attempt to parse json error in stdout
            try:
                payload = json.loads(stdout.strip()) if stdout.strip() else {}
                if 'error' in payload:
                    raise RuntimeError(f'worker error: {payload.get("error")}; stderr: {stderr}')
            except Exception:
                raise RuntimeError(f'prediction worker failed, rc={proc.returncode}; stdout={stdout}; stderr={stderr}')
        # read out.json
        try:
            with open(out_json, 'r') as f:
                payload = json.load(f)
            probs = payload.get('probs', [])
            return np.asarray(probs, dtype=float)
        except Exception as e:
            raise RuntimeError(f'failed to read worker output: {e}')
    
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

