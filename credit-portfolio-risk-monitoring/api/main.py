# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import torch, joblib, json, numpy as np, pandas as pd
import shap, os
from fastapi.responses import HTMLResponse
from pydantic.dataclasses import dataclass
from fastapi import Query

# local insights helpers (use package-relative import so uvicorn can import api.main)
from .modules import insights
from .modules import risk_metrics

app = FastAPI()

# @dataclass
# class Selection(BaseModel):
#     X : list[float]
#     Start: Optional[datetime] = None
#     End: Optional[datetime] = None
    
# class GraphOutput(BaseModel):
#     Start: Optional[datetime] = None
#     End: Optional[datetime] = None
#     Metric : list[float]


@app.get("/overview/summary")
def overview_summary(period: Optional[str] = None, loan_purpose: Optional[str] = None):
    """Return summary metrics for the requested period and optional loan_purpose filter.
    Query params:
    - period: e.g. '2018Q4'
    - loan_purpose: comma-separated list of loan purpose values
    """
    df = insights.load_data(period)
    if df is None or df.empty:
        return {'total_lent': 0.0, 'repaid': 0.0, 'default_rate': 0.0, 'interest_earned': 0.0}
    if loan_purpose:
        vals = [v.strip() for v in loan_purpose.split(',') if v.strip()]
        # ensure loan_purpose column exists
        if 'loan_purpose' not in df.columns and 'purpose' in df.columns:
            df['loan_purpose'] = df['purpose']
        if 'loan_purpose' in df.columns:
            df = df[df['loan_purpose'].isin(vals)]
    return insights.summary(df)


@app.get("/overview/aggregates")
def overview_aggregates(period: Optional[str] = None, loan_purpose: Optional[str] = None):
    """Return grouped aggregates by loan purpose.
    Returns JSON: { data: [ {loan_purpose, lent, repaid, interest}, ... ] }
    """
    df = insights.load_data(period)
    if df is None or df.empty:
        return {'data': []}
    if loan_purpose:
        vals = [v.strip() for v in loan_purpose.split(',') if v.strip()]
        if 'loan_purpose' not in df.columns and 'purpose' in df.columns:
            df['loan_purpose'] = df['purpose']
        if 'loan_purpose' in df.columns:
            df = df[df['loan_purpose'].isin(vals)]
    rows = insights.aggregates(df)
    return {'data': rows}


@app.get("/overview/options")
def overview_options(period: Optional[str] = None):
    """Return available loan purpose options for the period (or latest when omitted)."""
    df = insights.load_data(period)
    if df is None or df.empty:
        return {'loan_purposes': []}
    opts = insights.loan_purposes(df)
    return {'loan_purposes': opts}


@app.get("/overview/periods")
def overview_periods():
    """Return available period identifiers (e.g. '2018Q4')."""
    try:
        periods = insights.list_periods()
        return {'periods': periods}
    except Exception:
        return {'periods': []}



@app.get("/risk/expected_loss")
def risk_expected_loss(period: Optional[str] = None):
    """Return total expected loss and simple breakdown for the requested period."""
    try:
        df = insights.load_data(period)
        if df is None or df.empty:
            return {'total_expected_loss': 0.0, 'n_loans': 0}
        total_el, per_loan = risk_metrics.expected_loss(df)
        return {'total_expected_loss': total_el, 'n_loans': int(len(df))}
    except Exception as e:
        return {'total_expected_loss': 0.0, 'n_loans': 0, 'error': str(e)}


@app.get("/risk/var")
def risk_var(period: Optional[str] = None, confidence: Optional[float] = 0.95, nsim: Optional[int] = 5000):
    """Simulate portfolio loss distribution and return VaR and histogram bins.

    Query params:
    - period: quarter identifier
    - confidence: VaR confidence (0-1)
    - nsim: number of Monte Carlo simulations
    """
    try:
        df = insights.load_data(period)
        if df is None or df.empty:
            return {'var': 0.0, 'hist': {'bins': [], 'counts': []}}
        losses = risk_metrics.simulate_portfolio_losses(df, nsim=int(nsim))
        if losses.size == 0:
            return {'var': 0.0, 'hist': {'bins': [], 'counts': []}}
        var = risk_metrics.var_from_simulations(losses, confidence=float(confidence))
        # histogram
        import numpy as _np
        counts, bin_edges = _np.histogram(losses, bins=50)
        # return as lists for JSON serialization
        return {'var': float(var), 'hist': {'bins': bin_edges.tolist(), 'counts': counts.tolist()}}
    except Exception as e:
        return {'var': 0.0, 'hist': {'bins': [], 'counts': []}, 'error': str(e)}
