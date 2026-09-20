"""Portfolio risk metrics: expected loss, survival analysis, and loss simulation.

Survival analysis section
--------------------------
Given a loan-level DataFrame with `issue_d`, `loan_status`, and (for defaulted
loans) `last_pymnt_d`, this module builds a duration/event-flag representation
suitable for time-to-default modelling, then exposes:

- `kaplan_meier_survival` : non-parametric baseline survival curve S(t)
- `cox_hazard`            : covariate-adjusted proportional hazards model
- `survival`              : convenience wrapper combining both

Censoring convention
---------------------
A loan is treated as an "event" (default) if its status matches a default-like
keyword (charged off, default, late, grace period, etc). For an event, the
observation ends at `last_pymnt_d` (the last point we know the loan was still
alive). For a non-event (current/paid-off) loan, the observation is
right-censored at "now" -- we only know it survived at least that long.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAYS_PER_MONTH = 30.4375  # average calendar month length, accounts for leap years

DEFAULT_STATUS_KEYWORDS = (
    "charged off",
    "default",
    "late (",
    "late",
    "grace",
    "does not meet",
)

DEFAULT_COX_COVARIATES = (
    "credit_score",
    "ltv",
    "dti",
    "loan_amount",
    "funded_amnt",
    "annual_inc",
    "int_rate",
)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _finite_or_none(value: Any) -> Optional[float]:
    """Cast to float, returning None for NaN/inf/unparseable values."""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if np.isfinite(fv) else None


def _finite_list(values: Sequence[Any]) -> List[Optional[float]]:
    """Element-wise `_finite_or_none` over an iterable."""
    return [_finite_or_none(v) for v in values]


def _empty_survival_result() -> Dict[str, Any]:
    """Shared shape for callers when there isn't enough data to fit a curve."""
    return {
        "months": [],
        "survival": [],
        "events": [],
        "at_risk": [],
        "median_months": None,
        "n_loans": 0,
    }


# ---------------------------------------------------------------------------
# duration / event construction (shared by KM, Cox, and the legacy fallback)
# ---------------------------------------------------------------------------

def is_default_status(status: Any) -> bool:
    """True if a loan_status value indicates a default-like event."""
    if pd.isna(status):
        return False
    value = str(status).strip().lower()
    if not value:
        return False
    return any(keyword in value for keyword in DEFAULT_STATUS_KEYWORDS)


# Backwards-compatible alias: keep the original private name working in case
# other modules import it directly.
_default_event_flag = lambda status: int(is_default_status(status))


def _event_flags(df: pd.DataFrame) -> pd.Series:
    """1/0 event indicator per row, defaulting to "no event" if status is absent."""
    if "loan_status" not in df.columns:
        return pd.Series(0, index=df.index)
    return df["loan_status"].apply(is_default_status).astype(int)


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse to UTC then drop tz info, so mixed tz-aware/naive inputs compare cleanly."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)


def months_since_issue(X: pd.DataFrame) -> pd.Series:
    """Observation window, in months, for each loan.

    Defaulted loans are measured issue_d -> last_pymnt_d (the last date we know
    the loan was alive). All other loans are right-censored at "now". Returns 0
    for rows that can't be parsed rather than dropping them, so the output
    always aligns 1:1 with the input index.
    """
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if df.empty or "issue_d" not in df.columns:
        return pd.Series(dtype=float, index=df.index)

    issue_d = _to_naive_datetime(df["issue_d"])
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()

    reference_dates = pd.Series(now, index=df.index)
    if "last_pymnt_d" in df.columns:
        last_pymnt_d = _to_naive_datetime(df["last_pymnt_d"])
        default_mask = _event_flags(df).astype(bool)
        # only use last_pymnt_d where it's actually populated; otherwise fall
        # back to "now" rather than introducing NaT censoring dates
        usable_last_pymnt = default_mask & last_pymnt_d.notna()
        reference_dates = reference_dates.mask(usable_last_pymnt, last_pymnt_d)

    months = ((reference_dates - issue_d).dt.days / DAYS_PER_MONTH).clip(lower=0)
    return months.fillna(0.0)


@dataclass
class _DurationFrame:
    """Loan-level duration + event data, ready for KM/Cox fitting."""

    duration: pd.Series
    event: pd.Series
    n_loans: int
    frame: pd.DataFrame = field(repr=False)


def _build_duration_frame(X: pd.DataFrame) -> Optional[_DurationFrame]:
    """Parse issue_d, drop unparseable rows, and compute duration/event columns.

    Returns None if there isn't enough information to build a curve at all.
    """
    if X is None or X.empty or "issue_d" not in X.columns:
        return None

    df = X.copy()
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d"])
    if df.empty:
        return None

    duration = months_since_issue(df)
    event = _event_flags(df)
    return _DurationFrame(duration=duration, event=event, n_loans=len(df), frame=df)


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------

def _legacy_survival_curve(dframe: _DurationFrame) -> Dict[str, Any]:
    """Manual Kaplan-Meier calculation, used only if `lifelines` isn't installed."""
    month_index = np.ceil(dframe.duration).astype(int)
    months = sorted(month_index.unique().tolist())

    survival_curve, events_list, at_risk_list = [], [], []
    running_survival = 1.0

    for month in months:
        at_risk = int((month_index >= month).sum())
        if at_risk == 0:
            continue
        events = int(((month_index == month) & (dframe.event == 1)).sum())
        if events > 0:
            running_survival *= 1 - (events / at_risk)
        survival_curve.append(float(running_survival))
        events_list.append(events)
        at_risk_list.append(at_risk)

    median_months = next(
        (float(m) for m, s in zip(months, survival_curve) if s <= 0.5), None
    )

    return {
        "months": [int(m) for m in months],
        "survival": survival_curve,
        "events": events_list,
        "at_risk": at_risk_list,
        "median_months": median_months,
        "n_loans": dframe.n_loans,
    }


def _km_confidence_interval(kmf) -> tuple:
    """Extract (lower, upper) CI bounds from a fitted KaplanMeierFitter, if available."""
    try:
        ci = kmf.confidence_interval_
    except Exception:
        return None, None

    if ci.empty:
        return None, None

    cols = ci.columns
    lower = _finite_list(ci.iloc[:, 0].tolist())
    upper = _finite_list(ci.iloc[:, 1].tolist()) if len(cols) >= 2 else list(lower)
    return lower, upper


def kaplan_meier_survival(X: pd.DataFrame) -> Dict[str, Any]:
    """Non-parametric baseline survival curve S(t) for the given cohort.

    Falls back to a manual KM calculation if `lifelines` is not installed.
    """
    dframe = _build_duration_frame(X)
    if dframe is None:
        return _empty_survival_result()

    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return _legacy_survival_curve(dframe)

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=dframe.duration.to_numpy(),
        event_observed=dframe.event.to_numpy(),
        label="Portfolio",
    )

    survival_table = kmf.survival_function_
    event_table = kmf.event_table
    lower_ci, upper_ci = _km_confidence_interval(kmf)

    median = kmf.median_survival_time_
    return {
        "months": _finite_list(survival_table.index.tolist()),
        "survival": _finite_list(survival_table.iloc[:, 0].tolist()),
        "events": event_table["observed"].astype(int).tolist(),
        "at_risk": event_table["at_risk"].astype(int).tolist(),
        "median_months": _finite_or_none(median) if pd.notna(median) else None,
        "n_loans": dframe.n_loans,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }


# ---------------------------------------------------------------------------
# Cox proportional hazards
# ---------------------------------------------------------------------------

def _cox_average_survival_curve(cph, cov_df: pd.DataFrame) -> Optional[Dict[str, List[Optional[float]]]]:
    """Cohort-average predicted survival curve, for plotting alongside KM."""
    if cov_df.empty:
        return None
    try:
        predicted = cph.predict_survival_function(cov_df)
        mean_survival = predicted.mean(axis=1)
    except Exception:
        logger.warning("Cox survival-curve prediction failed", exc_info=True)
        return None

    return {
        "months": _finite_list(mean_survival.index.tolist()),
        "survival": _finite_list(mean_survival.values.tolist()),
    }


def cox_hazard(X: pd.DataFrame, covariates: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Fit a Cox proportional hazards model and return hazard ratios + summary.

    `covariates` defaults to whichever of DEFAULT_COX_COVARIATES are present
    in `X`. Rows with missing values in duration/event/covariates are dropped
    (Cox fitting requires complete cases).
    """
    empty_result = {"hazard_ratios": {}, "summary": None, "n_loans": 0}

    dframe = _build_duration_frame(X)
    if dframe is None:
        return empty_result

    covariates = list(covariates) if covariates is not None else [
        col for col in DEFAULT_COX_COVARIATES if col in dframe.frame.columns
    ]

    cph_df = dframe.frame.assign(duration=dframe.duration, event=dframe.event)
    cph_df = cph_df[["duration", "event", *covariates]].dropna()

    if len(cph_df) < 2:
        return {**empty_result, "n_loans": dframe.n_loans}

    try:
        from lifelines import CoxPHFitter
    except ImportError:
        return {
            **empty_result,
            "n_loans": dframe.n_loans,
            "fallback": "lifelines not installed; Cox model unavailable",
        }

    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col="duration", event_col="event")

    return {
        "hazard_ratios": cph.hazard_ratios_.to_dict(),
        "summary": cph.summary.to_dict(),
        "n_loans": len(cph_df),
        "survival_curve": _cox_average_survival_curve(cph, cph_df[covariates]),
    }


def survival(X: pd.DataFrame) -> Dict[str, Any]:
    """Kaplan-Meier curve for the cohort, enriched with Cox output when available.

    The Cox fit is best-effort: if it fails (e.g. too few events, collinear
    covariates), the KM curve is still returned on its own.
    """
    result = kaplan_meier_survival(X)

    try:
        cox_out = cox_hazard(X)
    except Exception:
        logger.warning("Cox hazard model failed; returning KM curve only", exc_info=True)
        return result

    if cox_out.get("survival_curve"):
        result["cox_survival"] = cox_out["survival_curve"]
        result["cox_hazard_ratios"] = cox_out.get("hazard_ratios", {})

    return result


# ---------------------------------------------------------------------------
# expected loss / simulation (unchanged from original)
# ---------------------------------------------------------------------------

def expected_loss(X: pd.DataFrame):
    """Compute expected loss given a DataFrame with a `prob_default` column.

    EL is computed as min-max-scaled probability_of_default * exposure
    (funded_amnt). Returns total expected loss and the per-loan EL series.
    """
    df = X.copy()
    raw_probs = df["prob_default"].astype(float).fillna(0.0) if "prob_default" in df.columns else pd.Series(0.0, index=df.index)
    lo, hi = raw_probs.min(), raw_probs.max()
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        scaled = (raw_probs - lo) / (hi - lo)
    else:
        scaled = raw_probs.clip(0.0, 1.0)
    df["scaled_probs"] = scaled

    exposure = df.get("funded_amnt", pd.Series([0] * len(df))).astype(float).fillna(0.0)
    el = df["scaled_probs"] * exposure
    return float(el.sum()), el


def npv(X):
    return None


def simulate_portfolio_losses(df: pd.DataFrame, nsim: int = 2000, seed: Optional[int] = None, batch_size: int = 500) -> np.ndarray:
    """Monte Carlo simulation of portfolio-level losses via independent Bernoulli defaults."""
    rnd = np.random.RandomState(seed)
    raw = df["prob_default"].astype(float).fillna(0.0) if "prob_default" in df.columns else pd.Series(0.0, index=df.index)
    lo, hi = raw.min(), raw.max()
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        probs = np.clip(((raw - lo) / (hi - lo)).values, 0.0, 1.0)
    else:
        probs = np.clip(raw.values, 0.0, 1.0)

    exposure = df.get("funded_amnt", pd.Series([0] * len(df))).astype(float).fillna(0.0).values
    n = len(probs)
    if n == 0:
        return np.array([])

    port_losses = np.empty(nsim)
    for start in range(0, nsim, batch_size):
        end = min(start + batch_size, nsim)
        draws = rnd.rand(end - start, n)  # small chunk, not the whole nsim
        defaults = draws < probs
        port_losses[start:end] = (defaults * exposure).sum(axis=1)
        del draws, defaults  # free immediately
    return port_losses


def var_from_simulations(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Portfolio VaR at the given confidence level from simulated losses."""
    if losses is None or len(losses) == 0:
        return 0.0
    return float(np.quantile(losses, confidence))