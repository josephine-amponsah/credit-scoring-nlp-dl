import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / "credit-portfolio-risk-monitoring" / "dashboard" / "backend"

# Ensure the backend root is on `sys.path` before importing the app package.
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.services import risk_metrics


def test_months_since_issue_uses_last_pymnt_for_defaults():
    df = pd.DataFrame(
        {
            "issue_d": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "loan_status": ["Current", "Charged Off"],
            "last_pymnt_d": [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-03-15")],
        }
    )

    months = risk_metrics.months_since_issue(df)

    assert months.iloc[0] > 30
    assert months.iloc[1] < 3


def test_kaplan_meier_and_cox_survival_methods():
    df = pd.DataFrame(
        {
            "issue_d": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-02-01"),
                pd.Timestamp("2023-03-01"),
                pd.Timestamp("2023-04-01"),
            ],
            "loan_status": ["Current", "Charged Off", "Current", "Charged Off"],
            "last_pymnt_d": [
                pd.Timestamp("2024-03-15"),
                pd.Timestamp("2023-06-15"),
                pd.Timestamp("2024-02-15"),
                pd.Timestamp("2023-08-15"),
            ],
            "credit_score": [730, 680, 710, 660],
            "ltv": [0.65, 0.72, 0.68, 0.8],
            "dti": [0.22, 0.35, 0.27, 0.4],
        }
    )

    km = risk_metrics.kaplan_meier_survival(df)
    assert km["n_loans"] == 4
    assert len(km["months"]) > 0
    assert km["survival"][0] == 1.0

    cox = risk_metrics.cox_hazard(df)
    assert "hazard_ratios" in cox
    assert len(cox["hazard_ratios"]) >= 1
