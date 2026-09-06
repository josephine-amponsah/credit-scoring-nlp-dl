Credit Portfolio Risk Monitoring

Purpose
This project implements a monitoring and analytics dashboard for credit portfolios. Its primary analyses include portfolio Value-at-Risk (VaR), Expected Loss (EL), survival analysis for lifecycle risk, Net Present Value (NPV) of cashflows, and model backtesting for scorecard and ML models.

Key features
- Portfolio VaR: scenario-based and historical VaR at configurable confidence levels, with component contributions by segment.
- Expected Loss: forward-looking EL estimates built from PD, LGD and EAD pipelines, including cohort aggregation.
- Survival analysis: Kaplan–Meier and Cox-style analyses to estimate time-to-default distributions and hazard functions.
- NPV analysis: discounted cashflow valuation for loan vintages and portfolios, with sensitivity scans over discount curves.
- Backtesting and calibration: rolling-window backtests, calibration plots, PSI/Tables and performance summaries for model governance.
- Explainability: SHAP-based local and global attributions for model outputs to support decisions and audits.
- Dashboard visualizations: interactive charts and tables (Plotly) for drill-down, cohort comparisons, and exportable reports.

Architecture
1. Model training and artifacts
	- Training pipelines produce serialized model artifacts and preprocessing objects (scalers, encoders) stored under `artefacts/`.
	- Supported model types: scikit-learn, LightGBM, XGBoost, PyTorch — adapters standardize input/output contracts (logits, probabilities).

2. API layer
	- A FastAPI application (`api/`) loads model artifacts on startup and exposes JSON endpoints for scoring, portfolio-level metrics, explainability, and backtests.
	- Typical endpoints: `/predict`, `/predict_proba`, `/portfolio/var`, `/portfolio/expected_loss`, `/portfolio/npv`, `/backtest`, `/explain`, `/survival`.

3. Dashboard (Plotly Dash)
	- A Dash application consumes the API endpoints to render interactive visualizations and control parameters (time windows, segments, confidence levels).
	- The Dash app can be embedded or proxied behind the FastAPI service for a single service surface.

Data flow
- Raw data ingestion -> feature engineering -> persistence (intermediate parquet/csv) -> model scoring -> metrics & explanations -> dashboard visualizations and reports.

Deployment & integration notes
- The API serves machine-readable endpoints; the Dash frontend queries these endpoints rather than performing heavy computation in-browser.
- For production, run the FastAPI app with `uvicorn`/Gunicorn and host the Dash app either as a mounted sub-app or behind a reverse proxy.

Tech stack
- Python, FastAPI, Plotly Dash, pandas, NumPy
- scikit-learn, XGBoost, LightGBM, PyTorch
- SHAP for explainability; joblib/torch for model persistence

Outputs and artifacts
- Model artifacts in `artefacts/` (models, scalers, SHAP background arrays)
- Reports and backtest results (CSV/JSON) produced by `modules/` utilities
- Dashboard assets under `app/` for visualization and user interaction

Governance
- Ensure reproducible artifact versions and maintain calibration records for each model release. Follow repository license for redistribution.
