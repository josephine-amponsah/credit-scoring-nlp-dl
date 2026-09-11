import dash
from dash import Dash, html, dcc, Input, Output, callback, State
import plotly.express as px
from dash import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
import sys
import json
import numerize
from numerize import numerize
import os
import io
import requests
import functools

from flask_caching import Cache
sys.path.insert(0, '../modules')
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.css"

pio.templates["flatly"] = go.layout.Template(
    layout=dict(
        font=dict(color="#2C3E50", family="Open Sans, sans-serif"),
        paper_bgcolor="#F8F9FA",
        plot_bgcolor="#F8F9FA",
        title_font=dict(color="#2C3E50"),
        legend=dict(title_font=dict(color="#2C3E50"), font=dict(color="#2C3E50")),
        colorway=["#18BC9C", "#2C3E50", "#3498DB", "#F39C12", "#E74C3C", "#95A5A6"],
        xaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#CBD5E1", tickfont=dict(color="#2C3E50"), title_font=dict(color="#2C3E50")),
        yaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#CBD5E1", tickfont=dict(color="#2C3E50"), title_font=dict(color="#2C3E50")),
        margin=dict(l=40, r=20, t=40, b=40),
    )
)
pio.templates.default = "flatly"

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[
                dbc.themes.FLATLY, dbc.icons.BOOTSTRAP, dbc_css, dbc.icons.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .navbar {
                --bs-navbar-color: rgba(255,255,255,0.8);
                --bs-navbar-hover-color: rgba(255,255,255,1);
                --bs-navbar-active-color: #18bc9c;
            }
            .navbar .nav-link {
                color: rgba(255,255,255,0.8) !important;
                border-radius: 0.35rem;
                padding: 0.55rem 0.85rem;
                transition: color 0.2s ease-in-out, background-color 0.2s ease-in-out;
            }
            .navbar .nav-link:hover,
            .navbar .nav-link:focus {
                color: #ffffff !important;
                background-color: rgba(255,255,255,0.08);
            }
            .navbar .nav-link.active {
                color: #18bc9c !important;
                background-color: rgba(255,255,255,0.04);
            }
            .page-shell {
                margin-top: 0.75rem;
                padding: 1.25rem 1.25rem 0.5rem;
                background: #f8f9fa;
                border-radius: 0.5rem;
                border: 1px solid rgba(52, 73, 94, 0.1);
            }
            .page-shell .form-label {
                color: #2c3e50;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                text-transform: uppercase;
            }
            .page-shell .card {
                border: 1px solid rgba(44, 62, 80, 0.12);
                background-color: #ffffff;
                box-shadow: 0 0.125rem 0.5rem rgba(44, 62, 80, 0.06);
            }
            .page-shell .card-title {
                color: #7b8a8b;
                font-size: 0.75rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .page-shell .card h4 {
                color: #2c3e50;
            }
            body {
                background-color: #f5f7fa;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
server = app.run


data_url = "https://github.com/josephine-amponsah/credit-scoring-nlp-dl/tree/main/credit-portfolio-risk-monitoring/notebooks/data_splits"
timeout = 20

# Use simple in-memory cache by default to avoid filesystem backend import issues.
# For production, switch to a persistent backend (Redis, Memcached) and update config.
# Initialize Flask-Caching if available and configured correctly; fall back to
# an in-process LRU cache if the backend is not importable on this system.
try:
    cache = Cache(server, config={
        'CACHE_TYPE': os.environ.get('CACHE_TYPE', 'simple')
    })
except Exception as e:
    # avoid failing app startup due to missing cache backends
    print("[warning] flask-caching init failed, falling back to lru cache:", e, file=sys.stderr)
    cache = None

# Ensure `cache.memoize` exists so code using it as a decorator doesn't crash.
if cache is None:
    class _CacheStub:
        def memoize(self, timeout=None):
            def _decorator(func):
                return functools.lru_cache(maxsize=32)(func)
            return _decorator
    cache = _CacheStub()

def _app_data_impl(period: str | None = None):
    try:
        if not (data_url.startswith('https://github.com/') and '/tree/' in data_url):
            return json.dumps({})
        owner_repo, rest = data_url.replace('https://github.com/', '').split('/tree/', 1)
        branch, _, path = rest.partition('/')

        api_url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
        resp = requests.get(api_url, params={'ref': branch}, timeout=10)
        if resp.status_code != 200:
            return json.dumps({})
        items = resp.json()
        names = [it['name'] for it in items if it.get('name', '').startswith('data_')]
        if not names:
            return json.dumps({})
        if not period:
            period = sorted([n.split('data_')[1].split('.')[0] for n in names])[-1]

        raw_base = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{path}".rstrip('/')
        for name in (f"data_{period}.parquet", f"data_{period}.csv.gz", f"data_{period}.csv"):
            r = requests.get(f"{raw_base}/{name}", timeout=10)
            if r.status_code != 200:
                continue
            try:
                if name.endswith('.parquet'):
                    df = pd.read_parquet(io.BytesIO(r.content))
                else:
                    df = pd.read_csv(io.BytesIO(r.content), parse_dates=['issue_d'], low_memory=False)
                for c in df.select_dtypes(include=['object']).columns:
                    df[c] = df[c].apply(lambda v: None if pd.isna(v) else (v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v)))
                return df.to_json(date_format='iso')
            except Exception:
                continue
        return json.dumps({})
    except Exception:
        return json.dumps({})


# Apply caching: prefer Flask-Caching if initialized, otherwise use functools.lru_cache
if cache:
    app_data = cache.memoize(timeout=timeout)(_app_data_impl)
else:
    app_data = functools.lru_cache(maxsize=32)(_app_data_impl)
    
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id ="sales-store", data = app_data()),
    dbc.Row([
            html.Nav([
                html.Div([
                    html.A("Portfolio Monitor", className = "navbar-brand"),
                    html.Div([
                        html.Ul([
                        html.Li([
                            html.A("Insights", id="overview-link", className="nav-link", href=dash.page_registry['pages.overview']['path'])
                            ], id="overview-item", className="nav-item"),
                        html.Li(
                                [
                            html.A("NPV Analysis", id="risk-segments-link", className="nav-link", href=dash.page_registry['pages.risk_segments']['path'])
                                ], id="risk-segments-item", className = "nav-item"),
                        html.Li(
                                [
                            html.A("Survival Analysis", id="survival-analysis-link", className="nav-link", href=dash.page_registry['pages.survival_analysis']['path'])
                                ], id="survival-analysis-item", className = "nav-item"),
                        html.Li(
                                [
                             html.A("Model Backtesting", id="backtesting-link", className="nav-link", href=dash.page_registry['pages.npv_backtesting']['path'])
                                ], id="backtesting-item", className = "nav-item") 
                        #,
                        # html.Li(
                        #         [
                        #     html.A("Reports", className ="nav-link", href= dash.page_registry['pages.documentation']['path'])
                        #         ], className = "nav-item"),
                    ], className="navbar-nav me-auto"
                    )
                    ], className = "collapse navbar-collapse" ,id="navbarColor01"),
                ], className= "container-fluid"),
            ],
            className = "navbar navbar-expand-lg bg-primary",
            **{"data-bs-theme": "dark"}),
            html.Br(),
            html.Div(dbc.Container(dash.page_container, fluid=True), className="page-shell px-3")
    ], className = "")
]
)


@app.callback(
    Output("overview-item", "className"),
    Output("risk-segments-item", "className"),
    Output("survival-analysis-item", "className"),
    Output("backtesting-item", "className"),
    Output("overview-link", "className"),
    Output("risk-segments-link", "className"),
    Output("survival-analysis-link", "className"),
    Output("backtesting-link", "className"),
    Input("url", "pathname"),
)
def update_nav_classes(pathname):
    current_path = pathname or "/"

    def item_class(target):
        return "nav-item active" if current_path == target else "nav-item"

    def link_class(target):
        return "nav-link active" if current_path == target else "nav-link"

    return (
        item_class(dash.page_registry['pages.overview']['path']),
        item_class(dash.page_registry['pages.risk_segments']['path']),
        item_class(dash.page_registry['pages.survival_analysis']['path']),
        item_class(dash.page_registry['pages.npv_backtesting']['path']),
        link_class(dash.page_registry['pages.overview']['path']),
        link_class(dash.page_registry['pages.risk_segments']['path']),
        link_class(dash.page_registry['pages.survival_analysis']['path']),
        link_class(dash.page_registry['pages.npv_backtesting']['path']),
    )


if __name__ == '__main__':
    # model = joblib.load("modules/forecaster.pkl")
    app.run(debug=True, host='127.0.0.1', port=8080)