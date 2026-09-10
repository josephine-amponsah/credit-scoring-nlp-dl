import dash
from dash import Dash, html, dcc, Input, Output, callback, State
import plotly.express as px
from dash import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
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

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[
                dbc.themes.CYBORG, dbc.icons.BOOTSTRAP, dbc_css, dbc.icons.BOOTSTRAP, dbc.icons.FONT_AWESOME])
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
    dcc.Store(id ="sales-store", data = app_data()),
    dbc.Row([
            html.Nav([
                html.Div([
                    html.A("Portfolio Monitor", className = "navbar-brand txt-info"),
                    html.Div([
                        html.Ul([
                        html.Li([
                            html.A( "Insights", className ="nav-link", href= dash.page_registry['pages.overview']['path'] ) 
                            ], className="nav-item"),
                        html.Li(
                                [
                            html.A("NPV Analysis", className ="nav-link", href= dash.page_registry['pages.risk_segments']['path'])
                                ], className = "nav-item"),
                        html.Li(
                                [
                            html.A("Survival Analysis", className ="nav-link", href= dash.page_registry['pages.survival_analysis']['path'])
                                ], className = "nav-item"),
                        html.Li(
                                [
                             html.A("Model Backtesting", className ="nav-link", href= dash.page_registry['pages.npv_backtesting']['path'])
                                ], className = "nav-item") 
                        #,
                        # html.Li(
                        #         [
                        #     html.A("Reports", className ="nav-link", href= dash.page_registry['pages.documentation']['path'])
                        #         ], className = "nav-item"),
                    ], className="navbar-nav me-auto"
                    )
                    ], className = "collapse navbar-collapse" ,id="navbarColor02"),
                ], className= "container-fluid"),
            ],
            className = "navbar navbar-expand-lg navbar-dark bg-dark"),
            html.Br(),
            dbc.Row([dash.page_container], className = 'page-space')
    ], className = "")
]
)


if __name__ == '__main__':
    # model = joblib.load("modules/forecaster.pkl")
    app.run(debug=True, host='127.0.0.1', port=8080)