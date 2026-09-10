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
import requests

from flask_caching import Cache
sys.path.insert(0, '../modules')

# app = Dash(__name__)
dash.register_page(__name__, path = "/")

layout = html.Div([
    dcc.Store(id="sales-data"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(placeholder='Select period', id='date-time-filter', className="dbc year-dropdown .Select-control", value=None)
        ], width=3),
        dbc.Col([
            dcc.Dropdown(placeholder='Loan purpose', id='loan-purpose-filter', multi=True)
        ], width=4),
        dbc.Col([
            html.Button('Download Risk Report', id='downloader', className="btn btn-info")
        ], width=2),
    ], justify='end'),

    html.Br(),

    # summary cards
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Total Lent", className="card-title"), html.H4(id='card-total-lent')])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Repayments", className="card-title"), html.H4(id='card-repaid')])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Default Rate", className="card-title"), html.H4(id='card-default-rate')])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Interest Earned", className="card-title"), html.H4(id='card-interest')])]), width=3),
    ], className='mb-3'),

    # bar chart
    dbc.Row([
        dbc.Col(dcc.Graph(id='overview-bar-chart'), width=12)
    ]),

    html.Br()
])

# API base (browser will call this path). Adjust via env if needed.
# The FastAPI app exposes routes at the root (e.g. /overview/...),
# so the Dash frontend should point to the API root (no extra /api prefix).
API_BASE = os.environ.get('API_BASE', 'http://127.0.0.1:8050')




# Summary cards: call API for totals
@callback(
    [Output('card-total-lent', 'children'), Output('card-repaid', 'children'), Output('card-default-rate', 'children'), Output('card-interest', 'children')],
    [Input('date-time-filter', 'value'), Input('loan-purpose-filter', 'value')]
)
def update_summary(period, loan_purposes):
    params = {}
    if period:
        params['period'] = period
    if loan_purposes:
        params['loan_purpose'] = ','.join(loan_purposes) if isinstance(loan_purposes, (list, tuple)) else str(loan_purposes)
    try:
        resp = requests.get(f"{API_BASE}/overview/summary", params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        total = numerize.numerize(payload.get('total_lent', 0))
        repaid = numerize.numerize(payload.get('repaid', 0))
        default_rate = f"{payload.get('default_rate', 0):.2%}"
        interest = numerize.numerize(payload.get('interest_earned', 0))
        return total, repaid, default_rate, interest
    except Exception:
        return "-", "-", "-", "-"


# Bar chart aggregates
@callback(
    Output('overview-bar-chart', 'figure'),
    [Input('date-time-filter', 'value'), Input('loan-purpose-filter', 'value')]
)
def update_bar(period, loan_purposes):
    params = {}
    if period:
        params['period'] = period
    if loan_purposes:
        params['loan_purpose'] = ','.join(loan_purposes) if isinstance(loan_purposes, (list, tuple)) else str(loan_purposes)
    try:
        resp = requests.get(f"{API_BASE}/overview/aggregates", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        if not data:
            return go.Figure()
        df = pd.DataFrame(data)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Lent', x=df['loan_purpose'], y=df['lent']))
        fig.add_trace(go.Bar(name='Repaid', x=df['loan_purpose'], y=df['repaid']))
        fig.add_trace(go.Bar(name='Interest', x=df['loan_purpose'], y=df['interest']))
        fig.update_layout(barmode='group', xaxis_title='Loan Purpose', yaxis_title='Amount')
        return fig
    except Exception:
        return go.Figure()


# Populate loan purpose options from API; default select all
@callback(
    Output('loan-purpose-filter', 'options'),
    Output('loan-purpose-filter', 'value'),
    Input('date-time-filter', 'value')
)
def load_loan_purposes(period):
    params = {}
    if period:
        params['period'] = period
    try:
        resp = requests.get(f"{API_BASE}/overview/options", params=params, timeout=10)
        resp.raise_for_status()
        opts = resp.json().get('loan_purposes', [])
        options = [{'label': o, 'value': o} for o in opts]
        values = [o['value'] for o in options]
        return options, values
    except Exception:
        return [], []


# Populate period dropdown on page load (uses the API to list available quarters)
@callback(
    Output('date-time-filter', 'options'),
    Output('date-time-filter', 'value'),
    Input('sales-data', 'data')
)
def load_periods(_store_data):
    try:
        resp = requests.get(f"{API_BASE}/overview/periods", timeout=10)
        resp.raise_for_status()
        periods = resp.json().get('periods', [])
        options = [{'label': p, 'value': p} for p in periods]
        value = periods[-1] if periods else None
        return options, value
    except Exception:
        return [], None


