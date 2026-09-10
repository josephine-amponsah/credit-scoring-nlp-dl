import dash
from dash import Dash, html, dcc, Input, Output, callback, State
import plotly.express as px
from dash import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import sys
import os
import requests
import json
import numerize
from numerize import numerize

from flask_caching import Cache
sys.path.insert(0, '../modules')

# app = Dash(__name__)
dash.register_page(__name__, path = "/risk_segments")

API_BASE = os.environ.get('API_BASE', 'http://127.0.0.1:8050')


# Combined layout: include the date filter row, then summary cards and chart
layout = html.Div([
    dcc.Store(id='sales-data'),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(placeholder='Select period', id='date-time-filter', className="dbc year-dropdown .Select-control", value=None)
        ], width=3),
        dbc.Col([
            html.Button('Download Risk Report', id='downloader', className="btn btn-info")
        ], width=2),
    ], justify='end'),
    html.Br(),

    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Expected Loss", className="card-title"), html.H4(id='card-expected-loss')])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("VaR (95%)", className="card-title"), html.H4(id='card-var')])]), width=3),
    ], className='mb-3'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='risk-dist-chart'), width=12)
    ]),
    html.Br()
])


@callback(
    [Output('card-expected-loss', 'children'), Output('card-var', 'children'), Output('risk-dist-chart', 'figure')],
    [Input('date-time-filter', 'value')]
)
def update_risk_segments(period):
    params = {}
    if period:
        params['period'] = period
    try:
        # expected loss
        resp = requests.get(f"{API_BASE}/risk/expected_loss", params=params, timeout=20)
        resp.raise_for_status()
        el_payload = resp.json()
        expected_loss = el_payload.get('total_expected_loss', 0.0)
        n_loans = el_payload.get('n_loans', 0)
        # VaR + hist
        resp2 = requests.get(f"{API_BASE}/risk/var", params={**params, 'confidence': 0.95, 'nsim': 3000}, timeout=60)
        resp2.raise_for_status()
        var_payload = resp2.json()
        var_val = var_payload.get('var', 0.0)
        hist = var_payload.get('hist', {})
        bins = hist.get('bins', [])
        counts = hist.get('counts', [])
        # build histogram figure
        fig = go.Figure()
        if bins and counts:
            # bins are edges; compute centers
            import numpy as _np
            centers = 0.5 * (_np.array(bins[:-1]) + _np.array(bins[1:]))
            fig.add_trace(go.Bar(x=centers, y=counts, name='Simulated Losses'))
            # VaR line
            fig.add_vline(x=var_val, line_dash='dash', line_color='red', annotation_text=f'VaR={var_val:,.0f}', annotation_position='top right')
            fig.update_layout(xaxis_title='Portfolio Loss', yaxis_title='Counts')
        return f"{expected_loss:,.0f}", f"{var_val:,.0f}", fig
    except Exception:
        return '-', '-', go.Figure()
    except Exception:
        return '-', '-', go.Figure()

