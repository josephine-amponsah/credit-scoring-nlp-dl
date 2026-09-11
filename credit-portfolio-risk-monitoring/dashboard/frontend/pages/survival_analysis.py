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

dash.register_page(__name__, path="/survival_analysis")

API_BASE = os.environ.get('API_BASE', 'http://127.0.0.1:8050')

layout = html.Div([
    dcc.Store(id='survival-store'),
    dbc.Row([
        dbc.Col([
            html.Label('Period', className='form-label'),
            dcc.Dropdown(
                placeholder='Select period',
                id='survival-period-filter',
                clearable=True,
                searchable=True,
                style={'borderRadius': '0.5rem'}
            )
        ], width=3),
        dbc.Col([
            html.Label('Loan purpose', className='form-label'),
            dcc.Dropdown(
                placeholder='Select loan purpose',
                id='survival-purpose-filter',
                multi=True,
                clearable=True,
                searchable=True,
                style={'borderRadius': '0.5rem'}
            )
        ], width=4),
        dbc.Col([
            html.Label('Actions', className='form-label'),
            dbc.Button('Refresh', id='survival-refresh', color='primary', className='w-100')
        ], width=2),
    ], align='end', className='g-3 mb-3'),
    html.Br(),

    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H6('Cohort Size', className='card-title'), html.H4(id='survival-cohort-size', className='mb-0')])], className='h-100 shadow-sm border-0'), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6('Median Survival', className='card-title'), html.H4(id='survival-median', className='mb-0')])], className='h-100 shadow-sm border-0'), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6('Latest Survival', className='card-title'), html.H4(id='survival-latest', className='mb-0')])], className='h-100 shadow-sm border-0'), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6('Event Rate', className='card-title'), html.H4(id='survival-event-rate', className='mb-0')])], className='h-100 shadow-sm border-0'), width=3),
    ], className='g-3 mb-3'),

    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='survival-chart', config={'displayModeBar': False}, style={'height': '360px'}))], className='shadow-sm border-0'), width=12)
    ]),
    html.Br()
])


@callback(
    Output('survival-period-filter', 'options'),
    Output('survival-period-filter', 'value'),
    Input('survival-store', 'data')
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


@callback(
    Output('survival-purpose-filter', 'options'),
    Output('survival-purpose-filter', 'value'),
    Input('survival-period-filter', 'value')
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


@callback(
    [
        Output('survival-cohort-size', 'children'),
        Output('survival-median', 'children'),
        Output('survival-latest', 'children'),
        Output('survival-event-rate', 'children'),
        Output('survival-chart', 'figure')
    ],
    [Input('survival-period-filter', 'value'), Input('survival-purpose-filter', 'value')],
    prevent_initial_call=False
)
def update_survival(period, loan_purposes):
    params = {}
    if period:
        params['period'] = period
    if loan_purposes:
        params['loan_purpose'] = ','.join(loan_purposes) if isinstance(loan_purposes, (list, tuple)) else str(loan_purposes)

    try:
        resp = requests.get(f"{API_BASE}/survival", params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        months = payload.get('months', [])
        surv = payload.get('survival', [])
        events = payload.get('events', [])
        n_loans = payload.get('n_loans', 0)
        median_months = payload.get('median_months')

        if not months or not surv:
            fig = go.Figure()
            return '-', '-', '-', '-', fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=surv, mode='lines+markers', name='Survival', line=dict(color='#18BC9C', width=3)))
        fig.update_layout(
            xaxis_title='Months since origination',
            yaxis_title='Survival probability',
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode='x unified',
            template='flatly'
        )

        latest = surv[-1] if surv else 0.0
        total_events = sum(events) if events else 0
        event_rate = (total_events / max(1, n_loans)) if n_loans else 0.0

        return (
            f"{n_loans}",
            f"{median_months:.1f} mo" if median_months is not None else '—',
            f"{latest:.1%}",
            f"{event_rate:.1%}",
            fig,
        )
    except Exception:
        fig = go.Figure()
        return '-', '-', '-', '-', fig
