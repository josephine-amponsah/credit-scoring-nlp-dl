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

from flask_caching import Cache
sys.path.insert(0, '../modules')

# app = Dash(__name__)
dash.register_page(__name__, path = "/npv_backtesting")
layout = html.Div([
    dcc.Store(id = "sales-data"),
    dbc.Row([
        dbc.Col([
            html.Label('Period', className='form-label'),
            dcc.Dropdown(
                placeholder = 'Select period',
                id = 'date-time-filter',
                clearable=True,
                searchable=True,
                style={'borderRadius': '0.5rem'},
                value = None
            )
        ], width=3),
        dbc.Col([
            html.Label('Actions', className='form-label'),
            dbc.Button('Download Risk Report', id = 'downloader', color='primary', className='w-100')
        ], width = 2),
    ], align='end', className='g-3 mb-3'),
    html.Br()
])