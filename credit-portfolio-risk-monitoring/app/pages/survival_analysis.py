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
dash.register_page(__name__, path = "/survival_analysis")

layout = html.Div([
    dcc.Store(id = "sales-data"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown( placeholder = 'Select period', id = 'date-time-filter',
                 className="dbc year-dropdown .Select-control", value = None) 
            ], width=2),
        dbc.Col([
            html.Button( 'Download Risk Report', id = 'downloader',
                 className="btn btn-info") 
            ], width = 2),
    ], justify= 'end'),
    html.Br()
])