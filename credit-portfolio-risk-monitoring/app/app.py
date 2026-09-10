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
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.css"

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[
                dbc.themes.CYBORG, dbc.icons.BOOTSTRAP, dbc_css, dbc.icons.BOOTSTRAP, dbc.icons.FONT_AWESOME])
server = app.server