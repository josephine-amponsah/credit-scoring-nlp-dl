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
dash.register_page(__name__, path = "/")

layout = html.Div([
    dcc.Store(id = "sales-data"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown( placeholder = 'Year', id = 'date-time-filter',
                 className="dbc year-dropdown .Select-control", value = 2021) 
            ], width=2),
        dbc.Col([
            html.Button( 'Download Risk Report', id = 'downloader',
                 className="btn btn-info") 
            ], width = 2),
    ], justify= 'end'),
    html.Br()
])

@callback(
    [Output("highest-cat-name", "children"), Output("highest-cat-orders", "children")], 
    # Input("date-time-filter", "value"),
    Input("sales-data", "data")
)
def cat_stats(data):
    data = pd.DataFrame(json.loads(data))
    # data = data[data["year"] == year]
    high_cat = data[["Product_Category", "Order_Demand"]]
    high_cat = high_cat.groupby(["Product_Category"]).sum("Order_Demand")
    high_cat_name= high_cat.idxmax()
    high_cat_orders = numerize.numerize(high_cat["Order_Demand"].max())
    return [high_cat_name, high_cat_orders]