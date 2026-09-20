import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from services.api_client import get_periods, get_options, get_survival

dash.register_page(__name__, path="/survival_analysis")

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
        periods = get_periods().get('periods', [])
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
    try:
        opts = get_options(period).get('loan_purposes', [])
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
    try:
        payload = get_survival(period, loan_purposes)
        months = payload.get('months', [])
        surv = payload.get('survival', [])
        events = payload.get('events', [])
        n_loans = payload.get('n_loans', 0)
        median_months = payload.get('median_months')

        if not months or not surv:
            fig = go.Figure()
            return '-', '-', '-', '-', fig

        fig = go.Figure()
        # plot stepwise Kaplan-Meier survival curve
        fig.add_trace(go.Scatter(x=months, y=surv, mode='lines', name='Kaplan-Meier', line=dict(color='#18BC9C', width=3), hoverinfo='x+y', line_shape='hv'))
        # add confidence interval shading if present
        lower = payload.get('lower_ci')
        upper = payload.get('upper_ci')
        if lower and upper and len(lower) == len(months) and len(upper) == len(months):
            # create filled area between upper and lower bounds
            fig.add_trace(go.Scatter(
                x=months + months[::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(24,188,156,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=True,
                name='95% CI'
            ))
        # overlay cohort-averaged Cox predicted survival curve if available
        cox = payload.get('cox_survival')
        if cox and isinstance(cox, dict):
            cox_months = cox.get('months', [])
            cox_surv = cox.get('survival', [])
            if cox_months and cox_surv and len(cox_months) == len(cox_surv):
                fig.add_trace(go.Scatter(x=cox_months, y=cox_surv, mode='lines', name='Cox (cohort mean)', line=dict(color='#F39C12', width=2, dash='dash')))
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
