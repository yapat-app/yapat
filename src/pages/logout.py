# package imports
import dash
from dash import html, dcc

dash.register_page(__name__)

layout = html.Div(
    [
        html.Div(html.H2('You have been logged out')),
        html.Br(),
        dcc.Link('Log in', href='/login'),
    ]
)
