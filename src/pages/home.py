# package imports
import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(
    __name__,
    path='/',
    redirect_from=['/home'],
    title='Home | YAPAT'
)

layout = html.Div(
    [
        html.H1('Welcome to YAPAT'),
        html.Div(
            html.A('Select a project or start a new one here.', href='/project')
        ),
        html.Div(id='content')
    ]
)

