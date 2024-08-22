# package imports
import dash
import dash_bootstrap_components as dbc
from dash import html

# local imports

dash.register_page(__name__)
# login screen
layout = dbc.Row(
    dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader('Register'),
                dbc.CardBody(
                    [
                        dbc.Input(
                            placeholder='Username',
                            type='text',
                            id='register-username',
                            class_name='mb-2'
                        ),
                        dbc.Input(
                            placeholder='Password',
                            type='password',
                            id='register-password',
                            class_name='mb-2'
                        ),
                        dbc.Button(
                            'Register',
                            n_clicks=0,
                            type='submit',
                            id='register-button',
                            class_name='float-end'
                        ),
                        html.Div(children='', id='register-feedback')
                    ]
                )
            ]
        ),
        md=6,
        lg=4,
        xxl=3,
    ),
    justify='center'
)
