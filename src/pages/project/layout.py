import logging

import dash
import dash_bootstrap_components as dbc
from dash import html

from .callbacks import scan_projects

logger = logging.getLogger(__name__)

dash.register_page(
    __name__,
    path='/project',
    title='Project | YAPAT'
)

layout = dbc.Container([
    html.Div([
        # Header
        dbc.Row(
            # ['Header']

        ),
        # Main
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    html.H5('Select project'),
                    dbc.RadioItems(
                        id='project-list',
                        options=scan_projects(),
                    ),
                ], class_name='my-4'),
                dbc.Row([
                    html.Div([
                        dbc.Button('Add', id='collapse-button', class_name='my-4'),
                        dbc.Collapse([
                            html.Div([
                                dbc.FormFloating([
                                    dbc.Input(placeholder="Type name of new project...", type="text",
                                              id="project-name"),
                                    dbc.Label("Project name"),
                                    dbc.FormFeedback('This name is already in use. Please select another one.',
                                                     type='invalid')
                                ], class_name='my-1'),
                            ]),
                            html.Div([
                                dbc.FormFloating([
                                    dbc.Input(placeholder="Type path to directory containing audio files...",
                                              type="text",
                                              id='audio-path'),
                                    dbc.Label("Path to audio directory"),
                                    dbc.FormFeedback('Not a valid directory path', type='invalid')
                                ], class_name='py-1'),
                            ]),
                            dbc.Select(
                                options=[
                                    {'label': 'Birdnet 2.4 (recommended)', 'value': 'birdnet'},
                                    {'label': 'Perch', 'value': 'perch', 'disabled': True},
                                    {'label': 'VAE', 'value': 'vae', 'disabled': True},
                                ],
                                placeholder="Select embedding model", class_name='my-1', id='embedding-model'
                            ),
                            dbc.ButtonGroup([
                                dbc.Button('Cancel', color='danger', id='button-project-cancel', class_name='mx-1'),
                                dbc.Button('Create', id='button-project-create', class_name='mx-1')
                            ], class_name='my-2')],
                            id='collapse',
                            is_open=False
                        )
                    ]),
                ], class_name='my-4')
            ], width=3),
            dbc.Col([
                dbc.Row([
                    html.H5('Project summary'),
                ], class_name='my-4', id='project-summary'),
            ])
        ])
    ])
])
