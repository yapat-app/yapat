import glob
import logging
import os
from multiprocessing import Pool

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import tensorflow as tf
from dash import html, callback, Input, Output, State

from . import get_list_files, split_single_audio, load_audio_files_with_tf_dataset

logger = logging.getLogger(__name__)

dash.register_page(
    __name__,
    path='/project',
    title='Project | YAPAT'
)


def scan_projects():
    options = ([{'label': x, 'value': x} for x in os.listdir('../projects') if
                os.path.isdir(f'../projects/{x}')])
    # + [{'label': 'Add new...', 'value': 'new'}])
    return options


def gen_clips(project_name, audio_path, clip_duration):
    proj_path = f'../projects/{project_name}/clips'
    os.makedirs(proj_path, exist_ok=True)
    list_files = get_list_files(audio_path)
    with Pool() as pool:
        pool.starmap(split_single_audio, [(file, proj_path, clip_duration) for file in list_files])
    return 1


def gen_embeddings(project_name, embedding_model):
    sr = None
    list_files = glob.glob(f'../projects/{project_name}/clips/*.wav')

    if embedding_model == 'birdnet':
        model = tf.keras.layers.TFSMLayer('assets/models/BirdNET-Analyzer-V2.4/V2.4/BirdNET_GLOBAL_6K_V2.4_Model',
                                          call_endpoint='embeddings')
        sr = 48000

    audio_dataset = load_audio_files_with_tf_dataset(list_files, sample_rate=sr)
    results = tf.keras.Sequential([model]).predict(audio_dataset.batch(128), verbose=2)
    df = pd.DataFrame(index=[os.path.basename(file) for file in list_files], data=results['embeddings'])
    df.to_pickle(f'../projects/{project_name}/embeddings.pkl')


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
                        dbc.Collapse(
                            [html.Div([
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


@callback(
    Output('project-summary', 'children'),
    Input('project-list', 'value')
)
def update_project_summary(project_name):
    n_clips = glob.glob(f'../projects/{project_name}/clips/*.wav')
    children = [html.H5('Project summary')]
    if project_name:
        children.append(html.P(f'Found {len(n_clips)} clips, N of which are labelled for M class categories'))
        children.append(html.A("Start annotating", href='/annotate'))
        children.append(html.P('Import project data'))
    return children


@callback(
    Output('button-project-create', 'disabled'),
    Input('project-name', 'valid'),
    Input('audio-path', 'valid'),
    Input('embedding-model', 'valid'),
)
def check_create(v1, v2, v3):
    return not (v1 and v2 and v3)


@callback(
    Output("collapse", "is_open"),
    Input("collapse-button", "n_clicks"),
    Input("button-project-cancel", "n_clicks"),
    State("collapse", "is_open"),
)
def toggle_collapse(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(
    Output('project-name', 'valid'),
    Output('project-name', 'invalid'),
    Input('project-name', 'value')
)
def check_project_name(value):
    if value:
        is_valid = value not in os.listdir('../projects')
        return is_valid, not is_valid
    return False, False


@callback(
    Output('audio-path', 'valid'),
    Output('audio-path', 'invalid'),
    Input('audio-path', 'value')
)
def check_audio_path(value):
    if value:
        is_valid = os.path.isdir(os.path.abspath(value))
        return is_valid, not is_valid
    return False, False


@callback(
    Output('embedding-model', 'valid'),
    Output('embedding-model', 'invalid'),
    Input('embedding-model', 'value')
)
def check_embedding_model(value):
    if value:
        return True, False
    return False, False


@callback(
    Output('project-list', 'options'),
    Output('project-list', 'value'),
    Output('navbar', 'brand'),
    Input('project-list', 'value'),
    Input('button-project-create', 'n_clicks'),
    State('navbar', 'brand'),
    State('project-name', 'value'),
    State('audio-path', 'value'),
    State('embedding-model', 'value'),
)
def update_options_project(project_value, project_create, brand, project_name, audio_path, embedding_model):
    # logger_user_actions.info({
    #     f"triggered_id: {dash.callback_context.triggered_id}, prop_ids: {dash.callback_context.triggered_prop_ids}"})
    logger.info({
        f"triggered_id: {dash.callback_context.triggered_id}, prop_ids: {dash.callback_context.triggered_prop_ids}"})

    if dash.callback_context.triggered_id == 'button-project-create':
        clip_duration = 3 if embedding_model == 'birdnet' else None  # Clip duration in seconds
        project_value = project_name
        gen_clips(project_name, audio_path, clip_duration)
        gen_embeddings(project_name, embedding_model)

    options = scan_projects()
    if project_value:
        brand = f"YAPAT | {project_value}"
    return options, project_value, brand

# layout = html.Div([
#     html.Label('Select project'),
#     dcc.Dropdown(
#         id='project-list-dropdown',
#         options=[{'label': 'True', 'value': 'True'}, {'label': 'False', 'value': 'False'}],
#         # value=debug_mode
#     ),
#
#     html.H1('Environment Variable Control Panel'),
#
#     html.Label('Debug Mode'),
#     dcc.Dropdown(
#         id='debug-dropdown',
#         options=[{'label': 'True', 'value': 'True'}, {'label': 'False', 'value': 'False'}],
#         # value=debug_mode
#     ),
#
#     html.Label('Database URL'),
#     dcc.Input(
#         id='database-url-input',
#         type='text',
#         # value=database_url,
#         style={'width': '100%'}
#     ),
#
#     html.Label('API Key'),
#     dcc.Input(
#         id='api-key-input',
#         type='text',
#         # value=api_key,
#         style={'width': '100%'}
#     ),
#
#     html.Button('Save Changes', id='save-button', n_clicks=0),
#
#     html.Div(id='save-status')
# ])


#
# #
# #
# # @app.callback(
# #     Output('save-status', 'children'),
# #     [Input('save-button', 'n_clicks')],
# #     [State('debug-dropdown', 'value'),
# #      State('database-url-input', 'value'),
# #      State('api-key-input', 'value')]
# # )
# # def update_env_file(n_clicks, debug_value, db_url_value, api_key_value):
# #     if n_clicks > 0:
# #         # Update the .env file
# #         set_key('.env', 'DEBUG', debug_value)
# #         set_key('.env', 'DATABASE_URL', db_url_value)
# #         set_key('.env', 'API_KEY', api_key_value)
# #
# #         return 'Environment variables updated!'
# #     return ''
# #
