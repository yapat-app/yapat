import dash

import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import html, dcc

dash.register_page(
    __name__,
    path='/annotate',
    title='Annotation | YAPAT'
)

col_annotation = html.Div(
    [
        # dcc.Interval(id='interval_1s', interval=1000),
        dbc.Row(
            [
                dbc.Col(
                    html.H6('Sampling strategy:'), width='auto', style={'padding-right': '1'}
                ),
                # TODO set default value to validate
                dbc.Col(
                    dcc.RadioItems(id='radio_sampling', options=['validate', 'refine', 'discover', 'random'],
                                   value='random', inline=True,
                                   labelStyle={'display': 'inline-block', 'margin-right': '10px'}),
                    style={'padding-left': '0'},
                    width='auto'
                )
            ]
        ),

        html.Hr(style={'padding-top': '0', 'padding-bottom': '0', }),

        dbc.Row(
            [
                dbc.Col(
                    html.H6('Filter by:'), width='auto', style={'padding-right': '1'}
                ),
                dbc.Col(
                    dcc.Checklist(id='checklist_class', options=['Amphibia', 'Aves', 'Insecta', 'Mammalia', 'Others'],
                                  inline=True,
                                  labelStyle={'display': 'inline-block', 'margin-right': '10px'})
                )
            ]
        ),

        html.Hr(),

        dbc.Row(
            [
                dbc.Col(
                    html.H6('Annotation: '),
                    width='auto',
                    style={'padding-right': '1'}),
                dbc.Col(
                    dcc.Input(id='input_search', type='text', placeholder='Search for species'),
                    width='auto',
                    style={'padding-left': '0'}
                )
            ],
            align='center'
        ),
        dbc.Row(
            dcc.Checklist(id='checklist_annotation', style={'maxHeight': 300, 'height': 300, 'overflow-y': 'auto'})),
        html.H6(id='text_model_suggestions'),

        html.Hr(),

        dbc.Row(
            [
                dbc.Col(
                    html.H6('Add new species:'),
                    width='auto',
                    style={'padding-right': '1'}
                ),
                dbc.Col(
                    dcc.Input(id='input_new_class', type='text', placeholder='Placeholder_File_Format'),
                    width='auto',
                    style={'padding-right': '0', 'padding-left': '0', }
                ),
                dbc.Col(
                    dbc.Button('ADD', id='button_add_class'),
                    width='auto',
                    style={'padding-left': '0'})
            ],
            align='center'
        ),

        dbc.Row(
            [
                html.Hr(),
                dbc.Col(
                    html.H6(id='text_species_to_process'),
                    width='auto',
                    style={'padding-right': '1'}
                ),
                dbc.Col(
                    dcc.Dropdown(id='dropdown_selected_species', multi=True),
                    width=12,
                    style={'padding-right': '0', 'padding-left': '0'}
                )
            ],
            align='center',
            id='row_species_to_process',
            style={'display': 'block'}
        ),

        html.Hr(),

        dbc.Row(
            [
                dbc.Col(dbc.Button('SUBMIT',
                                   id='button_submit', color='success', size='lg',
                                   style={'width': '80%', 'height': '150%'}
                                   ),
                        className='d-grid gap-2'),
                dbc.Col(dbc.Button('SKIP',
                                   id='button_skip', color='danger', size='lg',
                                   style={'width': '80%', 'height': '150%'}
                                   ),
                        className='d-grid gap-2')
            ]
        )
    ]
)

col_complete_sample = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.H6('Greyscale'), width='auto', style={'padding-right': '0'}),
                dbc.Col(daq.BooleanSwitch(id='switch_colormap', on=False, style={'padding-left': '0'}),
                        width='auto'),
            ],
            align='center',
            justify='center',
        ),
        dcc.Loading(id='loading_audioComplete',
                    children=
                    [
                        dbc.Row(
                            html.H6(id='text_complete_file_name', style={'display': 'block'}),
                            justify='center',
                        ),
                        dbc.Row(
                            dcc.Graph(id='graph_complete_spectrogram', config={'displaylogo': False}),
                            justify='center',
                        ),
                    ],
                    type='circle',
                    ),
        dbc.Row(
            html.Audio(id='audio_complete_file',
                       controls=True,
                       autoPlay=False,
                       style={'width': '70%'}),
            justify='center',
        ),
    ]
)

layout = dbc.Container([html.Div([
    # dbc.Row([
    #     dbc.Col(html.Div(col_timer.layout), width='2'),
    # ]),
    dbc.Row([
        dbc.Col(html.Div(col_complete_sample), width='8',
                style={'border-right': '1px solid #dddddd'}),
        dbc.Col(html.Div(col_annotation)),
        # , width=2, style={'border-right': '1px solid #dddddd'}),
        # dbc.Col(html.Div(col_filtered_sample_selection.layout), width=2),
        # dbc.Col(html.Div(col_filtered_sample_audio.layout), width=4),
    ])
], style={
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "overflow": "scroll",
    "height": 2000
})
], fluid=True)
