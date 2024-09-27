import logging

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table

from .callbacks import list_existing_methods, fetch_embedding_and_clustering_methods
from .. import login_required_layout

logger = logging.getLogger(__name__)

dash.register_page(
    __name__,
    path='/explore',
    redirect_from=['/visualise', '/visualize', '/viz'],
    title='Explore | YAPAT'
)


# layout = html.Header([
#     html.H1('Explore'),
#     dbc.Breadcrumb(
#         items=[
#             {"label": "Embed", "href": "/docs", "external_link": True},
#             {
#                 "label": "Cluster",
#                 "href": "/docs/components",
#                 "external_link": True,
#             },
#             {"label": "Visualize", "active": True},
#         ],
#     )
# ])




@login_required_layout
def layout():
    columns = [
        {"name": "Metric", "id": "metric"},
        {"name": "BirdNET", "id": "birdnet"},
        {"name": "Acoustic Indices", "id": "acoustic_indices"},
        {"name": "VAE", "id": "vae"}
    ]

    # Define the data for each row
    data = [
        {"metric": "F1 Score (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Prediction Error (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "F1 Score (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Prediction Error (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Silhouette Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Davies Bouldin Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0}
    ]

    layout = dbc.Container([
        html.Div([
            # Header
            dbc.Row([
                html.H2("Visualization Pipelines"),
                html.P(["Mix and match methods for data representation/embedding, clustering, and "
                        "dimensionality reduction."
                        "Create new pipeline to calculate and Load Pipeline to evaluate and visualise."]),
            ]),

            # Main
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        # html.H5('Build pipeline'),
                        dbc.ButtonGroup([
                            #dbc.Button("Quickstart", id="quickstart-pipeline", n_clicks=0),
                            dbc.Button("New Pipeline", id="new-pipeline", n_clicks=0),
                            dbc.Button("Load Pipeline", id="load-pipeline", n_clicks=0)
                        ], vertical=True),
                    ], class_name='my-4')
                ], width=2),
                # Right Sidebar - Evaluation and Visualization
                dbc.Col([
                    dbc.Row([
                        html.H5("Evaluation Metrics"),
                        dash_table.DataTable(
                            id='evaluation-table',
                            columns=columns,
                            data=data,
                            merge_duplicate_headers=True,  # This will merge headers like 'birdnet' in one line
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'center'}
                        ),
                    ], class_name='my-4'),

                    dbc.Row([
                        # Visualization Section
                        html.H5("Visualization"),
                        dbc.Tabs([
                            dbc.Tab(label='Cluster Time Histogram', tab_id='time-histogram'),
                            dbc.Tab(label='Cluster Time Grid', tab_id='time-grid'),
                            dbc.Tab(label='State Space Visualisation', tab_id='state-space'),
                            dbc.Tab(label='Time Series Visualisation', tab_id='state-space'),
                            dbc.Tab(label='Temporal Rose Plot', tab_id='state-space'),
                            dbc.Tab(label='Explained Variance Plot', tab_id='state-space'),

                        ], id='visualization-tabs', active_tab='time-histogram'),

                        # Placeholder for visualizations
                        html.Div(id='visualization-content', className='my-4')
                    ])
                ], width=9),
            ]),

            # Modals
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("New pipeline")),

                    dbc.ModalBody(children=[
                        html.P("Representation space"),
                        dcc.Dropdown(
                            options=[{'label': method, 'value': method} for method in
                                     list_existing_methods('embeddings')],
                            value='birdnet', multi=True,
                            placeholder="Select embedding", id="methods-embedding",
                        ),
                        # html.P("Pre-clustering dimensionality reduction"),
                        # dcc.Dropdown(
                        #     options=[{'label': method, 'value': method} for method in
                        #              list_existing_methods('dimensionality_reduction')] + [
                        #                 {'label': 'None', 'value': 'None'}],
                        #     value='None', multi=True,
                        #     placeholder="Select pre-clustering reduction", id='new-preclust-reduct'
                        # ),
                        html.P("Clustering"),
                        dcc.Dropdown(
                            options=[{'label': method, 'value': method} for method in
                                     list_existing_methods('clustering')],
                            value='kmeans', multi=True,
                            placeholder="Select clustering method", id='methods-clustering'
                        ),
                        html.P("Dimensionality reduction (for vizualization)"),
                        dcc.Dropdown(
                            options=[{'label': method, 'value': method} for method in
                                     list_existing_methods('dimensionality_reduction')],
                            value='umap', multi=True,
                            placeholder="Select post-clustering reduction", id='methods-dimred-viz'
                        )
                    ], id="new-pipeline-modal-body"),

                    dbc.ModalFooter([
                        html.P(id='new-pipeline-summary'),
                        dbc.ButtonGroup([
                            dbc.Button("Cancel", id="cancel-new-pipeline", n_clicks=0, color="secondary"),
                            dbc.Button("Create", id="create-pipeline", n_clicks=0, color="primary"),
                        ])

                    ]),
                ],
                id="modal-pipeline",
                centered=True,
                is_open=False,
            ),
        ])
    ])
    return layout
