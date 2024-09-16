import logging

import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, callback

from .callbacks import list_existing_methods
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
    layout = dbc.Container([
        html.Div([
            # Header
            dbc.Row([
                html.H2("Visualization Pipelines"),
                html.P(["Mix and match methods for data representation/embedding, clustering, and "
                        "dimensionality reduction. Visualize the results with THIS and THIS plot types"]),
            ]),

            # Main
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        # html.H5('Build pipeline'),
                        dbc.ButtonGroup([
                            dbc.Button("Quickstart", id="quickstart-pipeline", n_clicks=0),
                            dbc.Button("Load", id="load-pipeline", n_clicks=0),
                            dbc.Button("New...", id="new-pipeline", n_clicks=0),
                        ], vertical=True),
                    ], class_name='my-4')
                ], width=1),
                dbc.Col([
                    dbc.Row([
                        # html.H5('Visualize results'),
                    ], class_name='my-4', id='pipeline-summary'),
                    dbc.Row([
                        # dbc.DropdownMenu()
                    ], class_name='my-4')
                ])
            ]),

            # Modals
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("New pipeline")),
                    dbc.ModalBody(children=[

                        "Choose a representation space",
                        dbc.Select(
                            options=[{'label': method, 'value': method} for method in
                                     list_existing_methods('embeddings')],
                            value='birdnet',
                            placeholder="Select embedding", class_name='my-1', id="methods-embedding"
                        ),

                        "Pre-clustering dimensionality reduction",
                        dbc.Select(
                            options=[{'label': method, 'value': method} for method in
                                     list_existing_methods('dimensionality_reduction')] + [
                                        {'label': 'None', 'value': None}],
                            value='None',
                            placeholder="Select pre-clustering reduction", class_name='my-1', id='new-preclust-reduct'
                        ),

                        "Clustering",
                        dbc.Select(
                            options=[{'label': method, 'value': method} for method in
                                     list_existing_methods('clustering')],
                            value='kmeans',
                            placeholder="Select clustering method", class_name='my-1', id='new-clustering'
                        ),

                        "Post-clustering dimensionality reduction",
                        dbc.Select(
                            options=[{'label': method, 'value': method} for method in
                                     list_existing_methods('dimensionality_reduction')],
                            value='umap',
                            placeholder="Select post-clustering reduction", class_name='my-1', id='new-postclust-reduct'
                        )
                    ], id="new-pipeline-modal-body"),
                    dbc.ModalFooter([
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


# def new_pipeline_modal_body():
#     children =
#     return children


@callback(
    Output("modal-pipeline", "is_open"),
    Input("new-pipeline", "n_clicks"),
    Input("cancel-new-pipeline", "n_clicks"),
    [State("modal-pipeline", "is_open")],
)
def toggle_modal(btn_new, btn_cancel, is_open):
    if btn_new or btn_cancel:
        is_open = not is_open
    return is_open

# @callback(
#     Output("summary-pipeline", "children"),
#     "methods-embedding", "new-preclust-reduct", "new-clustering", "new-postclust-reduct"
# )
