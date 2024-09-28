import logging

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

from pages import login_required_layout
from pages.explore.callbacks import list_existing_methods

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
                    ], class_name='my-4'),
                    html.P(id="create-pipeline-msg"),
                ])
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
