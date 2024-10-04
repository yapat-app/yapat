import logging
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

from pages.explore.callbacks import list_existing_methods
from .. import login_required_layout

logger = logging.getLogger(__name__)

dash.register_page(
    __name__,
    path='/explore',
    redirect_from=['/visualise', '/visualize', '/viz'],
    title='Explore | YAPAT'
)

@login_required_layout
def layout():
    sidebar = dbc.Col(
        dbc.Nav(
            [
                dbc.NavLink("Pipeline", id="pipeline-link", n_clicks=0,
                            className="custom-nav-link"),
                dbc.NavLink("Visualisation", id="visualisation-link", n_clicks=0,
                            className="custom-nav-link"),
                dbc.NavLink("Evaluation", id="evaluation-link", n_clicks=0,
                            className="custom-nav-link")
            ],
            vertical=True,
            pills=True
        ),
        width=2,
    )
    # sidebar = dbc.Col(
    #     dbc.Nav(
    #         [
    #             dbc.NavLink("Pipeline", href="/explore#pipeline", id="pipeline-link", active="exact",
    #                          className="custom-nav-link"),
    #             dbc.NavLink("Visualisation", href="/explore#visualisation", id="visualisation-link", active="exact",
    #                         className="custom-nav-link"),
    #             dbc.NavLink("Evaluation", href="/explore#evaluation", id="evaluation-link", active="exact",
    #                         className="custom-nav-link")
    #         ],
    #         vertical=True,
    #         pills=True
    #     ),
    #     width=2,
    # )

    # Pipeline Content
    pipeline_content = html.Div([
        html.H5("Pipeline Options"),

        # Dropdown for embedding methods
        html.P("Representation space"),
        dcc.Dropdown(
            options=[{'label': method, 'value': method} for method in list_existing_methods('embeddings')],
            value='birdnet',  # Default value
            multi=False,
            placeholder="Select embedding",
            id="pipeline-methods-embedding"
        ),

        # Dropdown for clustering methods
        html.P("Clustering"),
        dcc.Dropdown(
            options=[{'label': method, 'value': method} for method in list_existing_methods('clustering')],
            value='kmeans',  # Default value
            multi=False,
            placeholder="Select clustering method",
            id="pipeline-methods-clustering"
        ),

        # Dropdown for dimensionality reduction methods
        html.P("Dimensionality reduction (for visualization)"),
        dcc.Dropdown(
            options=[{'label': method, 'value': method} for method in list_existing_methods('dimensionality_reduction')],
            value='umap_reducer',  # Default value
            multi=False,
            placeholder="Select post-clustering reduction",
            id="pipeline-methods-dimred-viz"
        ),

        # Buttons for actions
        dbc.Button("Create Pipeline (Compute)", id="create-pipeline", n_clicks=0, className="my-2"),
        dbc.Button("Load Pipeline (Visualize)", id="load-pipeline", n_clicks=0, className="my-2"),
        html.Div(id='status-pipeline', style={"marginTop": "20px"}),
    ],
    style={'display': 'none'},
    id= "pipeline-content")

    # Visualization Content
    visualization_content = html.Div(
        children=[
            dbc.Tabs(
                id='visualisation-tabs',
                active_tab='cluster-time-histogram',
                children=[
                    dbc.Tab(label='Cluster Time Histogram', tab_id='cluster-time-histogram'),
                    dbc.Tab(label='Cluster Time Grid', tab_id='cluster-time-grid'),
                    dbc.Tab(label='State Space Visualisation', tab_id='cluster-state-space'),
                    dbc.Tab(label='Time Series Visualisation', tab_id='time-series'),
                    dbc.Tab(label='Temporal Rose Plot', tab_id='temp-rose-plot'),
                    dbc.Tab(label='Explained Variance Plot', tab_id='explained-variance'),
                ]
            ),
            html.Div(id='figure-display'),
            html.Div(id='status-vis', style={"marginTop": "20px"}),
        ],
        style={'display': 'none'},
        id='visualisation-content'
    )

    # Evaluation Content
    evaluation_content = html.Div(
        children=[
            html.H5("Evaluation Metrics"),
            dash_table.DataTable(
                id="evaluation-table",
                columns=[
                    {"name": "Metric", "id": "metric"},
                    {"name": "BirdNET", "id": "birdnet"},
                    {"name": "Acoustic Indices", "id": "acoustic_indices"},
                    {"name": "VAE", "id": "vae"}
                ],
                data=[
                    {"metric": "F1 Score (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
                    {"metric": "Accuracy (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
                    {"metric": "F1 Score (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
                    {"metric": "Accuracy (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
                    {"metric": "Explained Variance", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
                    {"metric": "Entropy", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
                    {"metric": "Silhouette Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
                    {"metric": "Davies Bouldin Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0}
                ],
                style_table={'overflowX': 'auto'},
            )
        ],
        style={'display': 'none'},
        id ="evaluation-content"
    )

    # Content area to display based on sidebar selection
    content = dbc.Col(
        [
            pipeline_content,
            visualization_content,
            evaluation_content
        ],
        width=10
    )

    # Main layout with sidebar and content
    return dbc.Container([
        dbc.Row([sidebar, content]),
        html.Div(id='loaded-figures', style={'display': 'none'})
    ])

