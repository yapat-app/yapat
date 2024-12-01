import os
import json

import dash
from dash import Input, Output, State, callback, callback_context, dcc, html

from src.schema_model import ClusteringMethod, ClusteringResult, EmbeddingMethod, DimReductionMethod, Dataset, EmbeddingResult
from src import sqlalchemy_db
from src.embeddings import get_embedding_model
from src.clustering import get_clustering_model
from src.dimensionality_reduction import get_dr_model
from src.evaluations.embedding_evaluation import EmbeddingsEvaluation
from src.evaluations.clustering_evaluation import ClusteringEvaluation
from src.visualizations.cluster_temporal_histogram import ClusterTemporalHist
from src.visualizations.cluster_time_grid import ClusterTimeGrid
from src.visualizations.state_space_visualization import StateSpaceVis
from src.visualizations.time_series_plot import TimeSeries
from src.visualizations.rose_plot import RosePlot


pipeline_steps = {
    'embeddings': EmbeddingMethod,
    'clustering': ClusteringMethod,
    'dimensionality_reduction': DimReductionMethod
}




def update_db_methods():
    add_methods = []
    del_methods = []

    for package_name in pipeline_steps.keys():
        method_names = [file[:-3] for file in os.listdir(package_name) if
                        file.endswith('.py') and file != '__init__.py']
        existing_methods = list_existing_methods(package_name)
        method_names = set(method_names) - set(existing_methods)
        Table = pipeline_steps[package_name]
        add_methods += [Table(method_name=method_name) for method_name in method_names]

    return add_methods


def list_existing_methods(package_name):
    Table = pipeline_steps[package_name]
    existing_methods = sqlalchemy_db.session.execute(sqlalchemy_db.select(Table.method_name)).fetchall()
    existing_methods = [i[0] for i in existing_methods]
    return existing_methods


def list_existing_datasets():
    existing_datasets = sqlalchemy_db.session.execute(sqlalchemy_db.select(Dataset.dataset_name)).fetchall()
    existing_datasets = [i[0] for i in existing_datasets]
    return existing_datasets

def extract_evaluation_results():
    data = [
        {"metric": "F1 Score (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Accuracy (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "F1 Score (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Accuracy (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Explained Variance", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Entropy", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Silhouette Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Davies Bouldin Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0}
    ]

    selected_dataset = sqlalchemy_db.session.query(Dataset).filter_by(is_selected=True).first()
    dataset_id = selected_dataset.id
    embedding_results = sqlalchemy_db.session.query(EmbeddingResult).filter_by(dataset_id=dataset_id, task='completed').all()
    for result in embedding_results:
        method = sqlalchemy_db.session.query(EmbeddingMethod).filter_by(id=result.embedding_id).first()
        if result.evaluation_results and result.evaluation_results != "{}":
            evaluation_data = json.loads(result.evaluation_results)
        else:
            evaluation_data = {}
        if method.method_name in ["birdnet", "acoustic_indices", "vae"]:
            prefix = method.method_name
            data[0][prefix] = evaluation_data.get("f1_score_time", "N/A")
            data[1][prefix] = evaluation_data.get("accuracy_time", "N/A")
            data[2][prefix] = evaluation_data.get("f1_score_location", "N/A")
            data[3][prefix] = evaluation_data.get("accuracy_location", "N/A")
            data[4][prefix] = evaluation_data.get("Explained Variance", "N/A")
            data[5][prefix] = evaluation_data.get("Entropy", "N/A")
            data[6][prefix] = evaluation_data.get("Silhouette Index", "N/A")
            data[7][prefix] = evaluation_data.get("Davies Bouldin Index", "N/A")


    for result in embedding_results:
        method = sqlalchemy_db.session.query(EmbeddingMethod).filter_by(id=result.embedding_id).first()
        full_sil = []
        full_db = []
        completed_clustering_results = sqlalchemy_db.session.query(ClusteringResult).filter_by(
            task='completed', embedding_id=result.id).all()
        for clustering_result in completed_clustering_results:
            clustering_method = sqlalchemy_db.session.query(ClusteringMethod).filter_by(method_id=clustering_result.method_id).first()
            if clustering_result.evaluation_results and clustering_result.evaluation_results != "{}":
                evaluation_data = json.loads(clustering_result.evaluation_results)
            else:
                evaluation_data = {}
            full_sil.append(f"{clustering_method.method_name}:{evaluation_data.get('Silhouette Score', 'N/A')}")
            full_db.append(f"{clustering_method.method_name}:{evaluation_data.get('Davies Bouldin Score', 'N/A')}")
        data[6][method.method_name] = ", ".join(full_sil)
        data[7][method.method_name] = ", ".join(full_db)

    return data

# Callback to switch content visibility based on the selected navigation link
@callback(
    [Output('pipeline-content', 'style'),
     Output('visualisation-content', 'style'),
     Output('evaluation-content', 'style'),
     Output('evaluation-table', 'data')],
    [Input('pipeline-link', 'n_clicks'),
     Input('visualisation-link', 'n_clicks'),
     Input('evaluation-link', 'n_clicks')]
)
def switch_tab_content(pipeline_click, visualisation_click, evaluation_click):
    # Default: Hide all sections
    pipeline_style = {'display': 'none'}
    visualisation_style = {'display': 'none'}
    evaluation_style = {'display': 'none'}
    evaluation_result = dash.no_update

    # Determine which button was clicked
    ctx = callback_context

    if not ctx.triggered:
        return pipeline_style, visualisation_style, evaluation_style, evaluation_result  # No tab clicked yet

    # Get the id of the button that triggered the callback
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'pipeline-link':
        pipeline_style = {'display': 'block'}  # Show pipeline content

    elif button_id == 'visualisation-link':
        visualisation_style = {'display': 'block'}  # Show visualisation content

    elif button_id == 'evaluation-link':
        evaluation_style = {'display': 'block'}
        evaluation_result = extract_evaluation_results()# Show evaluation content

    return pipeline_style, visualisation_style, evaluation_style, evaluation_result



@callback(
    Output("status-pipeline", "children"),
    Input("create-pipeline", "n_clicks"),
    State("pipeline-methods-embedding", "value"),
    State("pipeline-methods-clustering", "value"),
    State("pipeline-methods-dimred-viz", "value"),
    prevent_initial_call=True
)

def run_pipeline(n_clicks, m_e, m_c, m_dv):
    if n_clicks is None:
        return ""
    if m_e:
        embedding_method = m_e[0] if isinstance(m_e, list) else m_e
        embedding_instance = get_embedding_model(embedding_method)
        embedding_instance.process()
        evaluation_instance = EmbeddingsEvaluation(embedding_method, None)
        evaluation_instance.evaluate()
        if m_c:
            clustering_method = m_c[0] if isinstance(m_c, list) else m_c
            clustering_instance = get_clustering_model(clustering_method)
            clustering_instance.fit(embedding_method)
            evaluation_instance = ClusteringEvaluation(embedding_method, clustering_method)
            evaluation_instance.evaluate()
        if m_dv:
            dim_reduction_method = m_dv[0] if isinstance(m_dv, list) else m_dv
            dim_reduction_instance = get_dr_model(dim_reduction_method)
            dim_reduction_instance.fit_transform(embedding_method)
        return "Pipeline calculations done. Please proceed to Visualise or Evaluate."


# To generate and store figures when a particular pipeline loaded
@callback(
    Output('loaded-figures', 'data'),
    Output('status-pipeline', 'children', allow_duplicate=True),
    Input('load-pipeline', 'n_clicks'),
    State("pipeline-methods-embedding", "value"),
    State("pipeline-methods-clustering", "value"),
    State("pipeline-methods-dimred-viz", "value"),
    prevent_initial_call=True
)
def load_figures(n_clicks, m_e, m_c, m_dv):
    if n_clicks > 0:
        embedding_method = m_e[0] if isinstance(m_e, list) else m_e
        clustering_method = m_c[0] if isinstance(m_c, list) else m_c
        dim_reduction_method = m_dv[0] if isinstance(m_dv, list) else m_dv

        figures = {
            "cluster-time-histogram": ClusterTemporalHist(embedding_method, clustering_method,
                                                          dim_reduction_method).plot(),
            "cluster-time-grid": ClusterTimeGrid(embedding_method, clustering_method, dim_reduction_method).plot(),
            "time-series": TimeSeries(embedding_method, clustering_method, dim_reduction_method).plot(),
            "temp-rose-plot": RosePlot(embedding_method, clustering_method, dim_reduction_method).plot(),
            "cluster-state-space": StateSpaceVis(embedding_method, clustering_method, dim_reduction_method).plot()
        }
        return figures, "Figures Fetched. Please view them in the respective Tabs"
    return {}, "No figures are loaded."


# To toggle figures based on selected tab
@callback(
    Output('figure-display', 'children'),
    Output('status-vis', 'children', allow_duplicate=True),
    [Input('visualisation-tabs', 'active_tab')],
    [State('loaded-figures', 'data')],
    prevent_initial_call=True
)
def update_visualization_content(active_tab, figures_data):
    # If no figures have been loaded, prompt the user
    if not figures_data:
        return "", "Please load a pipeline to view visualizations."
    figure_data = figures_data.get(active_tab)
    if figure_data:
        figure_component = dcc.Graph(figure=figure_data)
    else:
        figure_component = html.Div("Figure not found for this tab.")

    return figure_component, ""




# @callback(
#     Output("new-pipeline-summary", "children"),
#     Input("methods-embedding", "value"),
#     Input("methods-clustering", "value"),
#     Input("methods-dimred-viz", "value")
# )
# def display_pipeline_summary(m_e, m_c, m_dv):
#     m_e = m_e if type(m_e) == list else [m_e]
#     m_c = m_c if type(m_c) == list else [m_c]
#     m_dv = m_dv if type(m_dv) == list else [m_dv]
#     n_pipelines = len(m_e) * len(m_c) * len(m_dv)
#     msg = f"{n_pipelines} pipelines will be computed"
#     if n_pipelines == 1: msg = msg.replace("pipelines", "pipeline")
#     return msg








