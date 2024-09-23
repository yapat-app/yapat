import os

from dash import Input, Output, State, callback, callback_context

from schema_model import ClusteringMethod, EmbeddingMethod, DimReductionMethod, Dataset
from src import sqlalchemy_db

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


@callback(
    Output("modal-pipeline", "is_open"),
    Input("new-pipeline", "n_clicks"),
    Input("cancel-new-pipeline", "n_clicks"),
    Input("create-pipeline", "n_clicks"),
    State("modal-pipeline", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(btn_new, btn_cancel, btn_creat, is_open):
    triggered_id = callback_context.triggered_id
    if triggered_id in ["new-pipeline", "cancel-new-pipeline", "create-pipeline"]:
        is_open = not is_open
    if triggered_id == "create-pipeline":
        pass
    return is_open


@callback(
    Output("new-pipeline-summary", "children"),
    Input("methods-embedding", "value"),
    Input("methods-clustering", "value"),
    Input("methods-dimred-viz", "value")
)
def display_pipeline_summary(m_e, m_c, m_dv):
    m_e = m_e if type(m_e) == list else [m_e]
    m_c = m_c if type(m_c) == list else [m_c]
    m_dv = m_dv if type(m_dv) == list else [m_dv]
    n_pipelines = len(m_e) * len(m_c) * len(m_dv)
    msg = f"{n_pipelines} pipelines will be computed"
    if n_pipelines == 1: msg = msg.replace("pipelines", "pipeline")
    return msg
