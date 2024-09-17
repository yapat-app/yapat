import os

from schema_model import ClusteringMethod, EmbeddingMethod, DimReductionMethod, Dataset
from src import db

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
        # existing_methods = db.session.execute(db.select(PipelineTable.method_name)).fetchall()
        # existing_methods = set([i[0] for i in existing_methods])
        method_names = set(method_names) - set(existing_methods)
        Table = pipeline_steps[package_name]
        add_methods += [Table(method_name=method_name) for method_name in method_names]

    return add_methods


def list_existing_methods(package_name):
    Table = pipeline_steps[package_name]
    existing_methods = db.session.execute(db.select(Table.method_name)).fetchall()
    existing_methods = [i[0] for i in existing_methods]
    return existing_methods


def list_existing_datasets():
    existing_datasets = db.session.execute(db.select(Dataset.dataset_name)).fetchall()
    existing_datasets = [i[0] for i in existing_datasets]
    return existing_datasets
