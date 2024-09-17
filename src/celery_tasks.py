import logging
from importlib import import_module

from sqlalchemy.exc import SQLAlchemyError

# from app import server
from extensions import make_celery
from schema_model import Dataset, EmbeddingMethod
from src import db, server

logger = logging.getLogger(__name__)

celery = make_celery(server)


@celery.task
def compute_embeddings(dataset_id, embedding_id):
    path_audio = db.session.execute(db.select(Dataset).filter_by(id=dataset_id)).scalar_one_or_none()
    embedding = db.session.execute(db.select(EmbeddingMethod).filter_by(id=embedding_id))
    embedding = import_module(f"embeddings/{embedding}")
    return embedding.fit_transform(path_audio)


@celery.task
def register_dataset(dataset_name, path_audio):
    try:
        db.session.add(Dataset(dataset_name=dataset_name, path_audio=path_audio))
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.exception(e)
