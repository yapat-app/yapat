from datetime import datetime

from flask_login import UserMixin
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship, backref

from src.extensions import db


# Define User model
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    __bind_key__ = 'user_db'
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    password = Column(String(150), nullable=False)


# 1. Datasets Table
class Dataset(db.Model):
    __tablename__ = 'datasets'
    __bind_key__ = "pipeline_db"

    dataset_id = Column(Integer, primary_key=True)
    dataset_name = Column(String(255), unique=True, nullable=False)
    path_audio = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# 2. Embedding Methods Table
class EmbeddingMethod(db.Model):
    __tablename__ = 'embeddings'
    __bind_key__ = "pipeline_db"

    method_id = Column(Integer, primary_key=True)
    method_name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# 3. Embeddings Table
class EmbeddingResult(db.Model):
    __tablename__ = 'embedding_results'
    __bind_key__ = "pipeline_db"

    embedding_id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.dataset_id'), nullable=False)
    method_id = Column(Integer, ForeignKey('embeddings.method_id'),
                       nullable=False)
    embedding_file_path = Column(String(255),
                                 nullable=False)  # Store path to the embedding file
    hyperparameters = Column(JSON, nullable=True)  # Hyperparameters stored as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship('Dataset', backref=backref('embedding_results', lazy=True))
    method = relationship('EmbeddingMethod', backref=backref('embedding_results', lazy=True))


# 4. Clustering Methods Table
class ClusteringMethod(db.Model):
    __tablename__ = 'clustering'
    __bind_key__ = "pipeline_db"

    method_id = Column(Integer, primary_key=True)
    method_name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# 5. Clustering Results Table
class ClusteringResult(db.Model):
    __tablename__ = 'clustering_results'
    __bind_key__ = "pipeline_db"

    result_id = Column(Integer, primary_key=True)
    embedding_id = Column(Integer, ForeignKey('embedding_results.embedding_id'),
                          nullable=False)
    method_id = Column(Integer, ForeignKey('clustering.method_id'),
                       nullable=False)
    cluster_file_path = Column(String(255),
                               nullable=False)  # Store path to the clustering result file
    hyperparameters = Column(JSON, nullable=True)  # Clustering hyperparameters stored as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    embedding = relationship('EmbeddingResult', backref=backref('clustering_results', lazy=True))
    method = relationship('ClusteringMethod', backref=backref('clustering_results', lazy=True))


# 6. Dimensionality Reduction Methods Table
class DimReductionMethod(db.Model):
    __tablename__ = 'dimensionality_reduction'
    __bind_key__ = "pipeline_db"

    method_id = Column(Integer, primary_key=True)
    method_name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# 7. Dimensionality Reduction Results Table
class DimReductionResult(db.Model):
    __tablename__ = 'dim_reduction_results'
    __bind_key__ = "pipeline_db"

    result_id = Column(Integer, primary_key=True)
    clustering_result_id = Column(Integer,
                                  ForeignKey('clustering_results.result_id'), nullable=False)
    method_id = Column(Integer, ForeignKey('dimensionality_reduction.method_id'),
                       nullable=False)
    reduction_file_path = Column(String(255),
                                 nullable=False)  # Path to the dimensionality reduction result file
    hyperparameters = Column(JSON,
                             nullable=True)  # Dimensionality reduction hyperparameters stored as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    clustering_result = relationship('ClusteringResult',
                                     backref=backref('dim_reduction_results', lazy=True))
    method = relationship('DimReductionMethod',
                          backref=backref('dim_reduction_results', lazy=True))

# TODO Add "task_state" field to all results tables