import logging
import os

import dash_bootstrap_components as dbc
from dash import Dash
from flask import Flask

from src.extensions import db, login_manager, make_celery

logger = logging.getLogger(__name__)


def create_server():
    # Create the Flask app
    server = Flask('yapat')

    # Configure Flask server
    server.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY'),
        SQLALCHEMY_DATABASE_URI='sqlite:///main.db',
        SQLALCHEMY_BINDS={
            'user_db': 'sqlite:///user_management.db',
            'pipeline_db': 'sqlite:///pipeline_data.db'
        },
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        # Celery Configuration
        CELERY_BROKER_URL='redis://localhost:6379/0',
        CELERY_RESULT_BACKEND='redis://localhost:6379/0'
    )

    # Initialize extensions with the app
    db.init_app(server)
    login_manager.init_app(server)
    login_manager.login_view = 'login'

    with server.app_context():
        # Create all database tables
        db.create_all(bind_key=['user_db', 'pipeline_db'])  # Create tables

    # # Initialize Celery with the Flask server
    # celery = make_celery(server)

    return server


server = create_server()