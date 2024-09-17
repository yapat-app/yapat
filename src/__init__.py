import logging
import os

import dash_bootstrap_components as dbc
from dash import Dash
from flask import Flask

from src.extensions import db, login_manager
from src.tasks import make_celery

logger = logging.getLogger(__name__)


def create_app():
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

    # Initialize Celery with the Flask app
    celery = make_celery(server)

    # Create Dash app and link it to Flask
    app = Dash(
        name='yapat',
        server=server,
        use_pages=True,  # Enable Dash pages
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
        title='YAPAT | Yet Another PAM Annotation Tool'
    )

    with server.app_context():
        # Create all database tables
        db.create_all(bind_key=['user_db', 'pipeline_db'])  # Create tables

    return app, celery
