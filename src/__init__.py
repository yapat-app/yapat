import logging
import os

from flask import Flask

from extensions import db, login_manager

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

    return server


server = create_server()
