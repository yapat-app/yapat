import logging
import os

from flask import Flask

from src.extensions import sqlalchemy_db, login_manager

logger = logging.getLogger(__name__)


def create_server():
    # Create the Flask app
    _server = Flask('yapat')

    # Configure Flask server
    _server.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY'),
        SQLALCHEMY_DATABASE_URI='sqlite:///main.db',
        SQLALCHEMY_BINDS={
            'user_db': 'sqlite:///user_management.db',
            'pipeline_db': 'sqlite:///pipeline_data.db'
        },
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )

    # Initialize extensions with the app
    sqlalchemy_db.init_app(_server)
    login_manager.init_app(_server)
    login_manager.login_view = 'login'

    return _server


server = create_server()
