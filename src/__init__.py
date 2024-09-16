import os

from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()


def create_server():
    # Create the Flask app
    server = Flask('yapat')

    # Configuration
    server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///main.db'
    server.config['SQLALCHEMY_BINDS'] = {
        'user_db': 'sqlite:///user_management.db',  # Database for user access
        'pipeline_db': 'sqlite:///pipeline_data.db'  # Database for ML pipeline
    }
    server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    server.config.update(SECRET_KEY=os.getenv('SECRET_KEY'))

    # Initialize extensions with the app
    db.init_app(server)
    login_manager.init_app(server)
    login_manager.login_view = 'login'

    # Import models and routes within the function to avoid circular imports
    with server.app_context():
        # from src.schema import User  # Import models inside the function
        db.create_all(bind_key=['user_db', 'pipeline_db'])  # Create tables

    # # Register routes (blueprints) here
    # from your_app.routes import main_blueprint  # Adjust based on your routes
    # server.register_blueprint(main_blueprint)

    return server
