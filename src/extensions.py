from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Initialize extensions without binding to the app yet
db = SQLAlchemy()
login_manager = LoginManager()
