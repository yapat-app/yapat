import logging

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State, Dash
from dash.exceptions import PreventUpdate
from flask_login import login_user
from sqlalchemy.exc import SQLAlchemyError

from components import navbar, footer
from components.login import login_location
from schema_model import User
from src import login_manager, sqlalchemy_db, server
from utils.db_operations import update_db_methods
from utils.settings import APP_HOST, APP_PORT, APP_DEBUG, DEV_TOOLS_PROPS_CHECK

logger = logging.getLogger(__name__)


def create_app(name='yapat', server=server, title='YAPAT | Yet Another PAM Annotation Tool'):
    """Create the Dash app and link it to Flask."""
    app = Dash(
        name=name,
        server=server,
        use_pages=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
        title=title
    )
    app.layout = serve_layout
    return app


@login_manager.user_loader
def load_user(user_id):
    """Load the user by their ID."""
    return sqlalchemy_db.session.execute(sqlalchemy_db.select(User).where(User.id == int(user_id))).scalar_one_or_none()


def serve_layout():
    """Define the layout of the application."""
    return html.Div(
        [
            login_location,
            navbar.navbar,
            dcc.Store(
                id='project-content',
                data={'project_name': '', 'current_sample': ''},
                storage_type='session'
            ),
            dbc.Container(dash.page_container, class_name='my-2'),
            footer.layout
        ]
    )


@callback(
    Output('login-feedback', 'children'),
    Output('url-login', 'pathname'),
    Input('login-button', 'n_clicks'),
    State('login-username', 'value'),
    State('login-password', 'value'),
    State('_pages_location', 'pathname'),
    prevent_initial_call=True
)
def login_button_click(n_clicks, username, password, pathname):
    """Handle login button click."""
    if n_clicks > 0:
        user = User.query.filter_by(username=username).first()
        from werkzeug.security import check_password_hash
        if user and check_password_hash(user.password, password):
            login_user(user)
            return 'Login Successful', '/'
        return 'Incorrect username or password', pathname
    raise PreventUpdate


@callback(
    Output('register-feedback', 'children'),
    Input('register-button', 'n_clicks'),
    State('register-username', 'value'),
    State('register-password', 'value')
)
def register_user(n_clicks, username, password):
    """Handle user registration."""
    if n_clicks > 0:
        if not username or not password:
            return "Username and password cannot be empty."

        # Check if the user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "This username is already taken. Please choose a different one."

        # Hash the password for security
        from werkzeug.security import generate_password_hash
        hashed_password = generate_password_hash(password, method='pbkdf2')

        # Create a new user and add to the database
        new_user = User(username=username, password=hashed_password)
        sqlalchemy_db.session.add(new_user)
        sqlalchemy_db.session.commit()

        return html.A("User registered successfully. Login here", href='/login')
    return ""


# Prevent app from running on Dask workers by using a main check
if __name__ == "__main__":
    # Only initialize the Dash app if this script is the entry point
    app = create_app()

    # Any additional initialization (such as database operations) should be kept in the main block
    with server.app_context():
        sqlalchemy_db.create_all(bind_key=['user_db', 'pipeline_db'])  # Create tables
        try:
            add_methods = update_db_methods()
            sqlalchemy_db.session.add_all(add_methods)
            sqlalchemy_db.session.commit()
        except SQLAlchemyError as e:
            sqlalchemy_db.session.rollback()
            logger.exception(e)

    # Run the Dash app
    app.run_server(
        host=APP_HOST,
        port=APP_PORT,
        debug=APP_DEBUG,
        dev_tools_props_check=DEV_TOOLS_PROPS_CHECK
    )


