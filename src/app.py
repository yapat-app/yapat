# from flask_login import LoginManager
import logging

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
from flask_login import login_user
from sqlalchemy.exc import SQLAlchemyError

from components import navbar, footer
from components.login import login_location
from pages.explore.callbacks import update_db_methods
from schema import User
from src import login_manager, db, create_server
from utils.settings import APP_HOST, APP_PORT, APP_DEBUG, DEV_TOOLS_PROPS_CHECK

logger = logging.getLogger(__name__)

server = create_server()

with server.app_context():
    db.create_all(bind_key=['user_db', 'pipeline_db'])
    try:
        add_methods = update_db_methods()
        db.session.add_all(add_methods)
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.exception(e)


@login_manager.user_loader
def load_user(user_id):
    from schema import User  # Import your models
    return db.session.execute(db.select(User).where(User.id == int(user_id))).scalar_one_or_none()


app = dash.Dash(
    name="yapat",
    server=server,
    use_pages=True,  # turn on Dash pages
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ],
    suppress_callback_exceptions=True,
    title='YAPAT | yet another PAM annotation tool'
)


def serve_layout():
    """Define the layout of the application"""
    return html.Div(
        [
            login_location,
            navbar.navbar,
            dcc.Store(
                id='project-content',
                data={
                    'project_name': '',
                    'current_sample': '',
                },
                storage_type='session'
            ),
            dbc.Container(
                dash.page_container,
                class_name='my-2'
            ),
            footer.layout
        ]
    )


app.layout = serve_layout


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
    if n_clicks > 0:
        user = User.query.filter_by(username=username).first()
        from werkzeug.security import check_password_hash
        if user and check_password_hash(user.password, password):
            login_user(user)
            return 'Login Successful', '/'
        return 'Incorrect username or password', pathname
    raise PreventUpdate


@app.callback(
    Output('register-feedback', 'children'),
    Input('register-button', 'n_clicks'),
    State('register-username', 'value'),
    State('register-password', 'value')
)
def register_user(n_clicks, username, password):
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
        db.session.add(new_user)
        db.session.commit()

        return html.A("User registered successfully. Login here", href='/login')

    return ""


def main():
    """Entry point for the Dash app"""
    app.run_server(
        host=APP_HOST,
        port=APP_PORT,
        debug=APP_DEBUG,
        dev_tools_props_check=DEV_TOOLS_PROPS_CHECK
    )


if __name__ == "__main__":
    main()
