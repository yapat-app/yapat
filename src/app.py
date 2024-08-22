# from flask_login import LoginManager
import os

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy

from components import navbar, footer
from utils.auth import User
from utils.settings import APP_HOST, APP_PORT, APP_DEBUG, DEV_TOOLS_PROPS_CHECK

# Initialize Flask app
server = Flask(__name__)
server.config.update(SECRET_KEY=os.getenv('SECRET_KEY'))
server.config.update(SQLALCHEMY_DATABASE_URI='sqlite:///users.db')
server.config.update(SQLALCHEMY_TRACK_MODIFICATIONS=False)

# Initialize Flask extensions
db = SQLAlchemy(server)
login_manager = LoginManager(server)
login_manager.login_view = '/login'


# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)


# Create database tables
with server.app_context():
    db.create_all()


# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Flask routes for registration and login
@server.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('New user created.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


# login_manager.init_app(server)

@server.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user, remember=True)
            return redirect(url_for('protected'))
        else:
            flash('Login unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')


@server.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


app = dash.Dash(
    __name__,
    server=server,
    use_pages=True,  # turn on Dash pages
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ],
    suppress_callback_exceptions=True,
    title='YAPAT | yet another PAM annotation tool'
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/dash/protected':
        return protected_layout()
    return login_redirect_layout()


def login_redirect_layout():
    return dcc.Link('Go to Login', href='/login')


@server.route('/protected')
@login_required
def protected():
    return redirect('/dash/protected')


def protected_layout():
    '''Define the layout of the application'''
    return html.Div(
        [
            # login_location,
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
            footer.footer
        ]
    )


# app.layout = serve_layout  # set the layout to the serve_layout function
# server = app.server  # the server is needed to deploy the application

if __name__ == "__main__":
    app.run_server(
        host=APP_HOST,
        port=APP_PORT,
        debug=APP_DEBUG,
        dev_tools_props_check=DEV_TOOLS_PROPS_CHECK
    )
