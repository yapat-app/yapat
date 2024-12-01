# main.py
import argparse
import os
import sys

import dash
# import dash_auth
import dash_bootstrap_components as dbc
from flask import Flask, session
from src.pages.annotate.callbacks import register_callbacks

sys.path.append(os.path.dirname(__file__))

# from src.callbacks import register_callbacks
# from src.layout import layout

# from flask_session import Session


# from mini_custom_styles import CONTENT_STYLE

# Create Flask server
server = Flask(__name__)
server.secret_key = 'supersecretkey'

# Configure server-side session
server.config['SESSION_TYPE'] = 'filesystem'
# Session(server)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    server=server
)
# server = app.server
app.title = 'YAPAT | Yet Another PAM Annotation Tool'

from src.pages.register import layout
app.layout = layout

register_callbacks()


@server.before_request
def initialize_session():
    if 'user_data' not in session:
        session['user_data'] = {
            'sampling_strategy': 'validate',
            'sampling_selected_species': [],
            'file_index_queue': [],  # list of tuples (file_index, file_path, audio, audio_concatenated)
            'file_index': None,
            'file_path': '',
            'sampling_start_time': [0, 0, 0, 0],
            'audio': [],
            'concatenated_audio': []
        }


def main():
    parser = argparse.ArgumentParser(description='Generate detections')
    parser.add_argument("--port", type=str, default="1050", help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    args = parser.parse_args()
    # run dashboard for annotation
    app.run_server(debug=False, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
