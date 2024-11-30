import logging
import os
import socket
from contextlib import closing

from flask import Flask

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# %%
def check_socket(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0


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
    )

    return server


server = create_server()
