import logging
import socket
from contextlib import closing

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# %%
def check_socket(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0
