import logging

logging.basicConfig(level=logging.INFO)

file_handler = logging.FileHandler('user_actions.log')  # Log to a file
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger_user_actions = (logging.getLogger(name='user_actions'))
logger_user_actions.addHandler(file_handler)

__all__ = ['logger_user_actions']