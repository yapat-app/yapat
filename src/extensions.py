from celery import Celery
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

# Initialize extensions without binding to the app yet
db = SQLAlchemy()
login_manager = LoginManager()


def make_celery(server):
    celery = Celery(
        "celery_" + server.import_name,
        backend=server.config['CELERY_RESULT_BACKEND'],
        broker=server.config['CELERY_BROKER_URL']
    )

    # celery.conf.update(server.config)

    # Ensure tasks are run within the server context
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with server.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
