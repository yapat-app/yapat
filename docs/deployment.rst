Deployment
===========

.. caution::
   Under construction (pre-alpha stage)


To deploy YAPAT using Docker and Gunicorn, follow these instructions to pull the Docker image and run it with Gunicorn.

**1. Pull the Docker Image**

First, pull the Docker image from the GitHub Container Registry:

.. code-block:: bash

   docker pull ghcr.io/yapat-app/yapat:latest

**2. Run the Docker Container**

To run the Docker container with Gunicorn, use the following command:

.. code-block:: bash

   docker run -d \
     -p 1050:1050 \
     -v $(pwd)/data:/data \
     -v $(pwd)/projects:/projects \
     -v $(pwd)/instance:/instance \
     --env ENVIRONMENT_FILE=".env" \
     ghcr.io/yapat-app/yapat:latest \
     gunicorn --config /app/gunicorn_config.py app:server

Explanation of the options used:

- ``-d``: Run the container in detached mode.
- ``-p 1050:1050``: Map port 1050 on the host to port 1050 in the container.
- ``-v $(pwd)/data:/data``: Mount the host directory `data` to `/data` in the container.
- ``-v $(pwd)/projects:/projects``: Mount the host directory `projects` to `/projects` in the container.
- ``-v $(pwd)/instance:/instance``: Mount the host directory `instance` to `/instance` in the container.
- ``--env ENVIRONMENT_FILE=".env"``: Pass environment variables from the `.env` file.
- ``ghcr.io/yapat-app/yapat:latest``: Specify the Docker image and version.
- ``gunicorn --config /app/gunicorn_config.py app:server``: Run Gunicorn with the specified configuration file and application module.

**3. Access the Application**

Once the container is running, you can access the YAPAT application by opening a web browser and navigating to:

`http://localhost:1050 <http://localhost:1050>`_

**4. Stopping the Container**

To stop and remove the Docker container, use the following command:

.. code-block:: bash

   docker stop $(docker ps -q --filter ancestor=ghcr.io/yapat-app/yapat:latest)
   docker rm $(docker ps -a -q --filter ancestor=ghcr.io/yapat-app/yapat:latest)

This will stop and remove all containers running the specified Docker image.

For more advanced configuration options and environment-specific settings, refer to the Docker and Gunicorn documentation or the project’s configuration files.