Running with Docker
===================

.. caution::
   Under construction.

To run the YAPAT application locally using Docker, follow these steps to pull and start the Docker container.

**1. Install Docker**

Ensure that Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/get-started).

**2. Pull the Docker Image**

Instead of building the Docker image, you can pull the pre-built Docker image from the GitHub Container Registry:

.. code-block:: bash

   docker pull ghcr.io/yapat-app/yapat:0.1.0-alpha

Replace `0.1.0-alpha` with the desired version tag if needed.

**3. Run the Docker Container**

Run the Docker container using the image you just pulled. Use the following command:

.. code-block:: bash

   docker run -d \
     --name yapat-container \
     -p 1050:1050 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/projects:/app/projects \
     -v $(pwd)/instance:/app/instance \
     --env ENVIRONMENT_FILE=".env" \
     ghcr.io/yapat-app/yapat:0.1.0-alpha

Here’s what each option does:

- ``-d``: Run the container in detached mode.
- ``--name yapat-container``: Assign a name to the container.
- ``-p 1050:1050``: Map port 1050 on your host to port 1050 in the container.
- ``-v $(pwd)/data:/app/data``: Mount the `data` directory from your host to `/app/data` in the container.
- ``-v $(pwd)/projects:/app/projects``: Mount the `projects` directory from your host to `/app/projects` in the container.
- ``-v $(pwd)/instance:/app/instance``: Mount the `instance` directory from your host to `/app/instance` in the container.
- ``--env ENVIRONMENT_FILE=".env"``: Pass the environment variable `ENVIRONMENT_FILE` to the container.
- ``ghcr.io/yapat-app/yapat:0.1.0-alpha``: Use the Docker image from the GitHub Container Registry.

**4. Access the Application**

Once the container is running, you can access the YAPAT application by opening a web browser and navigating to `http://localhost:1050`.

**5. Stopping and Removing the Container**

To stop the running container, execute:

.. code-block:: bash

   docker stop yapat-container

To remove the container, run:

.. code-block:: bash

   docker rm yapat-container

**6. Re-deploy with New Versions**

If a new version of the Docker image is available or if you need to update the container:

.. code-block:: bash

   docker pull ghcr.io/yapat-app/yapat:0.1.0-alpha
   docker stop yapat-container
   docker rm yapat-container
   docker run -d \
     --name yapat-container \
     -p 1050:1050 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/projects:/app/projects \
     -v $(pwd)/instance:/app/instance \
     --env ENVIRONMENT_FILE=".env" \
     ghcr.io/yapat-app/yapat:0.1.0-alpha

**7. Configuration**

Make sure you have a `.env` file in the root directory of your project or adjust the `ENVIRONMENT_FILE` environment variable as needed. This file should contain any necessary environment variables for the application.

For more detailed configuration and usage, refer to the Docker documentation.

These instructions will help you get YAPAT up and running locally using Docker.

