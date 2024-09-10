Deployment
===========

.. caution::
   Under construction.


To deploy the YAPAT application, you can use Docker along with Gunicorn. Follow the steps below to set up and run the application in a Docker container.

**1. Build the Docker Image**

First, ensure that you have Docker installed on your machine. Navigate to the root directory of the YAPAT project and build the Docker image with the following command:

.. code-block:: bash

   docker build -t yapat .

This command creates a Docker image named `yapat` based on the Dockerfile in the project root.

**2. Run the Docker Container**

Once the image is built, you can run a Docker container using the image. Use the following command to start the container:

.. code-block:: bash

   docker run -d \
     --name yapat-container \
     -p 1050:1050 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/projects:/app/projects \
     -v $(pwd)/instance:/app/instance \
     --env ENVIRONMENT_FILE=".env" \
     yapat

Here’s a breakdown of the options used:

- ``-d``: Run the container in detached mode.
- ``--name yapat-container``: Assign a name to the container.
- ``-p 1050:1050``: Map port 1050 on the host to port 1050 in the container.
- ``-v $(pwd)/data:/app/data``: Mount the `data` directory from the host to `/app/data` in the container.
- ``-v $(pwd)/projects:/app/projects``: Mount the `projects` directory from the host to `/app/projects` in the container.
- ``-v $(pwd)/instance:/app/instance``: Mount the `instance` directory from the host to `/app/instance` in the container.
- ``--env ENVIRONMENT_FILE=".env"``: Set the environment variable `ENVIRONMENT_FILE` to `.env`.
- ``yapat``: Specify the Docker image to use.

**3. Access the Application**

After starting the container, open your web browser and navigate to `http://localhost:1050` to access the YAPAT application.

**4. Stopping and Removing the Container**

To stop the running container, use:

.. code-block:: bash

   docker stop yapat-container

To remove the container, use:

.. code-block:: bash

   docker rm yapat-container

**5. Updating the Docker Image**

If you make changes to the application and need to update the Docker image, rebuild the image and restart the container:

.. code-block:: bash

   docker build -t yapat .
   docker stop yapat-container
   docker rm yapat-container
   docker run -d \
     --name yapat-container \
     -p 1050:1050 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/projects:/app/projects \
     -v $(pwd)/instance:/app/instance \
     --env ENVIRONMENT_FILE=".env" \
     yapat

**6. Configuration and Environment Variables**

Ensure that you have a `.env` file with the required environment variables for your application. This file should be placed in the root directory of your project or as specified by the `ENVIRONMENT_FILE` variable.

For more detailed configuration, refer to the Docker and Gunicorn documentation.

Feel free to adjust the instructions based on your specific deployment environment and requirements.
