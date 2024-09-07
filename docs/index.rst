.. Yapat documentation master file, created by
sphinx-quickstart on Sat Sep  7 16:58:12 2024.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

=======================
YAPAT Documentation
=======================

**AI-Driven PAM Annotation & Visualization Hub**

This yet another PAM annotation tool (YAPAT).
Designed for efficient analysis of PAM data, YAPAT utilizes machine learning to prioritize samples for expert annotation (as in `Kath et al., 2024 <https://www.sciencedirect.com/science/article/pii/S1574954124002528>`_).
The integrated interactive visualization suite combines embedding, dimensionality reduction, and clustering for dynamic data exploration.

.. toctree::

   usage
   api

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   docker
   contributing


Installation
============

To install the required dependencies, run the following:

.. code-block:: bash

   pip install -r requirements.txt

Usage
=====

To start the application, use:

.. code-block:: bash

   python main.py


Docker
======

YAPAT can also be run in a Docker container for an isolated environment.

To build the Docker image, run the following command in the root directory (where the `Dockerfile` is located):

.. code-block:: bash

   docker build -t yapat .

After building the image, run the container:

.. code-block:: bash

   docker run --env .env -p 1050:1050 -v $(pwd)/data:/data -v $(pwd)/projects:/projects -v $(pwd)/instance:/instance --rm yapat

This will run YAPAT inside a Docker container with the following options:

- ``--env .env``: Pass environment variables from the file `.env` to the container.
- ``-p 1050:1050``: Map port 1050 on the host to port 1050 on the container.
- ``-v $(pwd)/data:/data``: Mount the host directory `$(pwd)/data` to `/data` in the container.
- ``-v $(pwd)/projects:/projects``: Mount the host directory `$(pwd)/projects` to `/projects` in the container.
- ``-v $(pwd)/instance:/instance``: Mount the host directory `$(pwd)/instance` to `/instance` in the container.
- ``--rm``: Remove the container after it stops
- ``yapat``: The Docker image to use for the container.



Contributing
============

To contribute to YAPAT, follow these steps:

1. Fork the repository.
2. Clone your fork.
3. Create a new branch.
4. Make your changes.
5. Submit a pull request.

License
=======

This project is licensed under the GNU Affero General Public License version 3 (AGPL-3.0).

For more information, see the `LICENSE` file in the repository.

External Links
==============

- `GitHub Repository <https://github.com/yapat-app/yapat>`_
- `Issue Tracker <https://github.com/yapat-app/yapat/issues>`_


