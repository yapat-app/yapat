Quickstart Guide
================

This guide will help you get started with YAPAT.

Installation
------------

To install the required dependencies, run:

.. code-block:: bash

   pip install -r requirements.txt

Basic Usage
-----------

To start the application:

.. code-block:: bash

   python main.py

Running with Docker
-------------------

To run YAPAT in Docker:

.. code-block:: bash

   docker build -t yapat .
   docker run --env .env -p 1050:1050 -v $(pwd)/data:/data -v $(pwd)/projects:/projects -v $(pwd)/instance:/instance --rm yapat
