Quickstart
==========


.. admonition:: **tl;dr**

   **Open a terminal window at your desired location and execute the following commands:**

   .. code-block:: bash

      # On macOS/Linux:

      python -m venv venv
      source venv/bin/activate
      pip install yapat
      yapat

   .. code-block:: bash

      # On Windows:

      python -m venv venv
      venv\Scripts\activate
      pip install yapat
      yapat

   **Finally, open the provided URL in your web browser.**



1. Create a Virtual Environment
--------------------------------
**(Optional)**

It's recommended to use a virtual environment to manage dependencies. You can create one using `venv`:

.. code-block:: bash

   python -m venv venv

Activate the virtual environment:

- **On Windows:**

  .. code-block:: bash

     venv\Scripts\activate

- **On macOS/Linux:**

  .. code-block:: bash

     source venv/bin/activate

2. Install
----------------

Once your virtual environment is activated (or if you’re not using one), install the YAPAT package using `pip`:

.. code-block:: bash

   pip install yapat

3. Run
------------

To start the YAPAT application, use the following command:

.. code-block:: bash

   yapat

4. Open the Application in Your Browser
----------------------------------------

After running the command, you will see a message indicating that the application is running and providing a local URL (e.g., `http://127.0.0.1:1050 <http://127.0.0.1:1050>`_). Open this URL in your web browser to access the YAPAT application.

You’re all set! You can now use YAPAT to manage and visualize your PAM data.

