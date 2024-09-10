Contributing
============

.. caution::
   Under construction.

Thank you for your interest in contributing to YAPAT! We appreciate all contributions, whether they are code, documentation, or feedback. To get started with contributing, please follow these steps:

**1. Fork the Repository**

Begin by forking the YAPAT repository to your own GitHub account. This will create a copy of the repository under your control where you can make changes.

**2. Clone Your Fork**

Clone your forked repository to your local machine using the following command:

.. code-block:: bash

   git clone https://github.com/your-username/yapat.git

Replace `your-username` with your GitHub username.

**3. Create a New Branch**

Navigate into the cloned repository and create a new branch based off the `dev` branch for your changes:

.. code-block:: bash

   cd yapat
   git checkout dev
   git checkout -b my-feature-branch

Replace `my-feature-branch` with a descriptive name for your branch.

**4. Make Your Changes**

Make your desired changes or additions. Ensure that your changes are well-tested and follow the project's coding guidelines.

**5. Commit Your Changes**

After making your changes, commit them with a clear and descriptive commit message:

.. code-block:: bash

   git add .
   git commit -m "Describe your changes here"

**6. Push Your Changes**

Push your changes to your forked repository on GitHub:

.. code-block:: bash

   git push origin my-feature-branch

**7. Create a Pull Request**

Go to the YAPAT repository on GitHub and open a pull request from your branch to the `dev` branch of the original repository. Provide a detailed description of your changes and any relevant information.

**8. Review and Feedback**

Once your pull request is submitted, the project maintainers will review your changes. Be prepared to discuss and make adjustments based on feedback.

**9. Merging**

Once your pull request is approved, it will be merged into the `dev` branch. You will be notified once your changes have been merged.

**10. Stay Updated**

To keep your fork updated with the latest changes from the original repository, you can set up an upstream remote and pull the latest changes:

.. code-block:: bash

   git remote add upstream https://github.com/yapat-app/yapat.git
   git fetch upstream
   git checkout dev
   git merge upstream/dev

For more detailed instructions or if you encounter any issues, please refer to the project's documentation or contact the maintainers.

Thank you for contributing to YAPAT!
