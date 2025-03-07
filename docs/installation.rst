Installation
============

It is recommended to first set up a virtual environment using Conda:

.. code-block:: bash

    conda create -n dfttk python=3.12      
    conda activate dfttk

Clone the main branch of the repository:

.. code-block:: bash

    git clone https://github.com/PhasesResearchLab/dfttk.git

Or clone a specific branch:

.. code-block:: bash

    git clone -b <branch_name> https://github.com/PhasesResearchLab/dfttk.git

Then move to the `dfttk` directory and install in editable (`-e`) mode:

.. code-block:: bash

    cd dfttk
    pip install -e .
