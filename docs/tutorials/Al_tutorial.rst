Al Tutorial
===========

The Jupyter Notebook for this tutorial can be found in the `examples/Al` directory of the DFTTK repository. Before starting this tutorial, ensure the following prerequisites are completed:

- **Install DFTTK**: `DFTTK GitHub Repository <https://github.com/PhasesResearchLab/dfttk>`_
- **Install YPHON**: `YPHON GitHub Repository <https://github.com/PhasesResearchLab/YPHON>`_
- **Set up POTCAR for pymatgen**: Follow the instructions at `pymatgen POTCAR Setup <https://pymatgen.org/installation.html#potcar-setup>`_

## Import Necessary Libraries

Before running the tutorial, ensure the required libraries are imported. Use the following code:

.. code-block:: python

   # Enable automatic reloading of modules
   %reload_ext autoreload
   %autoreload 2

   # Standard Library Imports
   import os
   import subprocess

   # Third-Party Library Imports
   import numpy as np
   import plotly.graph_objects as go

   # DFTTK Imports
   from dfttk.config import Configuration
   from dfttk.plotly_format import plot_format