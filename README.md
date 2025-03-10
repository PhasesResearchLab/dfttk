# DFTTK
<p align="center">
    <img src="docs/_static/dfttk_logo.png" alt="DFTTK Logo">
</p>

[![GitHub Actions](https://github.com/PhasesResearchLab/dfttk/actions/workflows/test.yaml/badge.svg)](https://github.com/PhasesResearchLab/dfttk/actions/workflows/test.yaml)
[![Documentation Status](https://readthedocs.org/projects/dfttk/badge/?version=main)](https://www.dfttk.org/en/main/?badge=main)

## Overview
The **D**ensity **F**unctional **T**heory **T**ool**K**it is a Python package for automating VASP jobs and storing relevant results on MongoDB. The VASP workflows are based on [Custodian](https://github.com/materialsproject/custodian), and [PyMongo](https://github.com/mongodb/mongo-python-driver) is used to store the results on MongoDB.

## What does DFTTK do?

**Enumeration of Configurations**

- Enumerates unique collinear magnetic configurations for a given structure.

**VASP Workflows**

- Performs convergence tests for cutoff energy (ENCUT) and k-points grid density (kppa).
- Computes free energy using the quasiharmonic approximation.

**MongoDB Storage**

- Stores and retrieves VASP input data and post-processed results in MongoDB.


## Installation
It is recommended to first set up a virtual environment using Conda:

    conda create -n dfttk python=3.12      
    conda activate dfttk

Clone the main brach of the repository:
    
    git clone https://github.com/PhasesResearchLab/dfttk.git

Or clone a specific branch:
    
    git clone -b <branch_name> https://github.com/PhasesResearchLab/dfttk.git

  Then move to `dfttk` directory and install in editable (`-e`) mode.

    cd dfttk
    pip install -e .

## Documentation
A more complete description of DFTTK and its capabilities can be found in the [documentation](https://vasp-job-automation.readthedocs.io/en/latest/index.html). 
