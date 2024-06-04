# DFTTK
## Documentation
A more complete description of DFTTK and its capabilities can be found in the documentation at [https://vasp-job-automation.readthedocs.io/en/latest/index.html](https://vasp-job-automation.readthedocs.io/en/latest/index.html)
## Summary
The **d**ensity **f**unctional **t**heory **t**ool**k**it is a python package for automating VASP jobs and storing relevant results on MongoDB. We currently have workflows for:  

- ENCUT and KPOINTS convergence 
- Energy-volume curves
- Phonons

These workflows are based on Custodian and PyMongo is used to store the results on MongoDB. 

## Installation
It is recommended to first set up a virtual environment using Conda:

    conda create -n dfttk python=3.12      
    conda activate dfttk

Clone the main brach of the repository:
    
    git clone https://github.com/lukeamyers/dfttk.git

Or clone a specific branch:
    
    git clone -b <branch_name> https://github.com/lukeamyers/dfttk.git

  Then move to `dfttk` directory and install in editable (`-e`) mode.

    cd dfttk
    pip install -e .

