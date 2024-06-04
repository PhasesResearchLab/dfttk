# DFTTK
## Documentation
A more complete description of DFTTK and its capabilities can be found in the documentation at https://vasp-job-automation.readthedocs.io/en/latest/index.html
## Summary
The **d**ensity **f**unctional **t**heory **t**ool**k**it is a python package for automating VASP jobs and storing relevant results on MongoDB. We currently have workflows for:  

- ENCUT and KPOINTS convergence 
- Energy-volume curves
- Phonons

These workflows are based on Custodian and PyMongo is used to store the results on MongoDB. 

## Installation
It is recommended to first set up a virtual environment using Conda:

    conda create -n vasp-job-automation python=3.11      
    conda activate vasp-job-automation

Clone the main brach of the repository:
    
    git clone https://<your_username>:<your_personal_access_token>@github.com/lukeamyers/vasp-job-automation.git

Or clone a specific branch:
    
    git clone -b <branch_name> https://<your_username>:<your_personal_access_token>@github.com/lukeamyers/vasp-job-automation.git

  Then move to `vasp-job-automation` directory and install in editable (`-e`) mode.

    cd vasp-job-automation
    pip install -e .

