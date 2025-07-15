# DFTTK 
<p align="center">
    <img src="docs/_static/dfttk_logo.png" alt="DFTTK Logo">
</p>

[![GitHub Actions](https://github.com/PhasesResearchLab/dfttk/actions/workflows/test.yaml/badge.svg)](https://github.com/PhasesResearchLab/dfttk/actions/workflows/test.yaml)
[![Documentation Status](https://readthedocs.org/projects/dfttk/badge/?version=main)](https://www.dfttk.org/en/main/?badge=main)

## 📝 Overview

The **Density Functional Theory Toolkit (DFTTK)** is a Python package designed to automate VASP jobs and manage relevant results in MongoDB. VASP workflows leverage [Custodian](https://github.com/materialsproject/custodian), and data storage is handled via [PyMongo](https://github.com/mongodb/mongo-python-driver).

## 🔧 What does DFTTK do?

### Enumeration of Configurations
- Enumerates **unique collinear magnetic configurations** for a given structure.

### VASP Workflows
- Performs **convergence tests** for:
  - Cutoff energy (`ENCUT`)
  - k-points grid density (`kppa`)
- Computes **free energy** using the **quasiharmonic approximation**.

### MongoDB Storage
- Stores and retrieves VASP **input data** and **post-processed results** in MongoDB.

## ⚙️ Installation
It is recommended first to set up a virtual environment using Conda:

    conda create -n dfttk python=3.12      
    conda activate dfttk

Clone the main branch of the repository:
    
    git clone https://github.com/PhasesResearchLab/dfttk.git

Or clone a specific branch:
    
    git clone -b <branch_name> https://github.com/PhasesResearchLab/dfttk.git

  Then move to `dfttk` directory and install in editable (`-e`) mode.

    cd dfttk
    pip install -e .

> 🛠️ **Note:** A PyPI release is currently under development.

## 📖 Documentation

For a comprehensive description of **DFTTK** and its capabilities, please refer to the [Official Documentation](https://vasp-job-automation.readthedocs.io/en/latest/index.html).

> 🛠️ **Note:** The documentation is currently under construction. Some sections may be incomplete or subject to change.

## 📚 Citing DFTTK

If you use **DFTTK** in your work, please cite the following publication:

> **N. Hew et al.**,  
> *Density Functional Theory ToolKit (DFTTK) to automate first-principles thermodynamics via the quasiharmonic approximation*, **Computational Materials Science**, Volume 258, 2025, 114072, ISSN 0927-0256.  
> [https://doi.org/10.1016/j.commatsci.2025.114072](https://doi.org/10.1016/j.commatsci.2025.114072) ([View on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S092702562500415X))

## 🤝 Contributing

We welcome bug reports, feature suggestions, and pull requests!

### Getting Started
1. Fork and clone the repo:
   
       git clone https://github.com/<your-username>/dfttk.git

2. Create a new branch:

       git checkout -b my-feature

3. Make changes, commit, push, and open a pull request to `main`.

### 🐛 Reporting Issues
Found a bug or have a suggestion?  
Please open an issue at [GitHub Issues](https://github.com/PhasesResearchLab/dfttk/issues) with:
- A clear description
- Steps to reproduce (if applicable)
- Logs or screenshots

> Thanks for helping improve **DFTTK**!

