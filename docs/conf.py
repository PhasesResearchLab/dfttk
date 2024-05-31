# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(o, os.path.abspath('.'))

project = 'dfttk'
copyright = '2024, Luke A. Myers, Nigel Lee En Hew'
author = 'Luke A. Myers, Nigel Lee En Hew'
language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.linkcode',
              'sphinx.ext.duration',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx_autodoc_typehints',
              'myst_nb',
              'sphinx_github_changelog',
              'sphinx_rtd_size'
              ]

# Jupyter Notebook configuration
nb_execution_mode = "off"
nb_execution_cache_path = "../temp/jupyter_cache"

# Changelog configuration
sphinx_github_changelog_token = password = os.environ.get("sphinx_github_changelog_token")


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for napoleon ----------------------------------------------------
napoleon_use_param = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
sphinx_rtd_size_width = "70%"

html_context = {
    "display_github": True,
    "github_user": "lukeamyers",
    "github_repo": "dfttk",
    "github_version": "documentation2",
    "conf_py_path": "/docs/",
}