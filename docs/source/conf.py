# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../src/cvi'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cvi'
copyright = '2024, Sasha Petrenko'
author = 'Sasha Petrenko'
release = '0.6.0'
version = '0.6.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    "sphinx_multiversion",
]

autosummary_generate_overwrite = True

autodoc_inherit_docstrings = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

intersphinx_disabled_domains = ['std']


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'

html_static_path = ['_static']

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "versioning.html",
    ],
}

# -- Options for EPUB output
epub_show_urls = 'footnote'


# Ignore tags for now
smv_tag_whitelist = None

# Build docs for main and develop, whether sphinx-multiversion sees them
# as local branches or as remote branches.
smv_branch_whitelist = r'^(origin/)?(main|develop)$'

# Allow origin/main and origin/develop.
smv_remote_whitelist = r'^origin$'

# No tags are being built, so this does not matter much right now.
smv_released_pattern = r'^tags/.*$'

# Use simple output directories: main, develop.
smv_outputdir_format = '{ref.name}'

# Prefer remote refs in CI, since GitHub Actions reliably has origin/main
# and origin/develop after fetching.
smv_prefer_remote_refs = True

