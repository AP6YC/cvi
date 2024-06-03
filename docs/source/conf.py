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
release = '0.5.1'
version = '0.5.1'

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
    # 'sphinx.ext.autosectionlabel',
    # 'sphinx_autopackagesummary',
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

# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
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
    ]
}

# -- Options for EPUB output
epub_show_urls = 'footnote'


# Whitelist pattern for tags (set to None to ignore all tags)
# smv_tag_whitelist = r'^.*$'
# smv_tag_whitelist = None
smv_tag_whitelist = r'^v*$'

# Whitelist pattern for branches (set to None to ignore all branches)
# smv_branch_whitelist = r'^.*$
# smv_branch_whitelist = None
# smv_branch_whitelist = r'^(main|develop)$'
# smv_branch_whitelist = r'^self-host-docs$'
smv_branch_whitelist = r'^(main|develop|self-host-docs)$'

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = None

# Pattern for released versions
# smv_released_pattern = r'^tags/.*$'
smv_released_pattern = r'^tags/v*$'
# smv_released_pattern = None

# Format for versioned output directories inside the build directory
smv_outputdir_format = '{ref.name}'

# Determines whether remote or local git branches/tags are preferred if their output dirs conflict
smv_prefer_remote_refs = False

# # Skip param objects because of their weird rendering in docs
# def maybe_skip_member(app, what, name, obj, skip, options):
#     # print app, what, name, obj, skip, options
#     # if name == ""
#     return True

# def setup(app):
#     app.connect('autodoc-skip-member', maybe_skip_member)
