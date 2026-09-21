.PHONY: help html multiversion doctest linkcheck

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SPHINXMULTIVERSION ?= sphinx-multiversion
SOURCEDIR     = source
BUILDDIR      = build

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/current" $(SPHINXOPTS) $(O)

multiversion:
	@$(SPHINXMULTIVERSION) "$(SOURCEDIR)" "$(BUILDDIR)/html"

doctest:
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
