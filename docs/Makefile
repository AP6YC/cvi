# .PHONY: help Makefile
.PHONY: help html

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
# SPHINXBUILD   ?= sphinx-build
SPHINXBUILD   ?= sphinx-multiversion
SOURCEDIR     = source
BUILDDIR      = build/html

html:
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD) $(SOURCEDIR) $(BUILDDIR)

help:
# 	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD) help
