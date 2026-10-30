PYTHON ?= python3

.PHONY: check
check:
	$(PYTHON) -m unittest discover -s tests -v
