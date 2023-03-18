run:
	python src/main.py

setup:
	pip install -r requirements.txt

clear:
	rm -rf src/__pycache__

style:
	black src
	flake8 src
	python -m isort src

venv:
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt

help:
	@echo "Commands:"
	@echo "run     : runs the program."
	@echo "setup   : installs all the dependencies."
	@echo "clear   : clears all the cache files."
	@echo "style   : executes style formatting."
	@echo "venv    : creates a virtual environment."