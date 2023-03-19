run:
	python src/main.py

setup:
	pip install -r requirements.txt

clear-cache:
	rm -rf src/__pycache__

style:
	black src
	flake8 src
	python -m isort src

create-venv:
	python -m venv .venv
	# python -m pip install --upgrade pip
	# pip install -r requirements.txt

clear-venv:
	rm -rf .venv

test:
	coverage run -m pytest -v
	coverage report -m

help:
	@echo "Commands:"
	@echo "run     : runs the program."
	@echo "setup   : installs all the dependencies."
	@echo "clear   : clears all the cache files."
	@echo "style   : executes style formatting."
	@echo "venv    : creates a virtual environment."