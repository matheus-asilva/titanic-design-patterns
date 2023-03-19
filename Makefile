help:
	@echo "Commands:"
	@echo "run           : runs the program."
	@echo "setup         : installs all the dependencies."
	@echo "clear-cache   : clears all the cache files."
	@echo "clear-venv    : clears all virtual environmentß files."
	@echo "style         : executes style formatting."
	@echo "create-venv   : creates a virtual environment."
	@echo "test          : runs coverage tests."

run:
	python src/main.py --model decision_tree

setup:
	pip install -r requirements.txt

clear-cache:
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf tests/.pytest_cache
	rm -rf .coverage
	rm -rf src/model/__pycache__


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
