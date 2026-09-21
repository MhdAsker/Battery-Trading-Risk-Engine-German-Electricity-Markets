.PHONY: install test lint dashboard
install:
	python -m pip install -e ".[dev,dashboard]"
test:
	pytest
lint:
	ruff check .
dashboard:
	streamlit run dashboard/app.py

