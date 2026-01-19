PYSRC = src

# Python
checkformat:
	isort --check-only ${PYSRC}/vlagents
	black --check ${PYSRC}/vlagents

format:
	isort ${PYSRC}
	black ${PYSRC}

lint: ruff mypy

ruff:
	ruff check ${PYSRC}

mypy:
	mypy ${PYSRC} --install-types --non-interactive --no-namespace-packages

test:
	pytest ${PYSRC} -vv

bump:
	cz bump

commit:
	cz commit

.PHONY: checkformat format lint ruff mypy pytest bump commit
