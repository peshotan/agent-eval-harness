FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system evaluator \
    && adduser --system --ingroup evaluator evaluator

COPY pyproject.toml README.md ./
COPY harness ./harness
COPY cli.py ./cli.py

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN mkdir -p /app/artifacts \
    && chown -R evaluator:evaluator /app

USER evaluator

ENTRYPOINT ["python", "cli.py"]
CMD ["--version"]
