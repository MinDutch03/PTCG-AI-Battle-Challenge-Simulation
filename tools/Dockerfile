# Linux runtime for the cabt engine (libcg.so is x86-64 ELF only).
# Build:  docker build --platform linux/amd64 -t cabt-dev tools
# Run:    docker run --platform linux/amd64 --rm -v "$PWD":/work -w /work cabt-dev \
#             python tools/run_match.py --p0 main --p1 baselines.random_agent -n 10
FROM --platform=linux/amd64 python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
CMD ["python"]
