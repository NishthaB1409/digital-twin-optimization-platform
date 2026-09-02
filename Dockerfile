# syntax=docker/dockerfile:1
#
# Serving image for the DTMO scheduling API.
#
# The one thing worth knowing: torch must come from PyTorch's CPU index. The
# default PyPI wheel for Linux bundles CUDA and lands around 2.5 GB, which is
# pure waste here -- the policy is a 16-64-64-4 MLP and inference is far below
# the cost of a single HTTP round trip.

FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# CPU-only torch first, so the resolver never reaches for the CUDA wheel.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip \
 && pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-deps .


FROM python:3.10-slim AS runtime

# libgomp is what torch's fbgemm needs on Linux -- the same OpenMP dependency
# that surfaces on Windows as the missing VCOMP140.DLL.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 dtmo
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY configs/ ./configs/
COPY runs/ppo_shaped/ ./runs/ppo_shaped/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DTMO_CONFIG=/app/configs/factory.yaml \
    DTMO_MODEL=/app/runs/ppo_shaped/ppo_best

USER dtmo
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "dtmo.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
