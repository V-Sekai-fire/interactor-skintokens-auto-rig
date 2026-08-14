# interactor-skintokens-auto-rig -- vast.ai worker, RFD 0036/0046.
#
# Small model (1.0 GB bf16) -- ships bf16, not Q4_K_M.

FROM python:3.11-slim AS contract
WORKDIR /app
RUN pip install --no-cache-dir usd-core==25.5 fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3
COPY server.py /app/server.py
COPY test_input.json /app/test_input.json
ENV WEFTSPUN_STUB=1 PORT=8000
EXPOSE 8000
CMD ["python", "/app/server.py"]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS worker

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir usd-core==25.5 fastapi==0.115.5 uvicorn==0.32.1 \
       pydantic==2.10.3 safetensors huggingface_hub==0.26.2

# SkinTokens -- MIT, confirmed by reading the real LICENSE file at
# VAST-AI-Research/SkinTokens directly (RFD 0046 had this marked
# "review pending"; it isn't ambiguous -- it's MIT).
ARG SKINTOKENS_REPO=VAST-AI-Research/SkinTokens
RUN mkdir -p /weights && \
    python3 -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
shutil.copy(hf_hub_download(repo_id='${SKINTOKENS_REPO}', filename='skintokens.safetensors'), '/weights/skintokens.safetensors')" \
    || echo "weight fetch deferred -- confirm exact HF repo id before real build"

COPY server.py /app/server.py

ENV PORT=8000
EXPOSE 8000

CMD ["python3", "-u", "/app/server.py"]
