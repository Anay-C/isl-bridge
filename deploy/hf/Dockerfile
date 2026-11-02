# ISL Bridge on a Hugging Face Space (Docker SDK, free CPU). Built by deploy/build_space.py, which copies the
# runtime code + data next to this file. Local test: docker build -t isl . && docker run -p 7860:7860 isl
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    ISL_PUBLIC=1 ISL_DEVICE=cpu TORCH_THREADS=2 OMP_NUM_THREADS=2 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir "torch==2.11.*" --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
EXPOSE 7860
# --proxy-headers: the Space's proxy sets X-Forwarded-For (per-device rate limits use it)
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--ws-max-size", "4194304"]
