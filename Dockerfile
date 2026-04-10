FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
  fastapi==0.111.0 \
  uvicorn[standard]==0.29.0 \
  pydantic==2.7.0 \
  pyyaml==6.0.1 \
  openenv-core>=0.2.0

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["python", "main.py"]
