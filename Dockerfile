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

CMD ["python", "server/app.py"]
