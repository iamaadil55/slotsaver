# SlotSaver scoring API container.
# Build:  docker build -t slotsaver .
# Run:    docker run -p 8000:8000 slotsaver                       (synthetic demo mode)
#         docker run -p 8000:8000 -v ./data:/app/data slotsaver   (real data mounted)
FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api.py .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
