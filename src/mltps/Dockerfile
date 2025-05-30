FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PROMETHEUS_URL="http://prometheus-kube-prometheus-prometheus.default.svc.cluster.local:9090"
ENV BRAIN_CONTROLLER_URL="http://brain.kube-scale.svc.cluster.local:8080"
ENV PORT=5000
ENV MLTPS_CONFIG_FILE="/app/mltps_config.json"

# Optimize Python for numerical computing
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=2
ENV NUMBA_NUM_THREADS=2

# Expose ports for Flask and Prometheus metrics
EXPOSE 5000 8000

# Run with optimized settings for ML workloads
CMD ["gunicorn", "--config", "gunicorn_config.py", "app:app"]