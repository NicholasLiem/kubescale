FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PROMETHEUS_URL="http://prometheus-kube-prometheus-prometheus.default.svc.cluster.local:9090"
ENV BRAIN_CONTROLLER_URL="http://brain.kube-scale.svc.cluster.local:8080"
ENV PORT=5000
ENV MLTPS_CONFIG_FILE="/app/mltps_config.json"

# Expose ports for Flask and Prometheus metrics
EXPOSE 5000 8000

# Run the application with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "1", "--timeout", "120", "--config", "gunicorn_config.py"]