FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PROMETHEUS_URL="http://prometheus-kube-prometheus-prometheus.default.svc.cluster.local:9090"
ENV BRAIN_CONTROLLER_URL="http://brain.kube-scale.svc.cluster.local:8080"
ENV PORT=5000

# Expose ports for Flask and Prometheus metrics
EXPOSE 5000 8000

# Run the application with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2", "--timeout", "120"]