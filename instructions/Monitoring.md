# Install Grafana, Prometheus, Istio, Kiali and Integrate with muBench

1. Install the dependencies from `monitoring-install.sh` from muBench repository

2. Accessing from host browser
```sh
http://<MASTER_IP>:30000 for Prometheus
http://<MASTER_IP>:30001 for Grafana
http://<MASTER_IP>:30002 for Jaeger
http://<MASTER_IP>:30003 for Kiali
```

3. From the bash
```sh
minikube service -n monitoring prometheus-nodeport
minikube service -n monitoring grafana-nodeport
minikube service -n istio-system jaeger-nodeport
minikube service -n istio-system kiali-nodeport
```

4. Note: grafana username is `admin` and password is `prom-operator`