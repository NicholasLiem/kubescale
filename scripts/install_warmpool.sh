#!/bin/bash

# Set variables
NAMESPACE="warm-pool"
HELM_RELEASE_NAME="warm-pool"
HELM_CHART_PATH="./helm-charts/warm-pool"

# Check for required tools
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed. Please install it first."
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo "Error: helm is not installed. Please install it first."
    exit 1
fi

# Get DNS resolver IP
DNS_RESOLVER_IP=$(kubectl -n kube-system get svc kube-dns -o jsonpath='{.spec.clusterIP}')
if [ -z "$DNS_RESOLVER_IP" ]; then
    echo "Failed to get DNS resolver IP. Please check your cluster."
    exit 1
fi
echo "Using DNS resolver IP: $DNS_RESOLVER_IP"

# Install or upgrade the Helm release
echo "Installing/Upgrading Helm release..."
helm upgrade --install $HELM_RELEASE_NAME $HELM_CHART_PATH \
  --namespace $NAMESPACE \
  --set dnsResolverIP=$DNS_RESOLVER_IP \
  --create-namespace

# Check if the deployment is successful
echo "Checking deployment status..."
kubectl rollout status deployment/$HELM_RELEASE_NAME -n $NAMESPACE
if [ $? -eq 0 ]; then
    echo "Deployment successful!"
else
    echo "Deployment failed. Check the logs for more details."
    kubectl logs -l app=$HELM_RELEASE_NAME -n $NAMESPACE
fi

echo "MuBench Lite has been deployed to the '$NAMESPACE' namespace."
echo "You can access it with: kubectl port-forward svc/$HELM_RELEASE_NAME-service -n $NAMESPACE 8080:80"