#!/bin/bash

# Set variables
APP_NAME="mltps"
IMAGE_TAG="latest"
NAMESPACE="kube-scale"
HELM_RELEASE_NAME="mltps"
HELM_CHART_PATH="../../helm-charts/mltps"

# Point shell to Minikube's Docker daemon
echo "Connecting to Minikube's Docker daemon..."
eval $(minikube docker-env)

# Build the Docker image directly in Minikube
echo "Building Docker image in Minikube..."
docker build -t $APP_NAME:$IMAGE_TAG -f Dockerfile .

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Uninstall the existing Helm release (if it exists)
echo "Uninstalling existing Helm release (if any)..."
helm uninstall $HELM_RELEASE_NAME -n $NAMESPACE || echo "No existing release to uninstall."

# Install or upgrade the Helm release
echo "Installing Helm release..."
helm install $HELM_RELEASE_NAME $HELM_CHART_PATH --namespace $NAMESPACE || echo "Helm install failed. Trying upgrade..."
helm upgrade $HELM_RELEASE_NAME $HELM_CHART_PATH --namespace $NAMESPACE

# Check if the deployment is successful
echo "Checking deployment status..."
kubectl rollout status deployment/$APP_NAME -n $NAMESPACE
if [ $? -eq 0 ]; then
    echo "Deployment successful!"
else
    echo "Deployment failed. Check the logs for more details."
    kubectl logs -l app=$APP_NAME -n $NAMESPACE
fi

# Set up port forwarding for testing
POD_NAME=$(kubectl get pods -n $NAMESPACE --no-headers -o custom-columns=":metadata.name" | grep $APP_NAME | head -n 1)
if [ -z "$POD_NAME" ]; then
    echo "No pod found for the release."
    exit 1
fi
echo "Port forwarding to pod $POD_NAME..."
kubectl port-forward pod/$POD_NAME 5000:5000 -n $NAMESPACE