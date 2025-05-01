#!/bin/bash

# Set variables
APP_NAME="brain"
IMAGE_TAG="latest"
NAMESPACE="kube-scale"
HELM_RELEASE_NAME="brain"
HELM_CHART_PATH="../../helm-charts/brain"

# Point shell to Minikube's Docker daemon
echo "Connecting to Minikube's Docker daemon..."
eval $(minikube docker-env)

# Build the Docker image directly in Minikube
echo "Building Docker image in Minikube..."
docker build -t $APP_NAME:$IMAGE_TAG -f Dockerfile .

# No need to push when building directly in Minikube
echo "Done! Image '$APP_NAME:$IMAGE_TAG' is now available in Minikube's Docker environment"

# Uninstall the existing Helm release (if it exists)
echo "Uninstalling existing Helm release (if any)..."
helm uninstall $HELM_RELEASE_NAME || echo "No existing release to uninstall."

# Install or upgrade the Helm release
echo "Installing Helm release..."
helm install $HELM_RELEASE_NAME $HELM_CHART_PATH || echo "Helm install failed. Trying upgrade..."
helm upgrade $HELM_RELEASE_NAME $HELM_CHART_PATH

# Remind about Helm configuration
echo "Helm release '$HELM_RELEASE_NAME' has been installed/upgraded."

# Check if the deployment is successful
echo "Checking deployment status..."
kubectl rollout status deployment/$HELM_RELEASE_NAME -n $NAMESPACE
if [ $? -eq 0 ]; then
    echo "Deployment successful!"
else
    echo "Deployment failed. Check the logs for more details."
    kubectl logs -l app=$HELM_RELEASE_NAME
fi

# Checking health port
echo "Checking health port..."
# Get new pod name grep brain, get anything starts with brain
POD_NAME=$(kubectl get pods -n $NAMESPACE --no-headers -o custom-columns=":metadata.name" | grep $HELM_RELEASE_NAME | head -n 1)
if [ -z "$POD_NAME" ]; then
    echo "No pod found for the release."
    exit 1
fi
echo "Port forwarding to pod $POD_NAME..."
# Port forward to the pod
# Note: This will run in the background
# and you can access the service at http://localhost:8080
kubectl port-forward pod/$POD_NAME 8080:8080 -n $NAMESPACE