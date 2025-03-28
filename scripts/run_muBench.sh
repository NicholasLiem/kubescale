#!/bin/bash
# filepath: setup_mubench.sh

echo "Setting up muBench environment..."

# Check if minikube and kubectl are installed
if ! command -v minikube &> /dev/null; then
    echo "Error: minikube is not installed. Please install it first."
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed. Please install it first."
    exit 1
fi

# Step 1: Make sure minikube is running
echo "Ensuring minikube is running..."
if ! minikube status | grep -q "Running"; then
    echo "Starting minikube..."
    minikube start
fi

# Check if container already exists and remove it if it does
if docker ps -a --format '{{.Names}}' | grep -q "^mubench$"; then
    echo "Removing existing muBench container..."
    docker rm -f mubench
fi

# Run the muBench container
echo "Running muBench container..."
docker run -it -id --platform linux/amd64 --network minikube --name mubench msvcbench/mubench


# Step 2: Get minikube IP
echo "Getting minikube IP..."
MINIKUBE_IP=$(minikube ip)
echo "Minikube IP: $MINIKUBE_IP"

# Step 3 & 4: Create kubectl config and modify the server address
echo "Creating and modifying kubectl config..."
kubectl config view --flatten > config_temp
sed "s|server: https://.*|server: https://$MINIKUBE_IP:8443|g" config_temp > config
rm config_temp

# Step 5: Copy config to muBench container
# echo "Copying minikube certificates to muBench container..."
# docker cp ~/.minikube/ca.crt mubench:/root/.minikube/ca.crt
# docker cp ~/.minikube/client.crt mubench:/root/.minikube/client.crt
# docker cp ~/.minikube/client.key mubench:/root/.minikube/client.key

echo "Copying kubectl config to muBench container..."
docker cp config mubench:/root/.kube/config
rm config

# Step 6 & 7: Enter muBench container and check pods
echo "Setup complete! You can now:"
echo "1. Enter the muBench container: docker exec -it mubench bash"
echo "2. Check running pods: kubectl get pods -A"
echo ""
echo "Or run the following to directly check pods from the container:"
echo "docker exec -it mubench kubectl get pods -A"
echo ""

# Offer to enter the container immediately
read -p "Would you like to enter the muBench container now? (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    docker exec -it mubench bash
fi