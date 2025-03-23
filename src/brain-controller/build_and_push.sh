#!/bin/bash

# Set variables
APP_NAME="brain-controller"
REGISTRY="localhost:5000"
IMAGE_TAG="latest"

# Navigate to the directory where the Go code is located
cd src/brain-controller || exit

# Build the Go binary
echo "Building Go application..."
GOOS=linux GOARCH=amd64 go build -o $APP_NAME .

# Move back to the project root
cd ../../

# Build the Docker image
echo "Building Docker image..."
docker build -t $REGISTRY/$APP_NAME:$IMAGE_TAG -f src/brain-controller/Dockerfile src/brain-controller/

# Push the image to the local registry
echo "Pushing Docker image to local registry..."
docker push $REGISTRY/$APP_NAME:$IMAGE_TAG

echo "Done! Image available at $REGISTRY/$APP_NAME:$IMAGE_TAG"
