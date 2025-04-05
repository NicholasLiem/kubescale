package main

import (
	"fmt"
	"log"

	kubeclient "github.com/NicholasLiem/brain-controller/client"
	"github.com/NicholasLiem/brain-controller/resource_manager"
	"github.com/NicholasLiem/brain-controller/warm_pool_manager"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
    // Initialize GoDotEnv
    err := godotenv.Load()
    if err != nil {
        log.Fatalf("Error loading .env file")
    }

    // Initialize kubernetes client
    kubeClient, err := kubeclient.GetKubernetesClient()
    if err != nil {
        log.Fatalf("Failed to initialize Kubernetes client: %v", err)
    }

    // Initialize ResourceManager
    resourceManager, err := resource_manager.NewResourceManager(kubeClient)
    if err != nil {
        log.Fatalf("Failed to initialize ResourceManager: %v", err)
    }

    // Initialize WarmPoolManager
    warmPoolManager, err := warm_pool_manager.NewWarmPoolManager(kubeClient)
    if err != nil {
        log.Fatalf("Failed to initialize WarmPoolManager: %v", err)
    }

    // Initialize Gin router
    router := gin.Default()

    // Register routes
    RegisterRoutes(router, resourceManager, warmPoolManager)

    // Start the HTTP server
    port := "8080"
    fmt.Printf("Starting server on port %s...\n", port)
    log.Fatal(router.Run(":" + port))
}