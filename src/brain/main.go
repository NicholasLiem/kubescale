package main

import (
	"fmt"
	"log"
	"os"

	kubeclient "github.com/NicholasLiem/brain-controller/client"
	"github.com/NicholasLiem/brain-controller/resource_manager"
	"github.com/NicholasLiem/brain-controller/services"
	"github.com/NicholasLiem/brain-controller/state_manager"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	istioclientset "istio.io/client-go/pkg/clientset/versioned"
)

func main() {
	// Initialize GoDotEnv
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file")
	}

	// Initialize kubernetes client
	kubeClient, restConfig, err := kubeclient.GetKubernetesClientAndConfig()
	if err != nil {
		log.Fatalf("Failed to initialize Kubernetes client: %v", err)
	}

	istioClient, err := istioclientset.NewForConfig(restConfig)
	if err != nil {
		log.Fatalf("Failed to initialize Istio client: %v", err)
	}

	// Initialize Prometheus service
	prometheusURL := os.Getenv("PROMETHEUS_URL")
    if prometheusURL == "" {
        prometheusURL = "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"
    }
    prometheusService := services.NewPrometheusService(prometheusURL)

	// Initialize ResourceManager
	resourceManager, err := resource_manager.NewResourceManager(kubeClient, istioClient, *prometheusService)
	if err != nil {
		log.Fatalf("Failed to initialize ResourceManager: %v", err)
	}

	// Create a new state manager
	stateManager := state_manager.NewStateManager(resourceManager)

	// Initialize Gin router
	router := gin.Default()

	// Register routes
	RegisterRoutes(router, resourceManager, stateManager, kubeClient, prometheusService)

	// Start the HTTP server
	port := "8080"
	fmt.Printf("Starting server on port %s...\n", port)
	log.Fatal(router.Run(":" + port))
}
