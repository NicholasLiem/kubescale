package main

import (
	"fmt"
	"log"
	"net/http"
	"strconv"

	"github.com/NicholasLiem/brain-controller/resource_manager"
	"github.com/NicholasLiem/brain-controller/warm_pool_manager"
	"github.com/joho/godotenv"
)

func main() {
	// Initialize GoDotEnv
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file")
	}
    // Initialize ResourceManager
    resourceManager, err := resource_manager.NewResourceManager()
    if err != nil {
        log.Fatalf("Failed to initialize ResourceManager: %v", err)
    }
    fmt.Println("ResourceManager initialized successfully!")

    // Initialize WarmPoolManager
    _, err = warm_pool_manager.NewWarmPoolManager()
    if err != nil {
        log.Fatalf("Failed to initialize WarmPoolManager: %v", err)
    }
    fmt.Println("WarmPoolManager initialized successfully!")

    http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        fmt.Fprintf(w, "OK")
    })

	http.HandleFunc("/scale", func(w http.ResponseWriter, r *http.Request) {
        replicasStr := r.URL.Query().Get("replicas")
        if replicasStr == "" {
            http.Error(w, "Missing 'replicas' query parameter", http.StatusBadRequest)
            return
        }

        replicas, err := strconv.Atoi(replicasStr)
        if err != nil {
            http.Error(w, "Invalid 'replicas' value", http.StatusBadRequest)
            return
        }

        err = resourceManager.ScalePods(replicas)
        if err != nil {
            http.Error(w, fmt.Sprintf("Failed to scale replicas: %v", err), http.StatusInternalServerError)
            return
        }

        fmt.Fprintf(w, "Successfully scaled to %d replicas", replicas)
    })

    // Start the HTTP server
    port := "8080"
    fmt.Printf("Starting server on port %s...\n", port)
    log.Fatal(http.ListenAndServe(":"+port, nil))
}