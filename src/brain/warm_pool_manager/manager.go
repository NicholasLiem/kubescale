package warm_pool_manager

import (
	"fmt"

	kubeclient "github.com/NicholasLiem/brain-controller/client"
	"k8s.io/client-go/kubernetes"
)

type WarmPoolManager struct {
    clientSet *kubernetes.Clientset
}

// NewWarmPoolManager initializes the WarmPoolManager with a Kubernetes client
func NewWarmPoolManager() (*WarmPoolManager, error) {
    clientSet, err := kubeclient.GetKubernetesClient()
    if err != nil {
        return nil, fmt.Errorf("failed to initialize WarmPoolManager: %w", err)
    }

    return &WarmPoolManager{clientSet: clientSet}, nil
}