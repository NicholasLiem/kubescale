package warm_pool_manager

import (
	"fmt"

	"k8s.io/client-go/kubernetes"
)

type WarmPoolManager struct {
    clientSet *kubernetes.Clientset
}

func NewWarmPoolManager(kubeClient *kubernetes.Clientset) (*WarmPoolManager, error) {
    fmt.Println("WarmPoolManager initialized successfully!")
    return &WarmPoolManager{clientSet: kubeClient}, nil
}