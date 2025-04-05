package kubeclient

import (
    "fmt"
    "os"
    "path/filepath"

    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/clientcmd"
    "k8s.io/client-go/rest"
)

func GetKubernetesClient() (*kubernetes.Clientset, error) {
    // Try to use in-cluster config
    config, err := rest.InClusterConfig()
    if err != nil {
        fmt.Println("In-cluster config not found, falling back to kubeconfig:", err)

        // Fall back to kubeconfig
        kubeconfig := filepath.Join(homeDir(), ".kube", "config")
        config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
        if err != nil {
            return nil, fmt.Errorf("failed to load kubeconfig: %v", err)
        }
    }

    // Create Kubernetes clientset
    clientset, err := kubernetes.NewForConfig(config)
    if err != nil {
        return nil, fmt.Errorf("failed to create Kubernetes clientset: %v", err)
    }

    return clientset, nil
}

func homeDir() string {
    if h := os.Getenv("HOME"); h != "" {
        return h
    }
    return os.Getenv("USERPROFILE")
}