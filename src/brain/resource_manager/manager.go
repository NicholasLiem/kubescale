package resource_manager

import (
	"context"
	"fmt"
	"os"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type ResourceManager struct {
    clientSet *kubernetes.Clientset
}

func NewResourceManager(kubeClient *kubernetes.Clientset) (*ResourceManager, error) {
    fmt.Println("ResourceManager initialized successfully!")
    return &ResourceManager{clientSet: kubeClient}, nil
}

func (rm *ResourceManager) ScalePods(replicaCount int) error {
    deploymentName := os.Getenv("DEPLOYMENT_NAME")
	if deploymentName == "" {
		return fmt.Errorf("environment variable DEPLOYMENT_NAME is not set")
	}
    namespace := os.Getenv("NAMESPACE")
	if namespace == "" {
		return fmt.Errorf("environment variable NAMESPACE is not set")
	}

    scale, err := rm.clientSet.AppsV1().Deployments(namespace).GetScale(context.TODO(), deploymentName, metav1.GetOptions{})
    if err != nil {
        return fmt.Errorf("failed to get scale for deployment %s: %w", deploymentName, err)
    }

    scale.Spec.Replicas = int32(replicaCount)

    _, err = rm.clientSet.AppsV1().Deployments(namespace).UpdateScale(context.TODO(), deploymentName, scale, metav1.UpdateOptions{})
    if err != nil {
        return fmt.Errorf("failed to update scale for deployment %s: %w", deploymentName, err)
    }

    return nil
}