package resource_manager

import (
	"context"
	"fmt"

	kubeclient "github.com/NicholasLiem/brain-controller/client"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type ResourceManager struct {
    clientSet *kubernetes.Clientset
}

func NewResourceManager() (*ResourceManager, error) {
    clientSet, err := kubeclient.GetKubernetesClient()
    if err != nil {
        return nil, fmt.Errorf("failed to initialize ResourceManager: %w", err)
    }

    return &ResourceManager{clientSet: clientSet}, nil
}

func (rm *ResourceManager) ScalePods(replicaCount int) error {
    deploymentName := "brain" // Name of the deployment
    namespace := "default"   // Namespace of the deployment

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