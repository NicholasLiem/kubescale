package resource_manager

import (
	"context"
	"fmt"

	callback "github.com/NicholasLiem/brain-controller/dto"
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

func (rm *ResourceManager) ScalePods(scaleRequest callback.ScaleRequest) error {
    replicaCount := scaleRequest.ReplicaCount
    if replicaCount <= 0 {
        return fmt.Errorf("invalid replica count: %d", replicaCount)
    }

    deploymentName := scaleRequest.DeploymentName
    if deploymentName == "" {
        return fmt.Errorf("deployment name is required")
    }

    namespace := scaleRequest.Namespace
    if namespace == "" {
        return fmt.Errorf("namespace is required")
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