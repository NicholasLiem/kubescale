package resource_manager

import (
	"context"
	"fmt"

	callback "github.com/NicholasLiem/brain-controller/dto"
	istioclientset "istio.io/client-go/pkg/clientset/versioned"
	v1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

type ResourceManager struct {
	clientSet      *kubernetes.Clientset
	istioClientSet *istioclientset.Clientset
}

func NewResourceManager(kubeClient *kubernetes.Clientset, istioClient *istioclientset.Clientset) (*ResourceManager, error) {
	fmt.Println("ResourceManager initialized successfully!")
	return &ResourceManager{clientSet: kubeClient, istioClientSet: istioClient}, nil
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

func (rm *ResourceManager) UpdateTrafficSplit(defaultPercent, warmPoolPercent int32) error {
	if defaultPercent+warmPoolPercent != 100 {
		return fmt.Errorf("traffic split percentages must sum to 100, got %d and %d", defaultPercent, warmPoolPercent)
	}

	vs, err := rm.istioClientSet.NetworkingV1alpha3().VirtualServices("istio-system").Get(
		context.TODO(), "traffic-split", metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get VirtualService traffic-split: %w", err)
	}

	// Verify the http routes exist and have the expected structure
	if len(vs.Spec.Http) == 0 || len(vs.Spec.Http[0].Route) < 2 {
		return fmt.Errorf("virtual service doesn't have the expected route structure")
	}

	// Update the weights
	vs.Spec.Http[0].Route[0].Weight = defaultPercent
	vs.Spec.Http[0].Route[1].Weight = warmPoolPercent

	_, err = rm.istioClientSet.NetworkingV1alpha3().VirtualServices("istio-system").Update(
		context.TODO(), vs, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update VirtualService: %w", err)
	}

	fmt.Printf("Traffic split updated: %d%% to default, %d%% to warm pool\n",
		defaultPercent, warmPoolPercent)
	return nil
}

func (rm *ResourceManager) GetCurrentScale(deploymentName, namespace string) (int32, error) {
	if deploymentName == "" {
		return 0, fmt.Errorf("deployment name is required")
	}

	if namespace == "" {
		return 0, fmt.Errorf("namespace is required")
	}

	scale, err := rm.clientSet.AppsV1().Deployments(namespace).GetScale(
		context.TODO(), deploymentName, metav1.GetOptions{})
	if err != nil {
		return 0, fmt.Errorf("failed to get scale for deployment %s: %w", deploymentName, err)
	}

	return scale.Spec.Replicas, nil
}

func (rm *ResourceManager) GetDeployments(namespace string) (*v1.DeploymentList, error) {
	if namespace == "" {
		return nil, fmt.Errorf("namespace is required")
	}

	deployments, err := rm.clientSet.AppsV1().Deployments(namespace).List(
		context.TODO(),
		metav1.ListOptions{},
	)

	if err != nil {
		return nil, fmt.Errorf("failed to list deployments in namespace %s: %w", namespace, err)
	}

	return deployments, nil
}
