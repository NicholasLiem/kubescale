package resource_manager

import (
	"context"
	"fmt"
	"time"

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

// Configuration constants
const (
	DefaultNamespace      = "default"
	WarmPoolNamespace     = "warm-pool"
	MaxPodsPerService     = 3                                                                                                              // s0, s1, s2
	MaxServices           = 2                                                                                                              // maximum services per namespace
	DefaultCorePerPod     = 0.8                                                                                                            // 0.8 core per pod for default namespace
	WarmPoolCorePerPod    = 0.3                                                                                                            // 0.3 core per pod for warm pool namespace
	MaxTotalPods          = MaxPodsPerService * MaxServices                                                                                // 6 pods total (3 in default, 3 in warm pool)
	MaxTotalCore          = (DefaultCorePerPod * MaxPodsPerService * MaxServices) + (WarmPoolCorePerPod * MaxPodsPerService * MaxServices) // 6.6 cores total
	MinTrafficPercent     = 5                                                                                                              // minimum traffic percentage to prevent complete cut-off
	WarmUpIntervalSeconds = 15                                                                                                             // interval for warm-up steps
	RecoverySteps         = 4                                                                                                              // number of steps for traffic recovery
)

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

func (rm *ResourceManager) GetCurrentTotalReplicas(namespace string) (int32, error) {
	deployments, err := rm.GetDeployments(namespace)
	if err != nil {
		return 0, err
	}

	var totalReplicas int32
	for _, deployment := range deployments.Items {
		if deployment.Spec.Replicas != nil {
			totalReplicas += *deployment.Spec.Replicas
		}
	}

	return totalReplicas, nil
}

func (rm *ResourceManager) CalculateResourceBasedWeights(defaultReplicas, warmPoolReplicas int32) (int32, int32) {
	// Calculate current capacity for each namespace
	// Default: DefaultCorePerPod * actual_replicas
	defaultCapacity := float64(defaultReplicas) * DefaultCorePerPod

	// Warm-pool: WarmPoolCorePerPod * actual_replicas
	warmPoolCapacity := float64(warmPoolReplicas) * WarmPoolCorePerPod

	// Calculate total capacity
	totalCapacity := defaultCapacity + warmPoolCapacity

	// Avoid division by zero
	if totalCapacity == 0 {
		return 100, 0 // Default to 100% default if no capacity
	}

	// Calculate percentages based on capacity ratio
	defaultPercent := int32((defaultCapacity / totalCapacity) * 100)
	warmPoolPercent := int32((warmPoolCapacity / totalCapacity) * 100)

	// Ensure percentages sum to 100 (handle rounding)
	if defaultPercent+warmPoolPercent != 100 {
		if defaultPercent > warmPoolPercent {
			defaultPercent = 100 - warmPoolPercent
		} else {
			warmPoolPercent = 100 - defaultPercent
		}
	}

	// Apply minimum thresholds to prevent complete traffic cut-off
	if defaultPercent < MinTrafficPercent && defaultReplicas > 0 {
		defaultPercent = MinTrafficPercent
		warmPoolPercent = 100 - defaultPercent
	}
	if warmPoolPercent < MinTrafficPercent && warmPoolReplicas > 0 {
		warmPoolPercent = MinTrafficPercent
		defaultPercent = 100 - warmPoolPercent
	}

	return defaultPercent, warmPoolPercent
}

func (rm *ResourceManager) WarmUpTraffic(timeToWarmUp time.Duration, predictedEndTime time.Time) {
	fmt.Printf("Starting traffic warm-up for %v until %v\n", timeToWarmUp, predictedEndTime)

	// Calculate warm-up phases
	totalWarmUpSteps := int(timeToWarmUp.Seconds() / WarmUpIntervalSeconds) // 15-second intervals
	if totalWarmUpSteps == 0 {
		totalWarmUpSteps = 1
	}

	currentDefaultPercent := int32(100)
	currentWarmPoolPercent := int32(0)

	// Calculate maximum capacities using constants
	defaultMaxCapacity := DefaultCorePerPod * MaxPodsPerService * MaxServices   // 4.8
	warmPoolMaxCapacity := WarmPoolCorePerPod * MaxPodsPerService * MaxServices // 1.8
	totalMaxCapacity := defaultMaxCapacity + warmPoolMaxCapacity                // 6.6

	// Target traffic split based on max capacity ratio
	// Warm-pool target: 1.8 / 6.6 = ~27.27%
	// Default target: 4.8 / 6.6 = ~72.73%
	targetWarmPoolPercent := int32((warmPoolMaxCapacity / totalMaxCapacity) * 100) // ~27
	targetDefaultPercent := int32(100 - targetWarmPoolPercent)                     // ~73

	fmt.Printf("Target traffic split based on max capacity: %d%% default, %d%% warm pool\n",
		targetDefaultPercent, targetWarmPoolPercent)

	// Calculate step increments
	defaultDecrement := (currentDefaultPercent - targetDefaultPercent) / int32(totalWarmUpSteps)
	warmPoolIncrement := (targetWarmPoolPercent - currentWarmPoolPercent) / int32(totalWarmUpSteps)

	ticker := time.NewTicker(WarmUpIntervalSeconds * time.Second)
	defer ticker.Stop()

	step := 0
	warmUpCompleted := false

	for {
		select {
		case <-ticker.C:
			if !warmUpCompleted {
				step++
				if step >= totalWarmUpSteps {
					// Warm-up phase completed, switch to dynamic adjustment
					warmUpCompleted = true
					fmt.Println("Warm-up phase completed, switching to dynamic weight adjustment")
				}

				if !warmUpCompleted {
					// Continue gradual warm-up toward theoretical max capacity ratio
					newDefaultPercent := currentDefaultPercent - (defaultDecrement * int32(step))
					newWarmPoolPercent := currentWarmPoolPercent + (warmPoolIncrement * int32(step))

					// Ensure percentages sum to 100
					if newDefaultPercent+newWarmPoolPercent != 100 {
						newWarmPoolPercent = 100 - newDefaultPercent
					}

					err := rm.UpdateTrafficSplit(newDefaultPercent, newWarmPoolPercent)
					if err != nil {
						fmt.Printf("Error updating traffic split at step %d: %v\n", step, err)
						continue
					}

					fmt.Printf("Traffic warm-up step %d/%d: %d%% default, %d%% warm pool\n",
						step, totalWarmUpSteps, newDefaultPercent, newWarmPoolPercent)
				}
			}

			if warmUpCompleted {
				// Use dynamic weight adjustment based on actual replica counts
				rm.AdjustTrafficWeightDynamically()
			}

		case <-time.After(timeToWarmUp + 5*time.Second):
			// Safety timeout for warm-up phase only
			if !warmUpCompleted {
				fmt.Println("Traffic warm-up timeout reached")
				return
			}
		}

		// Check if we've reached the predicted end time
		if time.Now().After(predictedEndTime) {
			fmt.Println("Predicted spike end time reached, stopping dynamic adjustment")
			return
		}
	}
}

func (rm *ResourceManager) AdjustTrafficWeightDynamically() {
	// Get current replica counts for both namespaces
	defaultReplicas, err := rm.GetCurrentTotalReplicas(DefaultNamespace)
	if err != nil {
		fmt.Printf("Error getting default namespace replicas: %v\n", err)
		return
	}

	warmPoolReplicas, err := rm.GetCurrentTotalReplicas(WarmPoolNamespace)
	if err != nil {
		fmt.Printf("Error getting warm-pool namespace replicas: %v\n", err)
		return
	}

	// Calculate weights based on resource capacity
	defaultPercent, warmPoolPercent := rm.CalculateResourceBasedWeights(defaultReplicas, warmPoolReplicas)

	// Update traffic split
	err = rm.UpdateTrafficSplit(defaultPercent, warmPoolPercent)
	if err != nil {
		fmt.Printf("Error updating dynamic traffic split: %v\n", err)
		return
	}

	fmt.Printf("Dynamic weight adjustment: %d%% default (%d replicas), %d%% warm-pool (%d replicas)\n",
		defaultPercent, defaultReplicas, warmPoolPercent, warmPoolReplicas)
}

func (rm *ResourceManager) RecoverTraffic() error {
	fmt.Println("Starting traffic recovery - returning to normal distribution")

	// Get current traffic split from VirtualService
	vs, err := rm.istioClientSet.NetworkingV1alpha3().VirtualServices("istio-system").Get(
		context.TODO(), "traffic-split", metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get current VirtualService traffic-split: %w", err)
	}

	// Verify the http routes exist and have the expected structure
	if len(vs.Spec.Http) == 0 || len(vs.Spec.Http[0].Route) < 2 {
		return fmt.Errorf("virtual service doesn't have the expected route structure")
	}

	// Get current traffic percentages
	currentDefaultPercent := vs.Spec.Http[0].Route[0].Weight
	currentWarmPoolPercent := vs.Spec.Http[0].Route[1].Weight

	fmt.Printf("Current traffic split: %d%% default, %d%% warm pool\n",
		currentDefaultPercent, currentWarmPoolPercent)

	// Target: 100% default, 0% warm pool
	targetDefaultPercent := int32(100)
	targetWarmPoolPercent := int32(0)

	// Calculate increments per step
	defaultIncrement := (targetDefaultPercent - currentDefaultPercent) / RecoverySteps
	warmPoolDecrement := (currentWarmPoolPercent - targetWarmPoolPercent) / RecoverySteps

	for i := 1; i <= RecoverySteps; i++ {
		// Calculate new percentages
		newDefaultPercent := currentDefaultPercent + (defaultIncrement * int32(i))
		newWarmPoolPercent := currentWarmPoolPercent - (warmPoolDecrement * int32(i))

		// Ensure we reach exact target on final step
		if i == RecoverySteps {
			newDefaultPercent = targetDefaultPercent
			newWarmPoolPercent = targetWarmPoolPercent
		}

		// Ensure percentages sum to 100
		if newDefaultPercent+newWarmPoolPercent != 100 {
			newWarmPoolPercent = 100 - newDefaultPercent
		}

		err := rm.UpdateTrafficSplit(newDefaultPercent, newWarmPoolPercent)
		if err != nil {
			return fmt.Errorf("error during traffic recovery step %d: %w", i, err)
		}

		fmt.Printf("Recovery step %d/%d: %d%% default, %d%% warm pool\n",
			i, RecoverySteps, newDefaultPercent, newWarmPoolPercent)

		if i < RecoverySteps {
			time.Sleep(WarmUpIntervalSeconds * time.Second)
		}
	}

	fmt.Println("Traffic recovery completed - returned to normal distribution")
	return nil
}
