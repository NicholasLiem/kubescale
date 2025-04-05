package callback

type ScaleRequest struct {
	ReplicaCount   int    `json:"replica_count"`
	DeploymentName string `json:"deployment_name"`
	Namespace      string `json:"namespace"`
}
