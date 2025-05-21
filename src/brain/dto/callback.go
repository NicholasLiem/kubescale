package callback

type ScaleRequest struct {
	ReplicaCount   int    `json:"replica_count"`
	DeploymentName string `json:"deployment_name"`
	Namespace      string `json:"namespace"`
}

type SpikeStateResponse struct {
	IsInSpike        bool   `json:"is_in_spike"`
	SpikeStartTime   string `json:"spike_start_time,omitempty"`
	LastSpikeEndTime string `json:"last_spike_end_time,omitempty"`
	SpikeCount       int    `json:"spike_count"`
}
