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

type SpikeForecast struct {
	Success bool        `json:"success"`
	Spikes  []SpikeInfo `json:"spikes"`
}

type SpikeInfo struct {
	Index       int     `json:"index"`
	Time        string  `json:"time"`
	Value       float64 `json:"value"`
	SpikeID     int     `json:"spike_id"`
	Type        string  `json:"type"`
	TimeFromNow float64 `json:"time_from_now"`
}
