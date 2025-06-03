package state_manager

import (
	"fmt"
	"sync"
	"time"

	"github.com/NicholasLiem/brain-controller/resource_manager"
)

type SpikeState struct {
	IsInSpike             bool
	SpikeStartTime        time.Time
	LastSpikeEndTime      time.Time
	PredictedSpikeEndTime time.Time
	PredictedSpikeTime    time.Time
	TimeToSpike           time.Duration
}

type StateManager struct {
	mu              sync.RWMutex
	spikeState      SpikeState
	resourceManager *resource_manager.ResourceManager
}

func NewStateManager(resourceManager *resource_manager.ResourceManager) *StateManager {
	return &StateManager{
		spikeState: SpikeState{
			IsInSpike:         false,
			SpikeStartTime:    time.Time{},
			LastSpikeEndTime:  time.Time{},
		},
		resourceManager: resourceManager,
	}
}

func (sm *StateManager) StartSpike(timeToSpike time.Duration) {
    sm.mu.Lock()
    defer sm.mu.Unlock()

    if sm.spikeState.IsInSpike {
        return
    }

    sm.spikeState.IsInSpike = true
    sm.spikeState.SpikeStartTime = time.Now()
    sm.spikeState.TimeToSpike = timeToSpike
    sm.spikeState.PredictedSpikeTime = sm.spikeState.SpikeStartTime.Add(timeToSpike)
    
    // Auto recovery 90 seconds after predicted spike time
    autoRecoveryBuffer := 90 * time.Second
    sm.spikeState.PredictedSpikeEndTime = sm.spikeState.PredictedSpikeTime.Add(autoRecoveryBuffer)

    totalWaitTime := timeToSpike + autoRecoveryBuffer
    fmt.Printf("Spike started - predicted spike at %v, auto-recovery at %v (total wait: %v)\n", 
        sm.spikeState.PredictedSpikeTime.Format("15:04:05"), 
        sm.spikeState.PredictedSpikeEndTime.Format("15:04:05"),
        totalWaitTime)

    go sm.resourceManager.WarmUpTraffic(sm.spikeState.TimeToSpike, sm.spikeState.PredictedSpikeEndTime)
    
    go sm.autoRecoveryWatcher()
}

func (sm *StateManager) autoRecoveryWatcher() {
    // Wait until predicted spike end time
    timeUntilAutoRecovery := time.Until(sm.spikeState.PredictedSpikeEndTime)
    
    if timeUntilAutoRecovery > 0 {
        time.Sleep(timeUntilAutoRecovery)
    }
    
    // Check if still in spike state before triggering recovery
    sm.mu.RLock()
    isStillInSpike := sm.spikeState.IsInSpike
    sm.mu.RUnlock()
    
    if isStillInSpike {
        fmt.Printf("Auto-recovery triggered at %v - ending spike and starting gradual traffic recovery\n", 
            time.Now().Format("15:04:05"))
        sm.EndSpike()
    }
}

// In question
func (sm *StateManager) CheckMetricsAndRecover(currentCPUUsage float64, thresholdCPU float64) {
    sm.mu.RLock()
    isInSpike := sm.spikeState.IsInSpike
    spikeStartTime := sm.spikeState.SpikeStartTime
    sm.mu.RUnlock()

    if !isInSpike {
        return
    }

    // Check if we've been in spike for minimum duration (30 seconds)
    minSpikeDuration := 30 * time.Second
    if time.Since(spikeStartTime) < minSpikeDuration {
        return
    }

    // Check if metrics indicate recovery is possible
    if currentCPUUsage < thresholdCPU {
        fmt.Printf("Metrics indicate recovery possible - CPU: %.2f%% < threshold: %.2f%%\n", 
            currentCPUUsage, thresholdCPU)
        sm.EndSpike()
    }
}

func (sm *StateManager) IsInSpike() bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.spikeState.IsInSpike
}

func (sm *StateManager) EndSpike() {
    sm.mu.Lock()
    defer sm.mu.Unlock()

    if !sm.spikeState.IsInSpike {
        // Not in spike state
        return
    }

    sm.spikeState.IsInSpike = false
    sm.spikeState.LastSpikeEndTime = time.Now()

    spikeDuration := sm.spikeState.LastSpikeEndTime.Sub(sm.spikeState.SpikeStartTime)
    fmt.Printf("Spike ended - duration: %v, starting traffic recovery\n", spikeDuration)

    // Start traffic recovery in a separate goroutine
    go func() {
        err := sm.resourceManager.RecoverTraffic()
        if err != nil {
            fmt.Printf("Error during traffic recovery: %v\n", err)
        }
    }()
}

func (sm *StateManager) GetSpikeMetrics() map[string]interface{} {
    sm.mu.RLock()
    defer sm.mu.RUnlock()

    metrics := map[string]interface{}{
        "is_in_spike":          sm.spikeState.IsInSpike,
        "last_spike_end_time":  sm.spikeState.LastSpikeEndTime,
    }

    if sm.spikeState.IsInSpike {
        metrics["spike_start_time"] = sm.spikeState.SpikeStartTime
        metrics["predicted_spike_end_time"] = sm.spikeState.PredictedSpikeEndTime
        metrics["time_to_spike"] = sm.spikeState.TimeToSpike
        metrics["current_spike_duration"] = time.Since(sm.spikeState.SpikeStartTime)
    }

    return metrics
}
