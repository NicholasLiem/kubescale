package state_manager

import (
	"sync"
	"time"
)

type SpikeState struct {
	IsInSpike         bool
	SpikeStartTime    time.Time
	LastSpikeEndTime  time.Time
	SpikeRequestCount int
}

type StateManager struct {
	mu         sync.RWMutex
	spikeState SpikeState
}

func NewStateManager() *StateManager {
	return &StateManager{
		spikeState: SpikeState{
			IsInSpike:         false,
			SpikeStartTime:    time.Time{},
			LastSpikeEndTime:  time.Time{},
			SpikeRequestCount: 0,
		},
	}
}

func (sm *StateManager) StartSpike() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.spikeState.IsInSpike = true
	sm.spikeState.SpikeStartTime = time.Now()
	sm.spikeState.SpikeRequestCount++
}

func (sm *StateManager) EndSpike() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.spikeState.IsInSpike = false
	sm.spikeState.LastSpikeEndTime = time.Now()
}

func (sm *StateManager) IsInSpike() bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return sm.spikeState.IsInSpike
}

func (sm *StateManager) GetSpikeState() SpikeState {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return sm.spikeState
}
