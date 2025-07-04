package controller

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	callback "github.com/NicholasLiem/brain-controller/dto"
	"github.com/NicholasLiem/brain-controller/resource_manager"
	"github.com/NicholasLiem/brain-controller/state_manager"
	"github.com/gin-gonic/gin"
)

type MLController struct {
	resourceManager *resource_manager.ResourceManager
	stateManager    *state_manager.StateManager
}

func NewMLController(rm *resource_manager.ResourceManager, sm *state_manager.StateManager) *MLController {
	return &MLController{
		resourceManager: rm,
		stateManager:    sm,
	}
}

func (c *MLController) HandleUpdateTrafficSplit(ctx *gin.Context) {
	var request struct {
		DefaultPercent  int32 `json:"default_percent"`
		WarmPoolPercent int32 `json:"warm_pool_percent"`
	}

	if err := ctx.ShouldBindJSON(&request); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	if request.DefaultPercent+request.WarmPoolPercent != 100 {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Percentages must sum to 100"})
		return
	}

	err := c.resourceManager.UpdateTrafficSplit(request.DefaultPercent, request.WarmPoolPercent)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"message":           "Traffic split updated successfully",
		"default_percent":   request.DefaultPercent,
		"warm_pool_percent": request.WarmPoolPercent,
	})
}

func (c *MLController) HandleGetSpikeState(ctx *gin.Context) {
	spikeState := c.stateManager.GetSpikeMetrics()

	ctx.JSON(http.StatusOK, gin.H{
		"spike_state": spikeState,
	})
}

func (c *MLController) HandleSpikeForecast(ctx *gin.Context) {
	body, err := ctx.GetRawData()
	if err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request body"})
		return
	}

	var spikeForecast callback.SpikeForecast
	if err := json.Unmarshal(body, &spikeForecast); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON format"})
		return
	}

	// Debug the forecast
	if len(spikeForecast.Spikes) == 0 {
		ctx.JSON(http.StatusOK, gin.H{"message": "No spikes detected"})
		return
	}

	// Log the spike forecast for debugging
	fmt.Println("Received spike forecast:")
	for _, spike := range spikeForecast.Spikes {
		fmt.Printf("Index: %d, Time: %s, Value: %.2f, SpikeID: %d, Type: %s\n",
			spike.Index, spike.Time, spike.Value, spike.SpikeID, spike.Type)

		// Fix: Use the correct time format for parsing
		spikeTime, err := time.Parse("2006-01-02T15:04:05", spike.Time)
		if err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Invalid spike time format: %v", err)})
			return
		}

		fmt.Printf("Parsed spike time: %s\n", spikeTime)
		spikeTimeFromNow := spikeTime.Sub(time.Now()).Seconds()
		spike.TimeFromNow = spikeTimeFromNow
		fmt.Printf("Time from now: %.2f seconds\n", spike.TimeFromNow)
	}

	// Handle gradual adjustment of traffic weight
	if len(spikeForecast.Spikes) > 0 {
		for _, spike := range spikeForecast.Spikes {
			if spike.TimeFromNow < 0 {
				ctx.JSON(http.StatusBadRequest, gin.H{"error": "Spike time cannot be in the past"})
				return
			}

			if spike.Type == "BIG" {
				if !c.stateManager.IsInSpike() {
					timeToSpike := time.Duration(spike.TimeFromNow) * time.Second
					fmt.Printf("Spike detected: %s, Time to spike: %v\n", spike.Time, timeToSpike)

					if timeToSpike > 6*time.Minute {
						continue
					}

					preparationTime := 1 * time.Minute
					adjustedTimeToSpike := timeToSpike
					if timeToSpike > preparationTime {
						adjustedTimeToSpike = timeToSpike - preparationTime
					} else {
						adjustedTimeToSpike = 0
					}

					c.stateManager.StartSpike(adjustedTimeToSpike)
				}
				return
			}
		}
	}

	ctx.JSON(http.StatusOK, spikeForecast)
}

func (c *MLController) RegisterRoutes(router *gin.Engine) {
	mlCallbackGroup := router.Group("/ml-callback")
	{
		mlCallbackGroup.POST("/update-traffic-split", c.HandleUpdateTrafficSplit)
		mlCallbackGroup.POST("/spike-forecast", c.HandleSpikeForecast)
		mlCallbackGroup.GET("/spike-state", c.HandleGetSpikeState)
	}
}
