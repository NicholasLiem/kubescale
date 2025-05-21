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

func (c *MLController) HandleScale(ctx *gin.Context) {
	body, err := ctx.GetRawData()
	if err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request body"})
		return
	}

	var scaleRequest callback.ScaleRequest
	if err := json.Unmarshal(body, &scaleRequest); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON format"})
		return
	}

	// Check if this is a scale down request
	currentScale, err := c.resourceManager.GetCurrentScale(scaleRequest.DeploymentName, scaleRequest.Namespace)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to get current scale: %v", err)})
		return
	}

	if int32(scaleRequest.ReplicaCount) < currentScale && c.stateManager.IsInSpike() {
		ctx.JSON(http.StatusConflict, gin.H{
			"message":       "Cannot scale down during spike event",
			"current_state": "in_spike",
		})
		return
	}

	// If it's a scale up request, mark that we're in a spike
	if int32(scaleRequest.ReplicaCount) > currentScale {
		c.stateManager.StartSpike()
	}

	err = c.resourceManager.ScalePods(scaleRequest)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to scale: %v", err)})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{"message": "Scaling operation triggered successfully"})
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

func (c *MLController) HandleSpikeEnd(ctx *gin.Context) {
	c.stateManager.EndSpike()

	// Get all deployments in warm-pool namespace
	deployments, err := c.resourceManager.GetDeployments("warm-pool")
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to get warm pool deployments: %v", err)})
		return
	}

	// Scale down each deployment to 1 replica
	for _, deployment := range deployments.Items {
		scaleRequest := callback.ScaleRequest{
			DeploymentName: deployment.Name,
			Namespace:      "warm-pool",
			ReplicaCount:   1,
		}

		err := c.resourceManager.ScalePods(scaleRequest)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to scale down deployment %s: %v", deployment.Name, err)})
			return
		}
	}

	err = c.resourceManager.UpdateTrafficSplit(80, 20)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to update traffic split: %v", err)})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"message":        "Spike event ended successfully",
		"current_state":  "normal",
		"spike_ended_at": time.Now(),
	})
}

func (c *MLController) HandleGetSpikeState(ctx *gin.Context) {
	spikeState := c.stateManager.GetSpikeState()

	ctx.JSON(http.StatusOK, gin.H{
		"is_in_spike":         spikeState.IsInSpike,
		"spike_start_time":    spikeState.SpikeStartTime,
		"last_spike_end_time": spikeState.LastSpikeEndTime,
		"spike_count":         spikeState.SpikeRequestCount,
	})
}

func (c *MLController) RegisterRoutes(router *gin.Engine) {
	mlCallbackGroup := router.Group("/ml-callback")
	{
		mlCallbackGroup.POST("/scale", c.HandleScale)
		mlCallbackGroup.POST("/update-traffic-split", c.HandleUpdateTrafficSplit)
		mlCallbackGroup.POST("/spike-end", c.HandleSpikeEnd)
		mlCallbackGroup.GET("/spike-state", c.HandleGetSpikeState)
	}
}
