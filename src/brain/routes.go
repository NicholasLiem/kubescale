package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	callback "github.com/NicholasLiem/brain-controller/dto"
	"github.com/NicholasLiem/brain-controller/resource_manager"
	"github.com/NicholasLiem/brain-controller/warm_pool_manager"
	"github.com/gin-gonic/gin"
)

func RegisterRoutes(router *gin.Engine, resourceManager *resource_manager.ResourceManager, warmPoolManager *warm_pool_manager.WarmPoolManager) {
    // Health check route
    router.GET("/healthz", func(c *gin.Context) {
        c.String(http.StatusOK, "OK")
    })

    // ML Callback group
    mlCallbackGroup := router.Group("/ml-callback")
    {
        mlCallbackGroup.POST("/scale", func(c *gin.Context) {
            body, err := c.GetRawData()
            if err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request body"})
                return
            }

            var scaleRequest callback.ScaleRequest
            if err := json.Unmarshal(body, &scaleRequest); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON format"})
                return
            }
            err = resourceManager.ScalePods(scaleRequest)
            if err != nil {
                c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to scale up: %v", err)})
                return
            }
            c.JSON(http.StatusOK, gin.H{"message": "Scale-up triggered successfully"})
        })

		// TODO: Implement real preparation logic
        mlCallbackGroup.POST("/prepare", func(c *gin.Context) {
            c.JSON(http.StatusOK, gin.H{"message": "Preparation triggered successfully"})
        })
    }
}