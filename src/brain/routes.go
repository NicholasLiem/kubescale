package main

import (
	"fmt"
	"net/http"

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
		// TODO: Implement real scaling logic
        mlCallbackGroup.POST("/scale-up", func(c *gin.Context) {
            err := resourceManager.ScalePods(10)
            if err != nil {
                c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to scale up: %v", err)})
                return
            }
            c.JSON(http.StatusOK, gin.H{"message": "Scale-up triggered successfully"})
        })

		// TODO: Implement real scaling logic
        mlCallbackGroup.POST("/scale-down", func(c *gin.Context) {
            err := resourceManager.ScalePods(1)
            if err != nil {
                c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to scale down: %v", err)})
                return
            }
            c.JSON(http.StatusOK, gin.H{"message": "Scale-down triggered successfully"})
        })

		// TODO: Implement real preparation logic
        mlCallbackGroup.POST("/prepare", func(c *gin.Context) {
            c.JSON(http.StatusOK, gin.H{"message": "Preparation triggered successfully"})
        })
    }
}