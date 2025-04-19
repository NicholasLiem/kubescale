package controller

import (
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
)

type MetadataController struct {
	startTime time.Time
	version   string
}

func NewMetadataController(version string) *MetadataController {
	return &MetadataController{
		startTime: time.Now(),
		version:   version,
	}
}

func (c *MetadataController) HandleHealthCheck(ctx *gin.Context) {
	ctx.String(http.StatusOK, "OK")
}

func (c *MetadataController) HandleGetVersion(ctx *gin.Context) {
	ctx.JSON(http.StatusOK, gin.H{
		"version": c.version,
	})
}

func (c *MetadataController) HandleGetStatus(ctx *gin.Context) {
	hostname, _ := os.Hostname()

	ctx.JSON(http.StatusOK, gin.H{
		"status":      "running",
		"uptime":      time.Since(c.startTime).String(),
		"hostname":    hostname,
		"environment": os.Getenv("APP_ENV"),
	})
}

func (c *MetadataController) RegisterRoutes(router *gin.Engine) {
	router.GET("/healthz", c.HandleHealthCheck)

	metadataGroup := router.Group("/metadata")
	{
		metadataGroup.GET("/version", c.HandleGetVersion)
		metadataGroup.GET("/status", c.HandleGetStatus)
	}
}
