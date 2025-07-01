package controller

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/NicholasLiem/brain-controller/resource_manager"
	"github.com/NicholasLiem/brain-controller/services"
	"github.com/gin-gonic/gin"
	"k8s.io/client-go/kubernetes"
)

type DebugController struct {
	resourceManager   *resource_manager.ResourceManager
	kubeClient        *kubernetes.Clientset
	prometheusService *services.PrometheusService
}

func NewDebugController(rm *resource_manager.ResourceManager, kubeClient *kubernetes.Clientset, prometheusService *services.PrometheusService) *DebugController {
	return &DebugController{
		resourceManager:   rm,
		kubeClient:        kubeClient,
		prometheusService: prometheusService,
	}
}

func (c *DebugController) HandlePrometheusQuery(ctx *gin.Context) {
	query := ctx.Query("query")
	if query == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "Missing required parameter: query",
			"example": "/debug/prometheus?query=" +
				`sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"s[0-2].*|gw-nginx.*"}[1m])) by (pod)` +
				"&start_time=1640995200&end_time=1640998800&step=1m",
		})
		return
	}

	if c.prometheusService == nil {
		ctx.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "Prometheus service not available",
		})
		return
	}

	step := ctx.Query("step")

	// If step is provided, use range query
	if step != "" {
		var startTime, endTime time.Time

		if startTimeStr := ctx.Query("start_time"); startTimeStr != "" {
			if timestamp, err := strconv.ParseInt(startTimeStr, 10, 64); err == nil {
				startTime = time.Unix(timestamp, 0)
			} else {
				ctx.JSON(http.StatusBadRequest, gin.H{
					"error": "Invalid start_time format. Use Unix timestamp.",
				})
				return
			}
		} else {
			startTime = time.Now().Add(-1 * time.Hour)
		}

		if endTimeStr := ctx.Query("end_time"); endTimeStr != "" {
			if timestamp, err := strconv.ParseInt(endTimeStr, 10, 64); err == nil {
				endTime = time.Unix(timestamp, 0)
			} else {
				ctx.JSON(http.StatusBadRequest, gin.H{
					"error": "Invalid end_time format. Use Unix timestamp.",
				})
				return
			}
		} else {
			endTime = time.Now()
		}

		result, err := c.prometheusService.QueryRange(query, startTime, endTime, step)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to execute Prometheus range query",
				"details": err.Error(),
			})
			return
		}

		ctx.JSON(http.StatusOK, gin.H{
			"query_type": "range",
			"query":      query,
			"start_time": startTime.Format(time.RFC3339),
			"end_time":   endTime.Format(time.RFC3339),
			"step":       step,
			"result":     result,
		})
	} else {
		// Use instant query
		result, err := c.prometheusService.Query(query)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to execute Prometheus query",
				"details": err.Error(),
			})
			return
		}

		ctx.JSON(http.StatusOK, gin.H{
			"query_type": "instant",
			"query":      query,
			"timestamp":  time.Now().Format(time.RFC3339),
			"result":     result,
		})
	}
}

func (c *DebugController) HandleCheckRecovery(ctx *gin.Context) {
    namespacesParam := ctx.DefaultQuery("namespaces", "default,warm-pool")
    thresholdParam := ctx.DefaultQuery("threshold", "30.0")
    
    namespaces := strings.Split(namespacesParam, ",")
    threshold, err := strconv.ParseFloat(thresholdParam, 64)
    if err != nil {
        threshold = 30.0
    }

    if c.resourceManager == nil {
        ctx.JSON(http.StatusServiceUnavailable, gin.H{"error": "Resource manager not available"})
        return
    }

    nsFilter := strings.Join(namespaces, "|")
    if nsFilter == "" {
        nsFilter = "default"
    }
    query := fmt.Sprintf(`avg(rate(container_cpu_usage_seconds_total{namespace=~"%s", container!="POD", container!=""}[1m])) * 100`, nsFilter)

    result, err := c.prometheusService.Query(query)
    if err != nil {
        ctx.JSON(http.StatusInternalServerError, gin.H{
            "error": "Failed to query Prometheus",
            "details": err.Error(),
            "query": query,
        })
        return
    }

    shouldRecover, avgCPU, rmErr := c.resourceManager.CheckCPUForRecovery(namespaces, threshold)

    ctx.JSON(http.StatusOK, gin.H{
        "namespaces": namespaces,
        "threshold": threshold,
        "query_used": query,
        "raw_prometheus_result": result,
        "average_cpu_percent": avgCPU,
        "should_trigger_recovery": shouldRecover,
        "resource_manager_error": func() interface{} {
            if rmErr != nil { return rmErr.Error() }
            return nil
        }(),
        "status": func() string {
            if shouldRecover { return "RECOVERY_NEEDED" }
            return "NORMAL"
        }(),
    })
}

func (c *DebugController) RegisterRoutes(router *gin.Engine) {
	debugGroup := router.Group("/debug")
	{
		debugGroup.GET("/routes", func(ctx *gin.Context) {
			ctx.Set("engine", router)
		})
		debugGroup.GET("/prometheus", c.HandlePrometheusQuery)
		debugGroup.GET("/check-recovery", c.HandleCheckRecovery)
	}
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}
