package controller

import (
    "net/http"
    "runtime"

    "github.com/NicholasLiem/brain-controller/resource_manager"
    "github.com/gin-gonic/gin"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
)

type DebugController struct {
    resourceManager *resource_manager.ResourceManager
    kubeClient      *kubernetes.Clientset
}

func NewDebugController(rm *resource_manager.ResourceManager, kubeClient *kubernetes.Clientset) *DebugController {
    return &DebugController{
        resourceManager: rm,
        kubeClient:      kubeClient,
    }
}

func (c *DebugController) HandleGetResourceUsage(ctx *gin.Context) {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)

    ctx.JSON(http.StatusOK, gin.H{
        "memory": gin.H{
            "alloc":      m.Alloc,
            "total_alloc": m.TotalAlloc,
            "sys":        m.Sys,
            "num_gc":     m.NumGC,
        },
        "goroutines": runtime.NumGoroutine(),
    })
}

func (c *DebugController) HandleListPods(ctx *gin.Context) {
    namespace := ctx.DefaultQuery("namespace", "default")
    
    pods, err := c.kubeClient.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
    if err != nil {
        ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    var podNames []string
    for _, pod := range pods.Items {
        podNames = append(podNames, pod.Name)
    }
    
    ctx.JSON(http.StatusOK, gin.H{
        "namespace": namespace,
        "pod_count": len(podNames),
        "pods":      podNames,
    })
}

func (c *DebugController) HandlePrintRoutes(ctx *gin.Context) {
    engine := ctx.MustGet("engine").(*gin.Engine)
    routes := engine.Routes()
    
    var routeInfo []gin.H
    for _, route := range routes {
        routeInfo = append(routeInfo, gin.H{
            "method": route.Method,
            "path":   route.Path,
            "handler": route.Handler,
        })
    }
    
    ctx.JSON(http.StatusOK, gin.H{
        "routes": routeInfo,
    })
}
func (c *DebugController) RegisterRoutes(router *gin.Engine) {
    debugGroup := router.Group("/debug")
    {
        debugGroup.GET("/memory", c.HandleGetResourceUsage)
        debugGroup.GET("/pods", c.HandleListPods)
        debugGroup.GET("/routes", func(ctx *gin.Context) {
            ctx.Set("engine", router)
            c.HandlePrintRoutes(ctx)
        })
    }
}