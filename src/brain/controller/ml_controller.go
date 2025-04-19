package controller

import (
    "encoding/json"
    "fmt"
    "net/http"

    callback "github.com/NicholasLiem/brain-controller/dto"
    "github.com/NicholasLiem/brain-controller/resource_manager"
    "github.com/NicholasLiem/brain-controller/warm_pool_manager"
    "github.com/gin-gonic/gin"
)

type MLController struct {
    resourceManager *resource_manager.ResourceManager
    warmPoolManager *warm_pool_manager.WarmPoolManager
}

func NewMLController(rm *resource_manager.ResourceManager, wpm *warm_pool_manager.WarmPoolManager) *MLController {
    return &MLController{
        resourceManager: rm,
        warmPoolManager: wpm,
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
    
    err = c.resourceManager.ScalePods(scaleRequest)
    if err != nil {
        ctx.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to scale up: %v", err)})
        return
    }
    
    ctx.JSON(http.StatusOK, gin.H{"message": "Scale-up triggered successfully"})
}

func (c *MLController) HandlePrepare(ctx *gin.Context) {
    // TODO: Implement preparation logic
    ctx.JSON(http.StatusOK, gin.H{"message": "Preparation triggered successfully"})
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
    
    if request.DefaultPercent + request.WarmPoolPercent != 100 {
        ctx.JSON(http.StatusBadRequest, gin.H{"error": "Percentages must sum to 100"})
        return
    }
    
    err := c.resourceManager.UpdateTrafficSplit(request.DefaultPercent, request.WarmPoolPercent)
    if err != nil {
        ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    ctx.JSON(http.StatusOK, gin.H{
        "message": "Traffic split updated successfully",
        "default_percent": request.DefaultPercent,
        "warm_pool_percent": request.WarmPoolPercent,
    })
}

func (c *MLController) RegisterRoutes(router *gin.Engine) {
    mlCallbackGroup := router.Group("/ml-callback")
    {
        mlCallbackGroup.POST("/scale", c.HandleScale)
        mlCallbackGroup.POST("/prepare", c.HandlePrepare)
        mlCallbackGroup.POST("/update-traffic-split", c.HandleUpdateTrafficSplit)
    }
}