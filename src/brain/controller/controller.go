package controller

import "github.com/gin-gonic/gin"

type Controller interface {
    RegisterRoutes(router *gin.Engine)
}