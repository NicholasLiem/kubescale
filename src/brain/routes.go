package main

import (
	"github.com/NicholasLiem/brain-controller/controller"
	"github.com/NicholasLiem/brain-controller/resource_manager"
	"github.com/NicholasLiem/brain-controller/warm_pool_manager"
	"github.com/gin-gonic/gin"
	"k8s.io/client-go/kubernetes"
)

// Update the version number when you make significant changes
const VERSION = "1.0.0"

func RegisterRoutes(
	router *gin.Engine,
	resourceManager *resource_manager.ResourceManager,
	warmPoolManager *warm_pool_manager.WarmPoolManager,
	kubeClient *kubernetes.Clientset,
) {
	mlController := controller.NewMLController(resourceManager, warmPoolManager)
	debugController := controller.NewDebugController(resourceManager, kubeClient)
	metadataController := controller.NewMetadataController(VERSION)

	metadataController.RegisterRoutes(router)
	mlController.RegisterRoutes(router)
	debugController.RegisterRoutes(router)
}
