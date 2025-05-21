package main

import (
	"github.com/NicholasLiem/brain-controller/controller"
	"github.com/NicholasLiem/brain-controller/resource_manager"
	"github.com/NicholasLiem/brain-controller/state_manager"
	"github.com/gin-gonic/gin"
	"k8s.io/client-go/kubernetes"
)

const VERSION = "1.0.0"

func RegisterRoutes(
	router *gin.Engine,
	resourceManager *resource_manager.ResourceManager,
	stateManager *state_manager.StateManager,
	kubeClient *kubernetes.Clientset,
) {
	mlController := controller.NewMLController(resourceManager, stateManager)
	debugController := controller.NewDebugController(resourceManager, kubeClient)
	metadataController := controller.NewMetadataController(VERSION)

	metadataController.RegisterRoutes(router)
	mlController.RegisterRoutes(router)
	debugController.RegisterRoutes(router)
}
