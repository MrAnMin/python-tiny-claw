package main

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type PingResponse struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func pingHandler(w http.ResponseWriter, r *http.Request) {
	response := PingResponse{
		Code:    200,
		Message: "pong",
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

func main() {
	http.HandleFunc("/ping", pingHandler)
	
	port := "8080"
	fmt.Printf("HTTP ping 服务器启动在端口 %s...\n", port)
	fmt.Printf("访问 http://localhost:%s/ping 进行测试\n", port)
	
	err := http.ListenAndServe(":"+port, nil)
	if err != nil {
		fmt.Printf("服务器启动失败: %v\n", err)
	}
}