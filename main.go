package main

import (
	"log"
	"net/http"
	"os"
)

const (
	defaultPort = "3303"
)

// server represents the entire ctts service, and holds all dependencies.
type server struct {
	projectId  string
	locationId string
	queueId    string
	router     http.ServeMux
	db         interface{}
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

func newServer() *server {
	s := &server{os.Getenv("GCP_PROJECT_ID"), os.Getenv("GCP_LOCATION_ID"), os.Getenv("GCP_QUEUE_ID"), *http.NewServeMux(), nil}
	s.routes()
	return s
}

func main() {
	s := newServer()
	port := os.Getenv("PORT")
	if port == "" {
		log.Printf("using default port: %s", defaultPort)
		port = defaultPort
	}
	log.Println("starting server...")
	if err := http.ListenAndServe(":"+port, &s.router); err != nil {
		log.Fatal(err)
	}
}
