package main

import (
	"log"
	"net/http"
	"os"
)

var (
	projectId  = os.Getenv("PROJECT_ID")
	locationId = os.Getenv("LOCATION_ID")
	queueId    = os.Getenv("QUEUE_ID")
)

const (
	defaultPort = "3303"
)

// server represents the entire ctts service, and holds all dependencies.
type server struct {
	router http.ServeMux
	db     interface{}
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

func newServer() *server {
	s := &server{}
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
