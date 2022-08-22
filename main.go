package main

import (
	"embed"
	"log"
	"net/http"
	"os"
)

//go:embed client
var clientFiles embed.FS

const defaultPort = "3303"

// server represents the entire ctts service, and holds all dependencies
type server struct {
	assets embed.FS
	router http.ServeMux
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

func newServer() *server {
	s := &server{}
	s.assets = clientFiles
	s.routes()
	return s
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = defaultPort
		log.Printf("using default port: %s", defaultPort)
	}
	s := newServer()
	if err := http.ListenAndServe(":"+port, &s.router); err != nil {
		log.Fatal(err)
	}
}
