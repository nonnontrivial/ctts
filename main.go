package main

import (
	"embed"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
)

//go:embed client
var clientFiles embed.FS

// server represents the entire ctts service, and holds all dependencies
type server struct {
	assets embed.FS
	router http.ServeMux
	// mailer interface{}
	// db     interface{}
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
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		// TODO: any cleanup before exit(?)
		os.Exit(0)
	}()
	port := os.Getenv("PORT")
	defaultPort := "3303"
	if port == "" {
		port = defaultPort
		log.Printf("using default port: %s", defaultPort)
	}
	s := newServer()
	if err := http.ListenAndServe(":"+port, &s.router); err != nil {
		log.Fatal(err)
	}
}
