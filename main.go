package main

import (
	"embed"
	"log"
	"net/http"
	"net/smtp"
	"os"
)

var (
	//go:embed client
	clientFiles embed.FS
	port        = os.Getenv("PORT")
)

const defaultPort = "3303"

// server represents the entire ctts service, and holds all dependencies
type server struct {
	assets embed.FS
	mailer smtp.Client
	router http.ServeMux
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

func newServer() *server {
	s := &server{}
	s.assets = clientFiles
	s.mailer = smtp.Client{}
	s.routes()
	return s
}

func main() {
	if port == "" {
		port = defaultPort
		log.Printf("using default port: %s", defaultPort)
	}
	s := newServer()
	log.Println("starting server...")
	if err := http.ListenAndServe(":"+port, &s.router); err != nil {
		log.Fatal(err)
	}
}
