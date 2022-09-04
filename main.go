package main

import (
	"log"
	"net/http"
	"net/smtp"
	"os"
	"time"
)

const defaultPort = "3303"

// TODO: probably need to move after refactor for use with app engine cron
var now = time.Now()

// server represents the entire ctts service, and holds all dependencies
type server struct {
	mailer smtp.Client
	router http.ServeMux
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

func newServer() *server {
	s := &server{}
	s.mailer = smtp.Client{}
	s.routes()
	return s
}

func getPort() string {
	port := os.Getenv("PORT")
	if port == "" {
		log.Printf("using default port: %s", defaultPort)
		port = defaultPort
	}
	return port
}

func main() {
	s := newServer()
	log.Println("starting server...")
	if err := http.ListenAndServe(":"+getPort(), &s.router); err != nil {
		log.Fatal(err)
	}
}
