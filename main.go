package main

import (
	"embed"
	"flag"
	"log"
	"net/http"
	"os"
)

const (
	pathToEnvFile = ".env"
)

// server represents the entire ctts service, and holds all dependencies
type server struct {
	clientFiles embed.FS
	port        string
	datasetId   string
	tableId     string
	projectId   string
	locationId  string
	queueId     string
	router      http.ServeMux
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

//go:embed client/dist
var client embed.FS

func newServer() *server {
	s := &server{client, os.Getenv("PORT"), os.Getenv("GCP_DATASET_ID"), os.Getenv("GCP_TABLEID"), os.Getenv("GCP_PROJECT_ID"), os.Getenv("GCP_LOCATION_ID"), os.Getenv("GCP_QUEUE_ID"), *http.NewServeMux()}
	s.routes()
	return s
}

func main() {
	dev := flag.Bool("dev", true, "determines if running in dev mode")
	flag.Parse()
	if *dev {
		if err := loadEnv(pathToEnvFile); err != nil {
			log.Fatalln(err)
		}
	}
	s := newServer()
	log.Printf("starting server on port %s...\n", s.port)
	if err := http.ListenAndServe(":"+s.port, &s.router); err != nil {
		log.Fatalln(err)
	}
}
