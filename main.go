package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
)

// server represents the entire ctts service, and holds all dependencies
type server struct {
	router     *mux.Router
	port       string
	datasetId  string
	tableId    string
	projectId  string
	locationId string
	queueId    string
}

func newServer() *server {
	router := mux.NewRouter()
	s := &server{router, os.Getenv("PORT"), os.Getenv("GCP_DATASET_ID"), os.Getenv("GCP_TABLEID"), os.Getenv("GCP_PROJECT_ID"), os.Getenv("GCP_LOCATION_ID"), os.Getenv("GCP_QUEUE_ID")}
	s.routes()
	return s
}

func main() {
	dev := flag.Bool("dev", true, "determines if running in dev mode")
	flag.Parse()
	if *dev {
		if err := loadEnv(".env"); err != nil {
			log.Fatalln(err)
		}
	}
	s := newServer()
	srv := &http.Server{
		Handler:      s.router,
		Addr:         ":" + s.port,
		WriteTimeout: 10 * time.Second,
		ReadTimeout:  10 * time.Second,
	}
	log.Printf("serving HTTP on port %s...\n", s.port)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalln(err)
	}
}
