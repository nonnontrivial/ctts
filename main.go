package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"time"
)

type server struct {
	router *http.ServeMux
	port   string
}

func newServer() *server {
	router := http.NewServeMux()
	s := &server{router, os.Getenv("PORT")}
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
