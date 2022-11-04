package main

import (
	"encoding/json"
	"log"
	"net/http"

	"google.golang.org/grpc"
)

func (s *server) handleRead(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]bool{"42": true})
}

func (s *server) handleView(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]bool{"42": false})
}

type gServer struct {
}

func (s *server) routes() {
	g := grpc.NewServer()
	log.Println(g)

	api := s.router.PathPrefix("/api").Subrouter()
	api.HandleFunc("/read", s.handleRead).Methods(http.MethodPost, http.MethodGet)
	api.HandleFunc("/view", s.handleView)

	if err := buildClient(); err != nil {
		log.Fatalln(err)
	}
	spa := spaHandler{clientFiles, "client/dist", "index.html"}
	s.router.PathPrefix("/").Handler(spa)
}
