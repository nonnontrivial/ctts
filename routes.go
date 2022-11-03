package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func (s *server) handleRead(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]bool{"42": true})
}

func (s *server) handleView(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]bool{"42": false})
}

func (s *server) routes() {
	api := s.router.PathPrefix("/api").Subrouter()
	api.HandleFunc("/read", s.handleRead).Methods(http.MethodPost, http.MethodGet)
	api.HandleFunc("/view", s.handleView)

	if err := buildClient(); err != nil {
		log.Fatalln(err)
	}
	spa := spaHandler{clientFiles, "client/dist", "index.html"}
	s.router.PathPrefix("/").Handler(spa)
}
