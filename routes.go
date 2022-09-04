package main

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/nonnontrivial/ctts/internal/html"
)

func (s *server) routes() {
	s.router.HandleFunc("/api/site", s.handleSite())
	s.router.HandleFunc("/api/user", s.handleUser())

	s.router.HandleFunc("/site", s.handleSitePage())
}

func (s *server) handleSitePage() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		params := html.SiteParams{}
		html.Site(w, params)
	}
}

func (s *server) handleUser() http.HandlerFunc {
	type userResponse struct{}
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
		case http.MethodPatch:
		case http.MethodGet:
			return
		default:
			http.Error(w, "bad method", http.StatusMethodNotAllowed)
			return
		}
	}
}

func (s *server) handleSite() http.HandlerFunc {
	type siteResponse struct {
		BortleClass int    `json:"bortleClass"`
		Id          string `json:"id"`
		// magnitudes per square arcsecond
		Mpsas float32 `json:"mpsas"`
	}
	return func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		lat := q.Get("lat")
		lng := q.Get("lng")
		switch r.Method {
		case http.MethodGet:
			if lat == "" || lng == "" {
				http.Error(w, "missing lat or lng in query params", http.StatusBadRequest)
				return
			}
			site := newSite(now, lat, lng)
			if err := site.fitToModel(); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				log.Printf("!got error: %s", err.Error())
				return
			}
			sr := &siteResponse{
				Mpsas:       site.getMpsas(),
				BortleClass: site.getBortleClass(),
				Id:          site.getId(),
			}
			w.WriteHeader(http.StatusOK)
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(sr); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		default:
			http.Error(w, "bad method", http.StatusMethodNotAllowed)
			return
		}
	}
}
