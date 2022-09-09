package main

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"

	"github.com/nonnontrivial/ctts/internal/html"
)

func (s *server) routes() {
	// FIXME: needs to change once using cron
	s.router.HandleFunc("/api/site", s.handleSite())

	s.router.HandleFunc("/site", s.handleSitePage())
}

var (
	errMissingCoords = errors.New("missing lat or lng in query params")
)

type coords struct {
	lat, lng string
}

func getCoords(r *http.Request) (*coords, error) {
	q := r.URL.Query()
	lat := q.Get("lat")
	lng := q.Get("lng")
	if lat == "" || lng == "" {
		return nil, errMissingCoords
	}
	return &coords{lat, lng}, nil
}

func (s *server) handleSite() http.HandlerFunc {
	type siteResponse struct {
		BortleClass int     `json:"bortleClass"`
		Id          string  `json:"id"`
		Mpsas       float32 `json:"mpsas"`
	}
	return func(w http.ResponseWriter, r *http.Request) {
		coords, err := getCoords(r)
		if err != nil {
			http.Error(w, errMissingCoords.Error(), http.StatusBadRequest)
			return
		}
		switch r.Method {
		case http.MethodGet:
			site := newSite(now, coords.lat, coords.lng)
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

func (s *server) handleSitePage() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		coords, err := getCoords(r)
		if err != nil {
			http.Error(w, errMissingCoords.Error(), http.StatusBadRequest)
			return
		}
		site := newSite(now, coords.lat, coords.lng)
		if err := site.fitToModel(); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			log.Printf("!got error: %s", err.Error())
			return
		}
		params := html.SiteParams{Mpsas: site.getMpsas(), Lat: coords.lat, Lng: coords.lng}
		html.Site(w, params)
	}
}
