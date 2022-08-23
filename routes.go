package main

import (
	"encoding/json"
	"io/fs"
	"log"
	"net/http"

	"github.com/evanw/esbuild/pkg/api"
)

func (s *server) routes() {
	s.router.Handle("/", s.handleRoot())
	s.router.HandleFunc("/api/site", s.authGate(s.handleSite()))
}

func (s *server) authGate(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// FIXME: use real condition for auth
		if !true {
			http.NotFound(w, r)
			return
		}
		h(w, r)
	}
}

func (s *server) handleRoot() http.Handler {
	result := api.Build(api.BuildOptions{
		EntryPoints: []string{"./client/src/root.tsx"},
		Outfile:     "./client/web/build/index.js",
		Sourcemap:   api.SourceMapLinked,
		Format:      api.FormatESModule,
		Bundle:      true,
		Write:       true,
	})
	if errs := result.Errors; len(errs) != 0 {
		log.Fatal(errs[len(errs)-1])
	}
	staticAssets, _ := fs.Sub(fs.FS(s.assets), "client/web")
	return http.FileServer(http.FS(staticAssets))
}

func (s *server) handleSite() http.HandlerFunc {
	type siteResponse struct {
		// magnitudes per square arcsecond
		Mpsas       float32 `json:"mpsas"`
		BortleClass int     `json:"bortleClass"`
		Id          string  `json:"id"`
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "bad method", http.StatusMethodNotAllowed)
			return
		}
		q := r.URL.Query()
		lat := q.Get("lat")
		lng := q.Get("lng")
		if lat == "" || lng == "" {
			http.Error(w, "missing lat or lng in query params", http.StatusBadRequest)
			return
		}
		site := newSite(lat, lng)
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
	}
}
