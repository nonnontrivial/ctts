package main

import (
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"

	esbuild "github.com/evanw/esbuild/pkg/api"
)

func buildClient() error {
	result := esbuild.Build(esbuild.BuildOptions{
		EntryPoints: []string{"./client/src/index.tsx"},
		Outfile:     "./client/dist/build/out.js",
		Write:       true,
		Bundle:      true,
	})
	if len(result.Errors) > 0 {
		return errors.New("failed to build client")
	}
	return nil
}

//go:embed client/dist
var clientFiles embed.FS

type spaHandler struct {
	staticFiles embed.FS
	staticPath  string
	indexPath   string
}

// provide SPA-friendly implementation of this method
func (h spaHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path, err := filepath.Abs(r.URL.Path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	path = filepath.Join(h.staticPath, path)
	_, err = h.staticFiles.Open(path)
	if os.IsNotExist(err) {
		indexHtml, err := h.staticFiles.ReadFile(filepath.Join(h.staticPath, h.indexPath))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		w.WriteHeader(http.StatusAccepted)
		w.Write(indexHtml)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	files, err := fs.Sub(h.staticFiles, h.staticPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	http.FileServer(http.FS(files)).ServeHTTP(w, r)
}

func (s *server) handleRead(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]bool{"42": true})
}

func (s *server) routes() {
	if err := buildClient(); err != nil {
		log.Fatalln(err)
	}
	api := s.router.PathPrefix("/api").Subrouter()
	api.HandleFunc("/read", s.handleRead).Methods(http.MethodPost, http.MethodGet)

	spa := spaHandler{clientFiles, "client/dist", "index.html"}
	s.router.PathPrefix("/").Handler(spa)
}
