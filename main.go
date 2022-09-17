package main

import (
	"bufio"
	"embed"
	"flag"
	"log"
	"net/http"
	"os"
	"strings"
)

const (
	pathToEnvFile = ".env"
)

// server represents the entire ctts service, and holds all dependencies.
type server struct {
	frontend   embed.FS
	port       string
	projectId  string
	locationId string
	queueId    string
	router     http.ServeMux
	db         interface{}
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

//go:embed frontend/dist
var frontend embed.FS

func newServer() *server {
	s := &server{frontend, os.Getenv("PORT"), os.Getenv("GCP_PROJECT_ID"), os.Getenv("GCP_LOCATION_ID"), os.Getenv("GCP_QUEUE_ID"), *http.NewServeMux(), nil}
	s.routes()
	return s
}

// loadEnv sets environment variables defined in an `.env` file.
func loadEnv(pathToEnvFile string) (err error) {
	file, err := os.Open(pathToEnvFile)
	if err != nil {
		return err
	}
	defer file.Close()
	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	for _, l := range lines {
		s := strings.Split(l, "=")
		if len(s) != 2 {
			log.Printf("!failed to parse env var from line: %s", l)
			continue
		}
		log.Printf("setting env var %s", s[0])
		os.Setenv(s[0], s[1])
	}
	return nil
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
