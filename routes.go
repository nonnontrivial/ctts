package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"

	cloudtasks "cloud.google.com/go/cloudtasks/apiv2"
	esbuildAPI "github.com/evanw/esbuild/pkg/api"
	taskspb "google.golang.org/genproto/googleapis/cloud/tasks/v2"
)

// setupClientHandler builds the client
func setupClientHandler(s *server) (http.Handler, error) {
	result := esbuildAPI.Build(esbuildAPI.BuildOptions{
		EntryPoints: []string{"./client/src/index.tsx"},
		Outfile:     "./client/dist/build/out.js",
		Write:       true,
		Bundle:      true,
	})
	if len(result.Errors) > 0 {
		return nil, errors.New("failed to build client")
	}
	fsys := fs.FS(s.clientFiles)
	subtree, err := fs.Sub(fsys, "client/dist")
	if err != nil {
		return nil, err
	}
	return http.FileServer(http.FS(subtree)), nil
}

const apiPrefix = "/api"

var (
	recordsAppendPath = fmt.Sprintf("%s/records/append", apiPrefix)
	newReadPath       = fmt.Sprintf("%s/read", apiPrefix)
)

// routes registers the routes
func (s *server) routes() {
	// TODO: use gRPC
	s.router.HandleFunc(newReadPath, s.handleNewRead())
	s.router.HandleFunc(recordsAppendPath, s.handleRecordsAppend())

	h, err := setupClientHandler(s)
	if err != nil {
		log.Fatalln("got errors during client build")
	}
	log.Println("built client")
	// TODO: handle spa (404-based handling)
	s.router.Handle("/", h)
}

func (s *server) handleNewRead() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "only accepts POST", http.StatusMethodNotAllowed)
			return
		}
		ctx := context.Background()
		client, err := cloudtasks.NewClient(ctx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer client.Close()
		queuePath := fmt.Sprintf("projects/%s/locations/%s/queues/%s", s.projectId, s.locationId, s.queueId)
		req := &taskspb.CreateTaskRequest{
			Parent: queuePath,
			Task: &taskspb.Task{
				MessageType: &taskspb.Task_AppEngineHttpRequest{
					AppEngineHttpRequest: &taskspb.AppEngineHttpRequest{
						HttpMethod:  taskspb.HttpMethod_POST,
						RelativeUri: recordsAppendPath,
					},
				},
			},
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		req.Task.GetHttpRequest().Body = body
		_, err = client.CreateTask(ctx, req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
}

func (s *server) handleRecordsAppend() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Println("running records insert task handler")
		taskName := r.Header.Get("X-Appengine-Taskname")
		if taskName == "" {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		decoder := json.NewDecoder(r.Body)
		var sqmRead Read
		if err := decoder.Decode(&sqmRead); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// TODO: unmarshal into Row struct
		row, err := sqmRead.FindFeatures()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := InsertRow(row, s.projectId, s.datasetId, s.tableId); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		queuename := r.Header.Get("X-Appengine-Queuename")
		log.Printf("finished sites update @ %s", queuename)
	}
}
