package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"time"

	cloudtasks "cloud.google.com/go/cloudtasks/apiv2"
	"github.com/evanw/esbuild/pkg/api"
	"github.com/nonnontrivial/ctts/internal/records"
	"github.com/nonnontrivial/ctts/internal/rowgen"
	taskspb "google.golang.org/genproto/googleapis/cloud/tasks/v2"
)

var errFailedToBuildClient = errors.New("failed to build client")

const (
	// endpoint of task runner for appending to csv records.
	sitesRecordsAppendPath = "/edi/records/insert"
)

// TODO: handle spa (404-based handling)
func setupClientHandler(s *server) (http.Handler, error) {
	frontendBuildResult := api.Build(api.BuildOptions{
		EntryPoints: []string{"./frontend/src/index.tsx"},
		Outfile:     "./frontend/dist/build/out.js",
		Write:       true,
	})
	if len(frontendBuildResult.Errors) > 0 {
		return nil, errFailedToBuildClient
	}
	fsys := fs.FS(s.frontend)
	subtree, err := fs.Sub(fsys, "frontend/dist")
	if err != nil {
		return nil, err
	}
	return http.FileServer(http.FS(subtree)), nil
}

func (s *server) routes() {
	s.router.HandleFunc("/api/reads/new", s.authOnly(s.handleNewRead()))
	s.router.HandleFunc(sitesRecordsAppendPath, s.handleRecordsInsert())

	h, err := setupClientHandler(s)
	if err != nil {
		log.Fatalln("got errors during client build")
	}
	s.router.Handle("/", h)
}

type SQMRead struct {
	// sky brightness (presumably from a device) measured in mpsas
	// (http://www.unihedron.com/projects/darksky/faq.php#:~:text=The%20term%20magnitudes%20per%20square,square%20arcsecond%20of%20the%20sky.)
	Brightness        float32   `json:"brightness"`
	Lat               string    `json:"lat"`
	Lng               string    `json:"lng"`
	TimeOfMeasurement time.Time `json:"timeOfMeasurement"`
}

// generateRow uses the sqm read data to derive a csv record suitable for appending
// to the list of records.
func (r *SQMRead) generateRow() ([]string, error) {
	g := rowgen.NewGenerator(r.Lat, r.Lng, r.TimeOfMeasurement)
	var independentVars []string
	if err := g.Backfill(&independentVars); err != nil {
		return nil, err
	}
	brightness := strconv.FormatFloat(float64(r.Brightness), 'E', -1, 64)
	row := []string{brightness}
	row = append(row, independentVars...)
	return row, nil
}

// authOnly restricts use of the given http handler to authenticated requests.
func (s *server) authOnly(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if false {
			http.NotFound(w, r)
			return
		}
		h(w, r)
	}
}

// handleNewRead submits a valid SQM read to become a row in the sky quality model
// by triggering the model update task to be run.
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
						RelativeUri: sitesRecordsAppendPath,
					},
				},
			},
		}
		body, err := ioutil.ReadAll(r.Body)
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

// handleSitesRecordsAppend is a [task handler](https://cloud.google.com/tasks/docs/creating-appengine-handlers).
// When a user submits a SQM read, this function appends the csv with a new row.
func (s *server) handleRecordsInsert() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Println("running records insert task handler")
		taskName := r.Header.Get("X-Appengine-Taskname")
		if taskName == "" {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		decoder := json.NewDecoder(r.Body)
		var sqmRead SQMRead
		if err := decoder.Decode(&sqmRead); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		ctx := context.Background()
		rs, err := records.NewRecords(ctx, s.datasetId, s.tableId, s.projectId)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rs.Client.Close()
		row, err := sqmRead.generateRow()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := rs.Append(row); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		queuename := r.Header.Get("X-Appengine-Queuename")
		log.Printf("finished sites update @ %s", queuename)
	}
}
