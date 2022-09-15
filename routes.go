package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"time"

	cloudtasks "cloud.google.com/go/cloudtasks/apiv2"
	sites "github.com/nonnontrivial/ctts/internal/records"
	"github.com/nonnontrivial/ctts/internal/rowgen"
	taskspb "google.golang.org/genproto/googleapis/cloud/tasks/v2"
)

const (
	// endpoint of task runner for sites
	sitesUpdatePath = "/edi/sites/update"
)

type (
	SQMRead struct {
		// sky brightness (presumably from a device) measured in
		// [mpsas](http://www.unihedron.com/projects/darksky/faq.php#:~:text=The%20term%20magnitudes%20per%20square,square%20arcsecond%20of%20the%20sky.)
		Brightness        float32   `json:"brightness"`
		Lat               string    `json:"lat"`
		Lng               string    `json:"lng"`
		TimeOfMeasurement time.Time `json:"timeOfMeasurement"`
	}
)

// generateRow uses the sqm read data to derive a csv record suitable for
// appending to the record in cloud storage.
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

func (s *server) routes() {
	s.router.HandleFunc("/api/sites/submit", s.authOnly(s.handleSitesSubmit()))
	s.router.HandleFunc(sitesUpdatePath, s.handleSitesUpdate())
}

// TODO: implement
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

// handleSitesSubmit submits a valid SQM read to become a row in the sky quality
// model by triggering the model update task to be run.
func (s *server) handleSitesSubmit() http.HandlerFunc {
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
		queuePath := fmt.Sprintf("projects/%s/locations/%s/queues/%s", projectId, locationId, queueId)
		// build task payload
		req := &taskspb.CreateTaskRequest{
			Parent: queuePath,
			Task: &taskspb.Task{
				MessageType: &taskspb.Task_AppEngineHttpRequest{
					AppEngineHttpRequest: &taskspb.AppEngineHttpRequest{
						HttpMethod:  taskspb.HttpMethod_POST,
						RelativeUri: sitesUpdatePath,
					},
				},
			},
		}
		body, err := ioutil.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// TODO: validate body
		req.Task.GetHttpRequest().Body = body
		_, err = client.CreateTask(ctx, req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
}

// handleSitesUpdate is a [task handler](https://cloud.google.com/tasks/docs/creating-appengine-handlers).
//
// When a user submites a SQM read, this function appends the csv in cloud storage
// with the generated row.
func (s *server) handleSitesUpdate() http.HandlerFunc {
	type request struct{}
	type response struct{}
	return func(w http.ResponseWriter, r *http.Request) {
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
		records, err := sites.NewRecords(ctx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		row, err := sqmRead.generateRow()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// put the sqm read into cloud storage
		if err := records.Append(row); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		queuename := r.Header.Get("X-Appengine-Queuename")
		log.Printf("finished sites update @ %s", queuename)
	}
}
