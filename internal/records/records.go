package records

import (
	"context"
	"os"

	"cloud.google.com/go/storage"
	"google.golang.org/appengine/file"
)

var cloudStorageCSVFilename = os.Getenv("CLOUD_STORAGE_CSV_FILENAME")

const (
	// path to the csv template used in the case of no existing csv file in
	// cloud storage
	pathToCSVTemplate = "sitesTemplate.csv"
)

type Records struct {
	client     *storage.Client
	bucketName string
	bucket     *storage.BucketHandle
	ctx        context.Context
}

func NewRecords(ctx context.Context) (*Records, error) {
	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, err
	}
	bucket, err := file.DefaultBucketName(ctx)
	if err != nil {
		return nil, err
	}
	return &Records{client, "", client.Bucket(bucket), ctx}, nil
}

// Append appends the csv file in cloud storage with a row.
func (r *Records) Append(row []string) error {
	// TODO:
	return nil
}

// createFile creates a csv file in cloud storage.
func (r *Records) createFile() error {
	w := r.bucket.Object(cloudStorageCSVFilename).NewWriter(r.ctx)
	w.ContentType = "text/csv"
	// TODO: get csv data from template
	data := []byte{}
	if _, err := w.Write(data); err != nil {
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}
	return nil
}
