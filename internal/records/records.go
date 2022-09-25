package records

import (
	"context"

	"cloud.google.com/go/bigquery"
)

type Records struct {
	Client    *bigquery.Client
	datasetId string
	tableId   string
}

func NewRecords(ctx context.Context, datasetId, tableId, projectId string) (*Records, error) {
	c, err := bigquery.NewClient(ctx, projectId)
	if err != nil {
		return nil, err
	}
	return &Records{c, datasetId, tableId}, nil
}

// Append loads a new row into the bigquery table.
//
// This is typically called as part of a GCP task handler for SQM reads.
func (r *Records) Append(row []string) error {
	return nil
}
