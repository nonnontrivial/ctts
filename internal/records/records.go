package records

import (
	"context"
	"log"
)

type Records struct {
}

func NewRecords(ctx context.Context) (*Records, error) {
	return &Records{}, nil
}

// Append appends the csv with a row.
func (r *Records) Append(row []string) error {
	log.Println(row)
	return nil
}
