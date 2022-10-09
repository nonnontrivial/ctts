//go:generate protoc -I/usr/local/include -I. --go_out=. row.proto

package main

import (
	"context"
	"fmt"

	storage "cloud.google.com/go/bigquery/storage/apiv1"
	storagepb "google.golang.org/genproto/googleapis/cloud/bigquery/storage/v1"
)

func InsertRow(row *Row, projectId, datasetId, tableId string) error {
	ctx := context.Background()
	c, err := storage.NewBigQueryWriteClient(ctx)
	if err != nil {
		return err
	}
	defer c.Close()
	// TODO:
	_, err = c.CreateWriteStream(ctx, &storagepb.CreateWriteStreamRequest{
		Parent: fmt.Sprintf("projects/%s/datasets/%s/tables/%s", projectId, datasetId, tableId),
		WriteStream: &storagepb.WriteStream{
			Type: storagepb.WriteStream_COMMITTED,
		},
	})
	if err != nil {
		return err
	}
	_, err = c.AppendRows(ctx)
	if err != nil {
		return err
	}
	// var opts proto.MarshalOptions
	// var data [][]byte
	// buf, err := opts.Marshal(row)
	// if err != nil {
	// 	return err
	// }
	// data = append(data, buf)
	return nil
}
