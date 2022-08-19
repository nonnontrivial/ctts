package lrmodel

import (
	"encoding/csv"
	"os"
)

// readCSV reads the csv file at path and outputs the rows as string slices
func readCSV(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		return nil, err
	}
	return records, nil
}
