package model

import (
	"encoding/csv"
	"errors"
	"log"
	"math/rand"
	"os"
	"strconv"
	"sync"

	"github.com/sajari/regression"
)

var (
	errNoVars = errors.New("no vars supplied")
)

const (
	yValueKey = "mpsas"
)

type (
	model struct {
		trainingPath, testingPath, formula string
		regressor                          regression.Regression
	}
	lrModeller interface {
		// write training and testing datasets
		prepare(csvRecords [][]string) error
		// train the model
		Train() error
		// get the https://en.m.wikipedia.org/wiki/Coefficient_of_determination for the model
		Test() float64
		// predict the next y for the x vector
		Predict(vars []float64) (float64, error)
	}
)

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

// prepare splits data into training and testing csv files
// see https://medium.com/devthoughts/linear-regression-with-go-ff1701455bcd
func (m *model) prepare(csvRecords [][]string) error {
	header := csvRecords[0]
	shuffled := make([][]string, len(csvRecords)-1)
	ints := rand.Perm(len(csvRecords) - 1)
	for i, v := range ints {
		shuffled[v] = csvRecords[i+1]
	}
	trainingIdx := ((len(shuffled)) * 4) / 5
	trainingSet := shuffled[1 : trainingIdx+1]
	testingSet := shuffled[trainingIdx+1:]
	sets := map[string][][]string{m.trainingPath: trainingSet, m.testingPath: testingSet}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for path := range sets {
			log.Printf("removing %s", path)
			err := os.Remove(path)
			if err != nil {
				panic(err)
			}
		}
	}()
	wg.Wait()
	for path, set := range sets {
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		defer f.Close()
		w := csv.NewWriter(f)
		log.Printf("writing %s", path)
		if err := w.Write(header); err != nil {
			return err
		}
		if err := w.WriteAll(set); err != nil {
			return err
		}
		w.Flush()
	}
	return nil
}

func (m *model) Train() error {
	csvRecords, err := readCSV(m.trainingPath)
	if err != nil {
		return err
	}
	m.regressor.SetObserved(yValueKey)
	for i, h := range csvRecords[0][1:] {
		log.Printf("setting independent var %s", h)
		m.regressor.SetVar(i, h)
	}
	for _, record := range csvRecords[1:] {
		mpsas, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return err
		}
		var vars = []float64{}
		for _, v := range record[1:] {
			s, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return err
			}
			vars = append(vars, s)
		}
		log.Printf("training %f", vars)
		m.regressor.Train(regression.DataPoint(mpsas, vars))
	}
	log.Println("running regression...")
	if err = m.regressor.Run(); err != nil {
		return err
	}
	return nil
}

func (m *model) Test() float64 {
	return m.regressor.R2
}

func (m *model) Predict(vars []float64) (float64, error) {
	if len(vars) == 0 {
		return 0, errNoVars
	}
	// FIXME: panics
	// return m.Predict(vars)
	return 0, nil
}

func NewModel(trainingPath, testingPath, csvPath string) (lrModeller, error) {
	m := &model{trainingPath, testingPath, "", regression.Regression{}}
	csvRecords, err := readCSV(csvPath)
	if err != nil {
		return nil, err
	}
	if err := m.prepare(csvRecords); err != nil {
		return nil, err
	}
	return m, nil
}
