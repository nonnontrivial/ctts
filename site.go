package main

import (
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/nonnontrivial/ctts/pkg/lrmodel"
)

var (
	orderedColumns = []columnName{"elevation", "cloudCover", "temperature", "windSpeed", "airMoisture"}
	errBadR2       = errors.New("r2 below threshold")
)

const (
	// closer to 1 is better
	r2Limit      = 0.5
	csvPath      = "./data/sites.csv"
	trainingPath = "./data/training.csv"
	testingPath  = "./data/testing.csv"
)

type (
	columnName string
	vars       map[columnName]float64
	site       struct {
		id       string
		lat, lng string
		time     time.Time
		// magnitudes per square arcsecond
		mpsas float32
		vars
	}
	independentVarsDeriver interface {
		derive() error
	}
	siteModeller interface {
		getId() string
		getMpsas() float32
		getBortleClass() int
		fitToModel() error
		independentVarsDeriver
	}
)

func (s *site) getId() string     { return s.id }
func (s *site) getMpsas() float32 { return s.mpsas }
func (s *site) getBortleClass() int {
	return int(s.mpsas)
}

func (s *site) fitToModel() error {
	m, err := lrmodel.NewModel(trainingPath, testingPath, csvPath)
	if err != nil {
		return err
	}
	if err = m.Train(); err != nil {
		return err
	}
	if err = s.derive(); err != nil {
		return err
	}
	xs := []float64{}
	for _, c := range orderedColumns {
		xs = append(xs, s.vars[c])
	}
	y, err := m.Predict(xs)
	if err != nil {
		return err
	}
	log.Printf("predicted y value: %f", y)
	if r2 := m.Test(); r2 < r2Limit {
		return errBadR2
	}
	s.mpsas = float32(y)
	return nil
}

func newSite(lat, lng string) siteModeller {
	t := time.Now()
	id := fmt.Sprintf("%s,%s@%d", lat, lng, t.Nanosecond())
	vs := make(vars)
	s := &site{id: id, time: t, lat: lat, lng: lng, vars: vs}
	return s
}
