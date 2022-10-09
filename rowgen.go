package main

import (
	"context"
	"time"

	"github.com/nonnontrivial/ctts/internal/meteo"
	"golang.org/x/sync/errgroup"
)

type rowGenerator struct {
	lat               string
	lng               string
	timeOfMeasurement time.Time
}

type generator interface {
	Backfill(*Row) error
}

func (rg *rowGenerator) getElevation(mc *meteo.MeteoClient) func() (float32, error) {
	return func() (float32, error) {
		return mc.Data.Elevation, nil
	}
}

func (rg *rowGenerator) getTemperature(mc *meteo.MeteoClient) func() (float32, error) {
	return func() (float32, error) {
		return mc.Data.Temperature, nil
	}
}

func (rg *rowGenerator) Backfill(row *Row) error {
	_, err := meteo.SetupWeatherClient(rg.timeOfMeasurement, rg.lat, rg.lng)
	if err != nil {
		return err
	}
	ctx := context.Background()
	_, _ = errgroup.WithContext(ctx)
	// for k, v := range dl {
	// index := i
	// fn := f
	// eg.Go(func() error {
	// 	result, err := fn()
	// 	if err != nil {
	// 		return err
	// 	}
	// 	(*row)[index] = strconv.FormatFloat(float64(result), 'E', -1, 64)
	// 	return nil
	// })
	// }
	return nil
}

func NewRowGenerator(lat, lng string, t time.Time) generator {
	return &rowGenerator{lat, lng, t}
}
