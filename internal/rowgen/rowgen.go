package rowgen

import (
	"context"
	"strconv"
	"time"

	"golang.org/x/sync/errgroup"
)

type rowGenerator struct {
	lat               string
	lng               string
	timeOfMeasurement time.Time
}

type generator interface {
	Backfill(*[]string) error
	getElevation(*meteoClient) func() (float32, error)
	getTemperature(*meteoClient) func() (float32, error)
}

func (rg *rowGenerator) getElevation(mc *meteoClient) func() (float32, error) {
	return func() (float32, error) {
		return mc.data.elevation, nil
	}
}

func (rg *rowGenerator) getTemperature(mc *meteoClient) func() (float32, error) {
	return func() (float32, error) {
		return mc.data.temperature, nil
	}
}

// Backfill fills in values for independent variables according to known
// conditions at measurement.
//
// `columns` order corresponds to the order of columns in the generated row.
func (rg *rowGenerator) Backfill(r *[]string) error {
	weatherClient, err := setupWeatherClient(rg.timeOfMeasurement, rg.lat, rg.lng)
	if err != nil {
		return err
	}
	columns := []func() (float32, error){
		rg.getElevation(weatherClient),
		rg.getTemperature(weatherClient),
	}
	ctx := context.Background()
	eg, _ := errgroup.WithContext(ctx)
	for i, f := range columns {
		index := i
		fn := f
		eg.Go(func() error {
			result, err := fn()
			if err != nil {
				return err
			}
			(*r)[index] = strconv.FormatFloat(float64(result), 'E', -1, 64)
			return nil
		})
	}
	return nil
}

func NewGenerator(lat, lng string, t time.Time) generator {
	return &rowGenerator{lat, lng, t}
}
