package main

import (
	"time"

	"github.com/nonnontrivial/ctts/internal/rowgen"
)

// struct for the sky quality meter read used during POST
type Read struct {
	// measured in mpsas, see:
	// http://www.unihedron./projects/darksky/faq.php#:~:text=The%20term%20magnitudes%20per%20square,square%20arcsecond%20of%20the%20sky.
	Brightness        float32   `json:"brightness"`
	Lat               string    `json:"lat"`
	Lng               string    `json:"lng"`
	TimeOfMeasurement time.Time `json:"timeOfMeasurement"`
}

func (r *Read) FindFeatures() (*Row, error) {
	g := rowgen.NewGenerator(r.Lat, r.Lng, r.TimeOfMeasurement)
	var independentVars []string
	if err := g.Backfill(&independentVars); err != nil {
		return nil, err
	}
	// brightness := strconv.FormatFloat(float64(r.Brightness), 'E', -1, 64)
	row := Row{}
	return &row, nil
}
