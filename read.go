package main

import (
	"time"
)

type Read struct {
	// measured in mpsas, see http://www.unihedron./projects/darksky/faq.php#:~:text=The%20term%20magnitudes%20per%20square,square%20arcsecond%20of%20the%20sky.
	Brightness        float32   `json:"brightness"`
	Lat               string    `json:"lat"`
	Lng               string    `json:"lng"`
	TimeOfMeasurement time.Time `json:"timeOfMeasurement"`
}

func (r *Read) FindFeatures() (*Row, error) {
	g := NewRowGenerator(r.Lat, r.Lng, r.TimeOfMeasurement)
	var features Row
	if err := g.Backfill(&features); err != nil {
		return nil, err
	}
	row := Row{}
	return &row, nil
}
