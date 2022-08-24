package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
)

var (
	client           = &http.Client{Timeout: time.Second * 10}
	errMeteoResponse = errors.New("got bad meteo response")
)

const (
	meteoURLBase = "https://api.open-meteo.com/v1"
)

type (
	fn       func(*site) (float32, error)
	columns  map[columnName]fn
	collator interface {
		getTemperature(*site) (float32, error)
		getRelativeHumidity(*site) (float32, error)
		getCloudCoverTotal(*site) (float32, error)
		getWindSpeed(*site) (float32, error)
		getElevation(*site) (float32, error)
		getResponse(*site) error
	}
	meteoClient struct {
		response meteoResponse
		data     struct {
			cloudCover  float32
			humidity    float32
			temperature float32
			windspeed   float32
		}
	}
	meteoResponse struct {
		Elevation float32 `json:"elevation"`
		Hourly    struct {
			Cloudcover  []float32 `json:"cloudcover"`
			Humidity    []float32 `json:"relativehumidity_2m"`
			Temperature []float32 `json:"temperature_2m"`
			Windspeed   []float32 `json:"windspeed_10m"`
		} `json:"hourly"`
	}
)

func (m *meteoClient) getElevation(s *site) (float32, error) {
	return 0, nil
}

func (m *meteoClient) getTemperature(s *site) (float32, error) {
	return 0, nil
}

func (m *meteoClient) getRelativeHumidity(s *site) (float32, error) {
	return 0, nil
}

func (m *meteoClient) getCloudCoverTotal(s *site) (float32, error) {
	return 0, nil
}

func (m *meteoClient) getWindSpeed(s *site) (float32, error) {
	return 0, nil
}

func (m *meteoClient) getResponse(s *site) error {
	params := []string{"temperature_2m", "relativehumidity_2m", "cloudcover", "windspeed_10m"}
	url := fmt.Sprintf("%s/forecast?latitude=%s&longitude=%s&hourly=%s", meteoURLBase, s.lat, s.lng, strings.Join(params, ","))
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errMeteoResponse
	}
	bytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	var res meteoResponse
	if err := json.Unmarshal(bytes, &res); err != nil {
		return err
	}
	m.response = res
	return nil
}

func newMeteoClient() collator {
	return &meteoClient{}
}

func (s *site) derive() error {
	ctx := context.Background()
	errs, _ := errgroup.WithContext(ctx)
	m := newMeteoClient()
	if err := m.getResponse(s); err != nil {
		return err
	}
	fmt.Println(m)
	c := columns{
		orderedColumns[0]: m.getElevation,
		orderedColumns[2]: m.getTemperature,
	}
	for name, f := range c {
		columnName := name
		columnFn := f
		errs.Go(func() error {
			result, err := columnFn(s)
			if err != nil {
				return err
			}
			log.Printf("setting %s", columnName)
			s.vars[columnName] = float64(result)
			return nil
		})
	}
	errs.Wait()
	return nil
}
