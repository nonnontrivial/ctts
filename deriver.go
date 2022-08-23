package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
)

var (
	client               = &http.Client{Timeout: time.Second * 10}
	mapsAPIKey           = os.Getenv("MAPS_API_KEY")
	errMeteoResponse     = errors.New("got bad meteo response")
	errElevationResponse = errors.New("got bad elevation response")
	errNoMapsKey         = errors.New("missing google maps api key")
)

const (
	elevationURLBase = "https://maps.googleapis.com/maps/api/elevation/json"
	meteoURLBase     = "https://api.open-meteo.com/v1"
)

type (
	fn       func(*site) (float64, error)
	columns  map[columnName]fn
	collator interface {
		getTemperature(*site) (float64, error)
		getRelativeHumidity(*site) (float64, error)
		getCloudCoverTotal(*site) (float64, error)
		getWindSpeed(*site) (float64, error)
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
	elevationResponse struct {
		Results []struct {
			Elevation float32 `json:"elevation"`
		}
	}
)

func getElevation(s *site) (float64, error) {
	if mapsAPIKey == "" {
		return 0, errNoMapsKey
	}
	url := fmt.Sprintf("%s?locations=%s,%s&key=%s", elevationURLBase, s.lat, s.lng, mapsAPIKey)
	resp, err := client.Get(url)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, errElevationResponse
	}
	var elevationRes elevationResponse
	err = json.NewDecoder(resp.Body).Decode(&elevationRes)
	if err != nil {
		return 0, err
	}
	if len(elevationRes.Results) < 1 {
		return 0, nil
	}
	return float64(elevationRes.Results[len(elevationRes.Results)-1].Elevation), nil
}

func (m *meteoClient) getTemperature(s *site) (float64, error) {
	return 0, nil
}

func (m *meteoClient) getRelativeHumidity(s *site) (float64, error) {
	return 0, nil
}

func (m *meteoClient) getCloudCoverTotal(s *site) (float64, error) {
	return 0, nil
}

func (m *meteoClient) getWindSpeed(s *site) (float64, error) {
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
		orderedColumns[0]: getElevation,
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
			s.vars[columnName] = result
			return nil
		})
	}
	errs.Wait()
	return nil
}
