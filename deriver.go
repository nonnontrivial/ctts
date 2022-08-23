package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
)

var (
	client               = &http.Client{Timeout: time.Second * 10}
	weatherAPIKey        = os.Getenv("WEATHER_API_KEY")
	mapsAPIKey           = os.Getenv("MAPS_API_KEY")
	errWeatherResponse   = errors.New("got bad weather response")
	errElevationResponse = errors.New("got bad elevation response")
	errNoWeatherKey      = errors.New("missing oiko weather api key")
	errNoMapsKey         = errors.New("missing google maps api key")
)

const (
	elevationURLBase = "https://maps.googleapis.com/maps/api/elevation/json"
	weatherURLBase   = "https://api.oikolab.com/weather"
)

type (
	fn              func(*site) (float64, error)
	columns         map[columnName]fn
	weatherCollator interface {
		getTemperature(*site) (float64, error)
		getCloudCover(*site) (float64, error)
		getResponse(*site) (*weatherResponse, error)
		setResponse(wr *weatherResponse)
		getWeatherResultColumnData(c string) []float32
	}
	weatherResp struct {
		weatherResponse
	}
	weatherResponse struct {
		Attributes struct{}
		Data       struct {
			Columns []string    `json:"columns"`
			Index   []float32   `json:"index"`
			Data    [][]float32 `json:"data"`
		}
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

func (w *weatherResp) getTemperature(s *site) (float64, error) {
	data := w.getWeatherResultColumnData("temperature")
	return float64(data[0]), nil
}

func (w *weatherResp) getCloudCover(s *site) (float64, error) {
	data := w.getWeatherResultColumnData("cloud_cover")
	return float64(data[0]), nil
}

func (w *weatherResp) getWeatherResultColumnData(c string) []float32 {
	idx := -1
	for i, v := range w.weatherResponse.Data.Columns {
		if strings.HasPrefix(v, c) {
			idx = i
		}
	}
	if idx == -1 {
		return []float32{}
	}
	return w.weatherResponse.Data.Data[idx]
}

func (w *weatherResp) getResponse(s *site) (*weatherResponse, error) {
	if weatherAPIKey == "" {
		return nil, errNoWeatherKey
	}
	start := fmt.Sprintf("%d-%02d-%02d", s.time.Year(), s.time.Month(), s.time.Day())
	url := fmt.Sprintf("%s?start=%s&lat=%s&lon=%s&api-key=%s", weatherURLBase, start, s.lat, s.lng, weatherAPIKey)
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errWeatherResponse
	}
	var weatherRes weatherResponse
	// FIXME: does not decode correctly
	// see https://stackoverflow.com/a/47063104
	err = json.NewDecoder(resp.Body).Decode(&weatherRes)
	if err != nil {
		return nil, err
	}
	return &weatherRes, nil
}

func (w *weatherResp) setResponse(wr *weatherResponse) {
	log.Println("setting weather response")
	w.weatherResponse = *wr
}

func newWeatherResp() weatherCollator {
	return &weatherResp{}
}

func (s *site) derive() error {
	ctx := context.Background()
	errs, _ := errgroup.WithContext(ctx)
	w := newWeatherResp()
	c := columns{
		orderedColumns[0]: getElevation,
		orderedColumns[1]: w.getCloudCover,
		orderedColumns[2]: w.getTemperature,
	}
	errs.Go(func() error {
		res, err := w.getResponse(s)
		if err != nil {
			return err
		}
		w.setResponse(res)
		return nil
	})
	for name, f := range c {
		errs.Go(func() error {
			result, err := f(s)
			if err != nil {
				return err
			}
			log.Printf("setting %s", name)
			s.vars[name] = result
			return nil
		})
	}
	errs.Wait()
	return nil
}
