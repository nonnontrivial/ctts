package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

var (
	client          = &http.Client{Timeout: time.Second * 10}
	weatherAPIKey   = os.Getenv("WEATHER_API_KEY")
	mapsAPIKey      = os.Getenv("MAPS_API_KEY")
	errNoWeatherKey = errors.New("missing oiko weather api key")
	errNoMapsKey    = errors.New("missing google maps api key")
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
		setResponse(wr *weatherResponse)
		getWeatherResultColumnData(c string) []float32
	}
	weatherResp struct {
		weatherResponse
	}
	weatherResponse struct {
		Data struct {
			Columns []string    `json:"columns"`
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
	var elevationRes elevationResponse
	err = json.NewDecoder(resp.Body).Decode(&elevationRes)
	if err != nil {
		return 0, err
	}
	return float64(elevationRes.Results[len(elevationRes.Results)-1].Elevation), nil
}

func (w *weatherResp) getTemperature(s *site) (float64, error) {
	return 0., nil
}

func (w *weatherResp) getCloudCover(s *site) (float64, error) {
	return 0., nil
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

func (w *weatherResp) setResponse(resp *weatherResponse) {
	log.Println("setting weather response...")
	w.weatherResponse = *resp
}

func newWeatherResp() weatherCollator {
	return &weatherResp{}
}

func (s *site) derive() error {
	errChan := make(chan error)
	weatherChan := make(chan weatherResponse)
	// populate the weather results before column-based, weather-related assignment happens
	go func() {
		if weatherAPIKey == "" {
			errChan <- errNoWeatherKey
			return
		}
		start := fmt.Sprintf("%d-%02d-%02d", s.time.Year(), s.time.Month(), s.time.Day())
		url := fmt.Sprintf("%s?start=%s&lat=%s&lng=%s&api-key=%s", weatherURLBase, start, s.lat, s.lng, weatherAPIKey)
		resp, err := client.Get(url)
		if err != nil {
			errChan <- err
			return
		}
		defer resp.Body.Close()
		var weatherRes weatherResponse
		err = json.NewDecoder(resp.Body).Decode(&weatherRes)
		if err != nil {
			errChan <- err
			return
		}
		weatherChan <- weatherRes
	}()
	weatherResult := <-weatherChan
	err := <-errChan
	if err != nil {
		return err
	}
	w := newWeatherResp()
	w.setResponse(&weatherResult)

	var wg sync.WaitGroup
	c := columns{
		columnOrder[0]: getElevation,
		columnOrder[1]: w.getCloudCover,
		columnOrder[2]: w.getTemperature,
	}
	for name, f := range c {
		wg.Add(1)
		go func(name columnName, f fn) {
			defer wg.Done()
			result, err := f(s)
			if err != nil {
				errChan <- err
				return
			}
			s.vars[name] = result
		}(name, f)
	}
	if err != nil {
		return err
	}
	wg.Wait()
	log.Println("finished...")
	return nil
}
