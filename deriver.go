package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

var (
	client          = &http.Client{Timeout: time.Second * 10}
	weatherResult   weatherResponse
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
	weatherResponse struct {
		Data struct {
			Columns []string `json:"columns"`
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

func getCloudCover(s *site) (float64, error) {
	return 0., nil
}

func getTemperature(s *site) (float64, error) {
	return 0, nil
}

func getWindSpeed(s *site) (float64, error) {
	return 0, nil
}

func getAirMoisture(s *site) (float64, error) {
	return 0, nil
}

func (s *site) derive() error {
	var wg sync.WaitGroup
	c := columns{
		columnOrder[0]: getElevation,
		columnOrder[1]: getCloudCover,
		columnOrder[2]: getTemperature,
		columnOrder[3]: getWindSpeed,
		columnOrder[4]: getAirMoisture,
	}
	errChan := make(chan error)
	weatherChan := make(chan weatherResponse)
	// populate the weather results before column-based, weather-related assignment happens
	go func() {
		if weatherAPIKey == "" {
			errChan <- errNoWeatherKey
			return
		}
		url := fmt.Sprintf("%s/", weatherURLBase)
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
	// used by other methods; should change signature
	weatherResult = <-weatherChan
	err := <-errChan
	if err != nil {
		return err
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
