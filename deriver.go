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
	weatherAPIKey   = os.Getenv("WEATHER_API_KEY")
	mapsAPIKey      = os.Getenv("MAPS_API_KEY")
	errNoWeatherKey = errors.New("missing oiko weather api key")
	errNoMapsKey    = errors.New("missing google maps api key")
	client          = &http.Client{Timeout: time.Second * 10}
)

const (
	elevationURLBase = "https://maps.googleapis.com/maps/api/elevation/json"
	weatherURLBase   = "https://api.oikolab.com/weather"
)

type (
	fn                func(*site) (float64, error)
	columns           map[columnName]fn
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
	if weatherAPIKey == "" {
		return 0, errNoWeatherKey
	}
	return 0., nil
}

func (s *site) derive() error {
	var wg sync.WaitGroup
	c := columns{
		columnOrder[0]: getElevation,
		columnOrder[1]: getCloudCover,
	}
	errChan := make(chan error)
	for name, f := range c {
		wg.Add(1)
		go func(name columnName, f fn) {
			defer wg.Done()
			result, err := f(s)
			if err != nil {
				errChan <- err
			}
			s.vars[name] = result
		}(name, f)
	}
	if err := <-errChan; err != nil {
		return err
	}
	wg.Wait()
	log.Println("finished...")
	return nil
}
