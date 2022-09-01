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
	"path"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
)

var (
	client                  = &http.Client{Timeout: time.Second * 10}
	errMeteoResponse        = errors.New("got bad meteo response")
	errLocalTimezone        = errors.New("could not read timezone data")
	errUnsupportedRegion    = errors.New("this region is not yet supported")
	errTooFewTemporalParams = errors.New("too few hourly or daily params in request")
	hourlyParams            = []string{meteoTemperatureKey}
	dailyParams             = []string{meteoSunriseKey, meteoSunsetKey}
)

const (
	meteoURLBase        = "https://api.open-meteo.com/v1"
	meteoTemperatureKey = "temperature_2m"
	meteoSunriseKey     = "sunrise"
	meteoSunsetKey      = "sunset"
	// potential path segment in localtimezone symlink
	zoneinfo            = "zoneinfo"
	pathToLocalTimezone = "/etc/localtime"
)

type (
	isoTime       string
	fn            func(*site) (float32, error)
	columns       map[columnName]fn
	meteoCollator interface {
		getTemperature(*site) (float32, error)
		getElevation(*site) (float32, error)
		getResponse(*site, []string, []string) (*meteoResponse, error)
		setColumnarData(*meteoResponse)
	}
	meteoClient struct {
		columnarData struct {
			elevation   float32
			temperature float32
		}
	}
	meteoResponse struct {
		Elevation float32 `json:"elevation"`
		Daily     struct {
			Sunrise []isoTime `json:"sunrise"`
			Sunset  []isoTime `json:"sunset"`
		}
		Hourly struct {
			Time        []isoTime `json:"time"`
			Temperature []float32 `json:"temperature_2m"`
		} `json:"hourly"`
	}
)

// TODO:
func (m *meteoClient) getElevation(s *site) (float32, error) {
	return 0, nil
}

// TODO:
func (m *meteoClient) getTemperature(s *site) (float32, error) {
	return 0, nil
}

// getLocalTimezoneName gets the timezone name from linux symbolic link for use in
// the meteo client request
func getLocalTimezoneName() (string, error) {
	name, err := os.Readlink(pathToLocalTimezone)
	if err != nil {
		return "", errLocalTimezone
	}
	dir, file := path.Split(name)
	if dir == "" || file == "" {
		return "", errLocalTimezone
	}
	_, f := path.Split(dir[:len(dir)-1])
	if f == zoneinfo {
		return "", errUnsupportedRegion
	}
	return fmt.Sprintf("%s/%s", f, file), nil
}

// getResponse gets weather and elevation data from the open meteo api
func (m *meteoClient) getResponse(s *site, hourlyParams, dailyParams []string) (res *meteoResponse, err error) {
	if len(hourlyParams) == 0 || len(dailyParams) == 0 {
		return nil, errTooFewTemporalParams
	}
	timezone, err := getLocalTimezoneName()
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/forecast?latitude=%s&longitude=%s&hourly=%s&daily=%s&timezone=%s", meteoURLBase, s.lat, s.lng, strings.Join(hourlyParams, ","), strings.Join(dailyParams, ","), timezone)
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errMeteoResponse
	}
	bytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(bytes, &res); err != nil {
		return nil, err
	}
	return res, nil
}

// setColumnarData assigns data from the meteo response into the appropriate columns
func (m *meteoClient) setColumnarData(data *meteoResponse) {
	// TODO:
}

func newMeteoClient() meteoCollator {
	return &meteoClient{}
}

func (s *site) deriveIndependentVariables() error {
	m := newMeteoClient()
	data, err := m.getResponse(s, hourlyParams, dailyParams)
	if err != nil {
		return err
	}
	log.Printf("%+v", data)
	m.setColumnarData(data)
	c := columns{
		orderedColumns[0]: m.getElevation,
		orderedColumns[1]: m.getTemperature,
	}
	ctx := context.Background()
	eg, _ := errgroup.WithContext(ctx)
	for name, f := range c {
		columnName := name
		columnFn := f
		eg.Go(func() error {
			result, err := columnFn(s)
			if err != nil {
				return err
			}
			s.vars[columnName] = float64(result)
			return nil
		})
	}
	eg.Wait()
	return nil
}
