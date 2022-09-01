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
	errTooFewTemporalParams = errors.New("got too few hourly or daily params in request")
)

const (
	// potential path segment in localtimezone symlink
	zoneinfo            = "zoneinfo"
	pathToLocalTimezone = "/etc/localtime"
	meteoURLBase        = "https://api.open-meteo.com/v1"
	meteoTemperatureKey = "temperature_2m"
	meteoSunriseKey     = "sunrise"
	meteoSunsetKey      = "sunset"
)

type (
	fn            func(*site) (float32, error)
	columns       map[columnName]fn
	meteoCollator interface {
		getTemperature(*site) (float32, error)
		getElevation(*site) (float32, error)
		setData(*meteoResponse)
		getResponse(*site, []string, []string) (*meteoResponse, error)
		getData() *meteoResponse
	}
	meteoResponse struct {
		Elevation float32 `json:"elevation"`
		Daily     struct {
			// 7 values of RFC3339
			Sunrise []string `json:"sunrise"`
			Sunset  []string `json:"sunset"`
		}
		Hourly struct {
			Time        []string  `json:"time"`
			Temperature []float32 `json:"temperature_2m"`
		} `json:"hourly"`
	}
	meteoClient struct {
		data meteoResponse
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

// getHourlyIndexOfAstronomicalDusk gets the index from the meteo client response
// that corresponds to the upcoming hour of astronomical dusk
func getHourlyIndexOfAstronomicalDusk(hours []string, sunrise, sunset string) (int, error) {
	fmt.Println(hours, sunrise, sunset)
	return -1, nil
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

func (m *meteoClient) getData() *meteoResponse { return &m.data }
func (m *meteoClient) setData(data *meteoResponse) {
	m.data = *data
}

func newMeteoClient() meteoCollator {
	return &meteoClient{}
}

func (s *site) deriveIndependentVariables() error {
	m := newMeteoClient()
	hourlyParams := []string{meteoTemperatureKey}
	dailyParams := []string{meteoSunriseKey, meteoSunsetKey}
	data, err := m.getResponse(s, hourlyParams, dailyParams)
	if err != nil {
		return err
	}
	// TODO: use index...
	_, err = getHourlyIndexOfAstronomicalDusk(data.Hourly.Time, data.Daily.Sunrise[0], data.Daily.Sunset[0])
	if err != nil {
		return err
	}
	m.setData(data)
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
			log.Printf("setting %s", columnName)
			s.vars[columnName] = float64(result)
			return nil
		})
	}
	eg.Wait()
	return nil
}
