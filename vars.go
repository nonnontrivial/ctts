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
	client               = &http.Client{Timeout: time.Second * 10}
	errMeteoResponse     = errors.New("got bad meteo response")
	errLocalTimezone     = errors.New("could not read timezone data")
	errUnsupportedRegion = errors.New("this region is not yet supported")
)

const (
	// potential path segment in localtimezone symlink
	zoneinfo            = "zoneinfo"
	pathToLocalTimezone = "/etc/localtime"
	meteoURLBase        = "https://api.open-meteo.com/v1"
	meteoTemperatureKey = "temperature_2m"
)

type (
	fn            func(*site) (float32, error)
	columns       map[columnName]fn
	meteoCollator interface {
		getTemperature(*site) (float32, error)
		getElevation(*site) (float32, error)
		getData(*site) error
	}
	meteoClient struct {
		response  hourlyData
		duskIndex int
	}
	// TODO: include daily data
	hourlyData struct {
		Elevation float32 `json:"elevation"`
		Hourly    struct {
			Time        []string  `json:"time"`
			Temperature []float32 `json:"temperature_2m"`
		} `json:"hourly"`
	}
)

func (m *meteoClient) getElevation(s *site) (float32, error) {
	return 0, nil
}

func (m *meteoClient) getTemperature(s *site) (float32, error) {
	return 0, nil
}

// getHourlyIndexOfAstronomicalDusk gets the index from the meteo client response
// that corresponds to the upcoming hour of astronomical dusk
func getHourlyIndexOfAstronomicalDusk(hours []string, requestTime time.Time, lat, lng string) (int, error) {
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

// TODO: signature of this should probably change to return a value
func (m *meteoClient) getData(s *site) error {
	params := []string{meteoTemperatureKey}
	timezone, err := getLocalTimezoneName()
	if err != nil {
		return err
	}
	// FIXME: url needs more params
	url := fmt.Sprintf("%s/forecast?latitude=%s&longitude=%s&hourly=%s&timezone=%s", meteoURLBase, s.lat, s.lng, strings.Join(params, ","), timezone)
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
	var res hourlyData
	if err := json.Unmarshal(bytes, &res); err != nil {
		return err
	}
	m.response = res
	idx, err := getHourlyIndexOfAstronomicalDusk(m.response.Hourly.Time, s.time, s.lat, s.lng)
	if err != nil {
		return err
	}
	m.duskIndex = idx
	return nil
}

func newMeteoClient() meteoCollator {
	return &meteoClient{}
}

func (s *site) deriveIndependentVariables() error {
	m := newMeteoClient()
	if err := m.getData(s); err != nil {
		return err
	}
	c := columns{
		orderedColumns[0]: m.getElevation,
		orderedColumns[1]: m.getTemperature,
	}
	ctx := context.Background()
	errs, _ := errgroup.WithContext(ctx)
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
