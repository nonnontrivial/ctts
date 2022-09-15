package rowgen

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
	"strconv"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
)

type meteoClient struct {
	elevation   float32
	temperature float32
}
type meteoResponse struct {
	Elevation float32 `json:"elevation"`
	Daily     struct {
		Sunrise []string `json:"sunrise"`
		Sunset  []string `json:"sunset"`
	}
	Hourly struct {
		Time        []string  `json:"time"`
		Temperature []float32 `json:"temperature_2m"`
	} `json:"hourly"`
}

type rowGenerator struct {
	lat               string
	lng               string
	timeOfMeasurement time.Time
}

type generator interface {
	Backfill(*[]string) error
	getElevation(*meteoClient) func() (float32, error)
	getTemperature(*meteoClient) func() (float32, error)
}

func (rg *rowGenerator) getElevation(mc *meteoClient) func() (float32, error) {
	return func() (float32, error) {
		return mc.elevation, nil
	}
}

func (rg *rowGenerator) getTemperature(mc *meteoClient) func() (float32, error) {
	return func() (float32, error) {
		return mc.temperature, nil
	}
}

var (
	errTooFewTemporalParamsInMeteoClient = errors.New("too few hourly or daily params in request")
	errLocalTimezone                     = errors.New("could not read timezone data")
	errUnsupportedRegion                 = errors.New("this region is not yet supported")
	errMeteoResponse                     = errors.New("got bad meteo response")
	hourlyParams                         = []string{meteoTemperatureKey}
	dailyParams                          = []string{meteoSunriseKey, meteoSunsetKey}
	client                               = &http.Client{Timeout: time.Second * 10}
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

// getLocalTimezoneName gets the timezone name from linux symbolic link for
// use in the meteo client request
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

// parseMeteoTime parses the string from the meteo response
func parseMeteoTime(t string) (time.Time, error) {
	formatted := fmt.Sprintf("%s:05Z", string(t))
	return time.Parse(time.RFC3339, formatted)
}

// getHourlyAstroDuskIndex gets the index (within hourly data) of approximate
// astronomical dusk
func getHourlyAstroDuskIndex(hourly, sunset []string) (int, error) {
	parsedTime, err := parseMeteoTime(sunset[0])
	if err != nil {
		return 0, err
	}
	approxAstroDuskTime := parsedTime.Add(time.Hour * 2)
	for i, iso := range hourly {
		pt, err := parseMeteoTime(iso)
		if err != nil {
			return -1, err
		}
		// FIXME: should be hour(?)
		log.Println(pt.Day(), approxAstroDuskTime.Day())
		if pt.Day() == approxAstroDuskTime.Day() {
			return i, err
		}
	}
	return -1, nil
}

// fetchWeatherData uses the open meteo api to put datapoints intot eh meteo client struct fields.
func (mc *meteoClient) fetchWeatherData(lat, lng string, hourlyParams, dailyParams []string) error {
	if len(hourlyParams) == 0 || len(dailyParams) == 0 {
		return errTooFewTemporalParamsInMeteoClient
	}
	timezone, err := getLocalTimezoneName()
	if err != nil {
		return err
	}
	url := fmt.Sprintf("%s/forecast?latitude=%s&longitude=%s&hourly=%s&daily=%s&timezone=%s", meteoURLBase, lat, lng, strings.Join(hourlyParams, ","), strings.Join(dailyParams, ","), timezone)
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
	mc.elevation = res.Elevation
	idx, err := getHourlyAstroDuskIndex(res.Hourly.Time, res.Daily.Sunset)
	if err != nil {
		return err
	}
	mc.temperature = res.Hourly.Temperature[idx]
	return nil
}

func setupWeatherClient(lat, lng string) (*meteoClient, error) {
	client := &meteoClient{}
	if err := client.fetchWeatherData(lat, lng, hourlyParams, dailyParams); err != nil {
		return nil, err
	}
	return &meteoClient{}, nil
}

// Backfill fills in values for independent variables according to known
// conditions at measurement.
//
// `columns` order corresponds to the order of columns in the generated row.
func (rg *rowGenerator) Backfill(r *[]string) error {
	weatherClient, err := setupWeatherClient(rg.lat, rg.lng)
	if err != nil {
		return err
	}
	columns := []func() (float32, error){
		rg.getElevation(weatherClient),
		rg.getTemperature(weatherClient),
	}
	ctx := context.Background()
	eg, _ := errgroup.WithContext(ctx)
	for i, f := range columns {
		index := i
		fn := f
		eg.Go(func() error {
			result, err := fn()
			if err != nil {
				return err
			}
			(*r)[index] = strconv.FormatFloat(float64(result), 'E', -1, 64)
			return nil
		})
	}
	return nil
}

func NewGenerator(lat, lng string, t time.Time) generator {
	return &rowGenerator{lat, lng, t}
}
