package meteo

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strings"
	"time"
)

var (
	errLocalTimezone     = errors.New("could not read timezone data")
	errUnsupportedRegion = errors.New("this region is not yet supported")
	errMeteoResponse     = errors.New("got bad meteo response")
	hourlyParams         = []string{meteoTemperatureKey}
	dailyParams          = []string{meteoSunriseKey, meteoSunsetKey}
)

const (
	meteoURLBase        = "https://api.open-meteo.com/v1"
	meteoTemperatureKey = "temperature_2m"
	meteoSunriseKey     = "sunrise"
	meteoSunsetKey      = "sunset"
	// potential path segment in localtimezone symlink; if found in path to
	// local timezone, timezone is determined to be unsupported; this is not
	// strictly necessary...
	zoneinfo            = "zoneinfo"
	pathToLocalTimezone = "/etc/localtime"
)

type MeteoClient struct {
	client *http.Client
	Data   struct {
		Elevation   float32
		Temperature float32
	}
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
	// FIXME: should not need to error
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
	log.Println("failed to find astro dusk index")
	return -1, nil
}

type fetchParams struct {
	t            time.Time
	lat          string
	lng          string
	hourlyParams []string
	dailyParams  []string
}

// formatTime formats a time to yyyy-mm-dd
func (mc *MeteoClient) formatTime(t time.Time) string {
	return fmt.Sprintf("%d-%02d-%02d", t.Year(), t.Month(), t.Day())
}

// fetchWeatherData uses open meteo api to put datapoints into `meteoClient` struct fields
func (mc *MeteoClient) fetchWeatherData(p fetchParams) error {
	timezone, err := getLocalTimezoneName()
	if err != nil {
		return err
	}
	startDate := mc.formatTime(p.t)
	endDate := mc.formatTime(p.t.Add(time.Hour * 24))
	log.Printf("using time range in meteo client: %s->%s", startDate, endDate)
	url := fmt.Sprintf("%s/forecast?latitude=%s&longitude=%s&hourly=%s&daily=%s&timezone=%s&start_date=%s&end_date=%s", meteoURLBase, p.lat, p.lng, strings.Join(p.hourlyParams, ","), strings.Join(p.dailyParams, ","), timezone, startDate, endDate)
	resp, err := mc.client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errMeteoResponse
	}
	bytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	var res meteoResponse
	if err := json.Unmarshal(bytes, &res); err != nil {
		return err
	}
	mc.Data.Elevation = res.Elevation
	idx, err := getHourlyAstroDuskIndex(res.Hourly.Time, res.Daily.Sunset)
	if err != nil {
		return err
	}
	mc.Data.Temperature = res.Hourly.Temperature[idx]
	return nil
}

// setupWeatherClient creates a new weather client in terms of the time and location of a brightness measurement
func SetupWeatherClient(t time.Time, lat, lng string) (*MeteoClient, error) {
	client := &MeteoClient{
		client: &http.Client{Timeout: time.Second * 10},
	}
	if err := client.fetchWeatherData(fetchParams{t, lat, lng, hourlyParams, dailyParams}); err != nil {
		return nil, err
	}
	return client, nil
}
