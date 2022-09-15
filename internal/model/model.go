package model

type (
	model struct {
	}
	modeller interface {
		Train(string) error
		Predict(vars []float64) (float64, error)
	}
)

// TODO:
func (m *model) Train(yValueKey string) error {
	return nil
}

// TODO:
func (m *model) Predict(vars []float64) (float64, error) {
	return 0, nil
}

func NewModel(trainingPath, testingPath, csvPath string) (modeller, error) {
	m := &model{}
	return m, nil
}
