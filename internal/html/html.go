package html

import (
	"embed"
	"io"
	"text/template"
)

var (
	//go:embed *
	files     embed.FS
	dashboard = parse("dashboard.html")
)

const (
	layoutPath = "layout.html"
)

type (
	User struct {
	}
	DashboardParams struct {
		User
	}
)

func Dashboard(w io.Writer, p DashboardParams) error {
	return dashboard.Execute(w, p)
}

// parse
func parse(file string) *template.Template {
	return template.Must(template.New(layoutPath).ParseFS(files, layoutPath, file))
}
