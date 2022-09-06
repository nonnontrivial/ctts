package html

import (
	"embed"
	"io"
	"text/template"
)

var (
	//go:embed *
	files embed.FS
	site  = parse("pages/site.html")
)

const (
	layoutPath = "layout.html"
)

type (
	SiteParams struct {
		Lat, Lng string
		Mpsas    float32
	}
)

func Site(w io.Writer, p SiteParams) error {
	return site.Execute(w, p)
}

// parse parses the provided html file in terms of the layout
func parse(file string) *template.Template {
	return template.Must(template.New(layoutPath).ParseFS(files, layoutPath, file))
}
