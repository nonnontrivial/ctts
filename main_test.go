package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHandleSite(t *testing.T) {
	s := newServer()
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/site", nil)
	s.ServeHTTP(w, r)
	if statusCode := w.Result().StatusCode; statusCode != http.StatusOK {
		t.Errorf("status code is %d", statusCode)
	}
}

func TestHandleRoot(t *testing.T) {
	s := newServer()
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/", nil)
	s.ServeHTTP(w, r)
	if statusCode := w.Result().StatusCode; statusCode != http.StatusOK {
		t.Errorf("status code is %d", statusCode)
	}
}
