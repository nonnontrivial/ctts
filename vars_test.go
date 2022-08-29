package main

import "testing"

func TestDerive(t *testing.T) {
	s := &site{lat: "42", lng: "42"}
	if err := s.deriveIndependentVariables(); err != nil {
		t.Error(err)
	}
}
