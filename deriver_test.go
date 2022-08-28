package main

import "testing"

func TestDerive(t *testing.T) {
	s := &site{}
	if err := s.deriveIndependentVariables(); err != nil {
		t.Fail()
	}
}
