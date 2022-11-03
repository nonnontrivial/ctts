package main

import (
	"bufio"
	"log"
	"os"
	"strings"
)

func loadEnv(pathToEnvFile string) (err error) {
	file, err := os.Open(pathToEnvFile)
	if err != nil {
		return err
	}
	defer file.Close()
	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	for _, l := range lines {
		s := strings.Split(l, "=")
		if len(s) != 2 {
			log.Printf("!failed to parse env var from line: %s", l)
			continue
		}
		log.Printf("setting env var %s", s[0])
		os.Setenv(s[0], s[1])
	}
	return nil
}
