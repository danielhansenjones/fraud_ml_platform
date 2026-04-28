package config

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Port                    string
	ModelPath               string
	FeatureOrderPath        string
	CategoricalEncoderPath  string
	RedisHost               string
	RedisPort               string
	PostgresDSN             string
	FlagThreshold           float64
	PredictionBufferSize    int
	PredictionFlushInterval int // milliseconds
	LogLevel                string
}

func Load() (*Config, error) {
	cfg := &Config{
		Port:                   getEnv("PORT", "8080"),
		ModelPath:              mustGetEnv("MODEL_PATH"),
		FeatureOrderPath:       mustGetEnv("FEATURE_ORDER_PATH"),
		CategoricalEncoderPath: mustGetEnv("CATEGORICAL_ENCODER_PATH"),
		RedisHost:              getEnv("REDIS_HOST", "localhost"),
		RedisPort:              getEnv("REDIS_PORT", "6379"),
		PostgresDSN:            mustGetEnv("POSTGRES_DSN"),
		LogLevel:               getEnv("LOG_LEVEL", "info"),
	}

	var err error
	cfg.FlagThreshold, err = parseFloat("FLAG_THRESHOLD", "0.5")
	if err != nil {
		return nil, err
	}
	cfg.PredictionBufferSize, err = parseInt("PREDICTION_BUFFER_SIZE", "1000")
	if err != nil {
		return nil, err
	}
	cfg.PredictionFlushInterval, err = parseInt("PREDICTION_FLUSH_INTERVAL_MS", "1000")
	if err != nil {
		return nil, err
	}

	return cfg, nil
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func mustGetEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		panic(fmt.Sprintf("required environment variable %q is not set", key))
	}
	return v
}

func parseFloat(key, fallback string) (float64, error) {
	s := getEnv(key, fallback)
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid value for %s: %w", key, err)
	}
	return v, nil
}

func parseInt(key, fallback string) (int, error) {
	s := getEnv(key, fallback)
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("invalid value for %s: %w", key, err)
	}
	return v, nil
}
