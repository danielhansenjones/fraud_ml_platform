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
	RedisHost               string
	RedisPort               string
	PostgresDSN             string
	AdminToken              string
	FlagThreshold           float64
	PredictionBufferSize    int
	PredictionFlushInterval int // milliseconds
	LogLevel                string
}

func Load() (*Config, error) {
	required := []struct {
		key string
		dst *string
	}{}

	cfg := &Config{
		Port:      getEnv("PORT", "8080"),
		RedisHost: getEnv("REDIS_HOST", "localhost"),
		RedisPort: getEnv("REDIS_PORT", "6379"),
		LogLevel:  getEnv("LOG_LEVEL", "info"),
	}

	required = append(required,
		struct {
			key string
			dst *string
		}{"MODEL_PATH", &cfg.ModelPath},
		struct {
			key string
			dst *string
		}{"FEATURE_ORDER_PATH", &cfg.FeatureOrderPath},
		struct {
			key string
			dst *string
		}{"POSTGRES_DSN", &cfg.PostgresDSN},
		struct {
			key string
			dst *string
		}{"ADMIN_TOKEN", &cfg.AdminToken},
	)
	for _, r := range required {
		v := os.Getenv(r.key)
		if v == "" {
			return nil, fmt.Errorf("required environment variable %q is not set", r.key)
		}
		*r.dst = v
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
