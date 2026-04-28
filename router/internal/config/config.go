package config

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Port                 string
	ChampionURL          string
	ChallengerURL        string
	AdminToken           string
	PostgresDSN          string
	InitialCanaryPercent int
	InitialShadowPercent int
	UpstreamTimeoutMS    int
	ShadowBufferSize     int
	ShadowWorkers        int
}

func Load() (*Config, error) {
	cfg := &Config{
		Port:                 getEnv("PORT", "8081"),
		ChampionURL:          getEnv("CHAMPION_URL", "http://champion-model:8080"),
		ChallengerURL:        getEnv("CHALLENGER_URL", "http://challenger-model:8080"),
		AdminToken:           getEnv("ADMIN_TOKEN", ""),
		PostgresDSN:          getEnv("POSTGRES_DSN", ""),
		InitialCanaryPercent: getEnvInt("INITIAL_CANARY_PERCENT", 0),
		InitialShadowPercent: getEnvInt("INITIAL_SHADOW_PERCENT", 100),
		UpstreamTimeoutMS:    getEnvInt("UPSTREAM_TIMEOUT_MS", 500),
		ShadowBufferSize:     getEnvInt("SHADOW_BUFFER_SIZE", 256),
		ShadowWorkers:        getEnvInt("SHADOW_WORKERS", 4),
	}

	if cfg.AdminToken == "" {
		return nil, fmt.Errorf("ADMIN_TOKEN is required")
	}
	if cfg.PostgresDSN == "" {
		return nil, fmt.Errorf("POSTGRES_DSN is required")
	}

	return cfg, nil
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		n, err := strconv.Atoi(v)
		if err == nil {
			return n
		}
	}
	return fallback
}
