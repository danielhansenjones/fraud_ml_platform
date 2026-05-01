package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/config"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/features"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/metrics"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/model"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/predictions"
	srv "github.com/silentwraith/fraud_ml_platform/serving/internal/server"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		slog.Error("config error", "err", err)
		os.Exit(1)
	}

	logLevel := slog.LevelInfo
	if cfg.LogLevel == "debug" {
		logLevel = slog.LevelDebug
	}
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: logLevel})))

	reg := prometheus.NewRegistry()
	metrics.Register(reg)

	featClient := features.NewClient(cfg.RedisHost, cfg.RedisPort)
	ctx := context.Background()
	if err := featClient.Ping(ctx); err != nil {
		slog.Error("redis ping failed", "err", err)
		os.Exit(1)
	}
	slog.Info("redis connected")

	pool, err := pgxpool.New(ctx, cfg.PostgresDSN)
	if err != nil {
		slog.Error("postgres pool creation failed", "err", err)
		os.Exit(1)
	}
	if err := pool.Ping(ctx); err != nil {
		slog.Error("postgres ping failed", "err", err)
		os.Exit(1)
	}
	slog.Info("postgres connected")

	runner, err := model.NewRunner(cfg.ModelPath, cfg.FeatureOrderPath)
	if err != nil {
		slog.Error("model load failed", "err", err)
		os.Exit(1)
	}
	slog.Info("model loaded", "version", runner.ModelVersion())

	store := predictions.NewStore(pool, cfg.PredictionBufferSize, cfg.PredictionFlushInterval)

	handler := srv.NewHandler(featClient, runner, store, cfg.FlagThreshold)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handler.Health)
	mux.HandleFunc("POST /score", handler.Score)
	mux.Handle("GET /metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))

	chain := srv.Recovery(srv.Logging(mux))

	httpServer := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      chain,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		slog.Info("server starting", "addr", httpServer.Addr)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	<-quit
	slog.Info("shutting down")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		slog.Error("http shutdown error", "err", err)
	}

	handler.Drain()
	store.Shutdown()
	pool.Close()
	runner.Destroy()
	featClient.Close()

	slog.Info("shutdown complete")
}
