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
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/silentwraith/fraud_ml_platform/router/internal/canary"
	"github.com/silentwraith/fraud_ml_platform/router/internal/config"
	"github.com/silentwraith/fraud_ml_platform/router/internal/routing"
	"github.com/silentwraith/fraud_ml_platform/router/internal/server"
	"github.com/silentwraith/fraud_ml_platform/router/internal/shadow"
	"github.com/silentwraith/fraud_ml_platform/router/internal/upstream"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	cfg, err := config.Load()
	if err != nil {
		slog.Error("config load failed", "err", err)
		os.Exit(1)
	}

	db, err := pgxpool.New(context.Background(), cfg.PostgresDSN)
	if err != nil {
		slog.Error("postgres connect failed", "err", err)
		os.Exit(1)
	}
	defer db.Close()

	champion := upstream.New("champion", cfg.ChampionURL, cfg.UpstreamTimeoutMS)
	challenger := upstream.New("challenger", cfg.ChallengerURL, cfg.UpstreamTimeoutMS)

	state := routing.NewAtomicState(routing.State{
		CanaryEnabled:            false,
		ChallengerTrafficPercent: cfg.InitialCanaryPercent,
		ShadowPercent:            cfg.InitialShadowPercent,
	})

	store := canary.NewStore(db)
	dispatcher := shadow.NewDispatcher(challenger, cfg.ShadowBufferSize, cfg.ShadowWorkers)

	h := server.NewHandler(state, champion, challenger, dispatcher, store, cfg.AdminToken)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)
	mux.Handle("GET /metrics", promhttp.Handler())

	srv := &http.Server{
		Addr:    ":" + cfg.Port,
		Handler: server.Recovery(server.Logging(mux)),
	}

	go func() {
		slog.Info("router listening", "port", cfg.Port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	slog.Info("shutting down")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		slog.Error("server shutdown error", "err", err)
	}
	dispatcher.Shutdown()
	slog.Info("shutdown complete")
}
