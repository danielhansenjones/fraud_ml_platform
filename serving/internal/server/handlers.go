package server

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"log/slog"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/features"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/metrics"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/predictions"
)

type FeatureGetter interface {
	Get(ctx context.Context, transactionID int64) (map[string]float32, error)
}

type ModelScorer interface {
	Score(ctx context.Context, features map[string]float32) (float32, string, error)
	ModelVersion() string
}

type ModelReloader interface {
	Reload(modelPath, featureOrderPath string) (string, error)
}

type PredictionLogger interface {
	Log(ctx context.Context, r predictions.Record) error
}

type Pinger interface {
	Ping(ctx context.Context) error
}

type Handler struct {
	featureClient  FeatureGetter
	modelRunner    ModelScorer
	reloader       ModelReloader
	store          PredictionLogger
	redisPinger    Pinger
	postgresPinger Pinger
	threshold      float64
	adminToken     string
	logWg          sync.WaitGroup
}

func NewHandler(
	featureClient FeatureGetter,
	modelRunner ModelScorer,
	reloader ModelReloader,
	store PredictionLogger,
	redisPinger, postgresPinger Pinger,
	threshold float64,
	adminToken string,
) *Handler {
	return &Handler{
		featureClient:  featureClient,
		modelRunner:    modelRunner,
		reloader:       reloader,
		store:          store,
		redisPinger:    redisPinger,
		postgresPinger: postgresPinger,
		threshold:      threshold,
		adminToken:     adminToken,
	}
}

type healthResponse struct {
	Status       string `json:"status"`
	ModelVersion string `json:"model_version"`
	Redis        string `json:"redis"`
	Postgres     string `json:"postgres"`
}

func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 500*time.Millisecond)
	defer cancel()

	resp := healthResponse{
		Status:       "ok",
		ModelVersion: h.modelRunner.ModelVersion(),
		Redis:        "ok",
		Postgres:     "ok",
	}
	status := http.StatusOK

	if h.redisPinger != nil {
		if err := h.redisPinger.Ping(ctx); err != nil {
			resp.Status = "unhealthy"
			resp.Redis = "err: " + err.Error()
			status = http.StatusServiceUnavailable
		}
	}
	if h.postgresPinger != nil {
		if err := h.postgresPinger.Ping(ctx); err != nil {
			resp.Status = "unhealthy"
			resp.Postgres = "err: " + err.Error()
			status = http.StatusServiceUnavailable
		}
	}
	writeJSON(w, status, resp)
}

type scoreRequest struct {
	TransactionID int64 `json:"transaction_id"`
}

type scoreResponse struct {
	TransactionID    int64   `json:"transaction_id"`
	PredictionID     string  `json:"prediction_id"`
	FraudProbability float64 `json:"fraud_probability"`
	Flagged          bool    `json:"flagged"`
	ModelVersion     string  `json:"model_version"`
}

func (h *Handler) Score(w http.ResponseWriter, r *http.Request) {
	totalStart := time.Now()

	var req scoreRequest
	r.Body = http.MaxBytesReader(w, r.Body, 4096)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.TransactionID == 0 {
		http.Error(w, "invalid request body: transaction_id required", http.StatusBadRequest)
		return
	}

	ctx := r.Context()

	lookupStart := time.Now()
	featureMap, err := h.featureClient.Get(ctx, req.TransactionID)
	lookupMs := float64(time.Since(lookupStart).Microseconds()) / 1000.0

	if errors.Is(err, features.ErrNotFound) {
		metrics.FeatureLookupErrors.WithLabelValues("not_found").Inc()
		http.Error(w, "transaction features not found", http.StatusNotFound)
		return
	}
	if err != nil {
		metrics.FeatureLookupErrors.WithLabelValues("redis_error").Inc()
		slog.Error("feature lookup failed", "err", err, "transaction_id", req.TransactionID)
		http.Error(w, "feature lookup failed", http.StatusInternalServerError)
		return
	}

	metrics.FeatureLookupDuration.Observe(lookupMs / 1000.0)

	inferStart := time.Now()
	prob, featHash, err := h.modelRunner.Score(ctx, featureMap)
	inferMs := float64(time.Since(inferStart).Microseconds()) / 1000.0

	if err != nil {
		slog.Error("model inference failed", "err", err, "transaction_id", req.TransactionID)
		http.Error(w, "model inference failed", http.StatusInternalServerError)
		return
	}

	fraudProb := float64(prob)
	// NaN can leak from ORT on adversarial input. Refuse to write it: json.Encode
	// would fail mid-response (truncated body, status 200) and the audit log
	// would carry a non-comparable float into postgres.
	if math.IsNaN(fraudProb) || math.IsInf(fraudProb, 0) {
		slog.Error("model returned non-finite probability", "transaction_id", req.TransactionID, "prob", fraudProb)
		http.Error(w, "model returned non-finite probability", http.StatusInternalServerError)
		return
	}

	metrics.ModelInferenceDuration.Observe(inferMs / 1000.0)

	flagged := fraudProb >= h.threshold
	totalMs := float64(time.Since(totalStart).Microseconds()) / 1000.0

	outcome := "not_flagged"
	if flagged {
		outcome = "flagged"
	}
	metrics.PredictionsTotal.WithLabelValues(outcome).Inc()

	predictionID := uuid.New()
	resp := scoreResponse{
		TransactionID:    req.TransactionID,
		PredictionID:     predictionID.String(),
		FraudProbability: fraudProb,
		Flagged:          flagged,
		ModelVersion:     h.modelRunner.ModelVersion(),
	}
	writeJSON(w, http.StatusOK, resp)

	h.logWg.Add(1)
	go func() {
		defer h.logWg.Done()
		err := h.store.Log(context.Background(), predictions.Record{
			PredictionID:     predictionID,
			TransactionID:    req.TransactionID,
			ModelVersion:     h.modelRunner.ModelVersion(),
			FraudProbability: fraudProb,
			Flagged:          flagged,
			FeaturesHash:     featHash,
			FeatureLookupMs:  lookupMs,
			ModelInferenceMs: inferMs,
			TotalMs:          totalMs,
			CreatedAt:        time.Now(),
		})
		if err != nil {
			slog.Warn("prediction log dropped", "err", err, "prediction_id", predictionID)
		}
	}()
}

type reloadRequest struct {
	ModelPath        string `json:"model_path"`
	FeatureOrderPath string `json:"feature_order_path"`
}

type reloadResponse struct {
	ModelVersion string `json:"model_version"`
}

func (h *Handler) Reload(w http.ResponseWriter, r *http.Request) {
	if subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Admin-Token")), []byte(h.adminToken)) == 0 {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	var req reloadRequest
	r.Body = http.MaxBytesReader(w, r.Body, 4096)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.ModelPath == "" || req.FeatureOrderPath == "" {
		http.Error(w, "model_path and feature_order_path are required", http.StatusBadRequest)
		return
	}

	oldVersion := h.modelRunner.ModelVersion()
	newVersion, err := h.reloader.Reload(req.ModelPath, req.FeatureOrderPath)
	if err != nil {
		slog.Error("model reload failed", "err", err, "model_path", req.ModelPath)
		http.Error(w, "reload failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	slog.Info("model reloaded", "old_version", oldVersion, "new_version", newVersion, "model_path", req.ModelPath)
	writeJSON(w, http.StatusOK, reloadResponse{ModelVersion: newVersion})
}

// Must be called before store.Shutdown() to avoid orphaning buffered log records.
func (h *Handler) Drain() { h.logWg.Wait() }

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		// Status line already sent; can't downgrade to 500. Log so the truncated
		// body shows up somewhere.
		slog.Warn("response body encode failed", "err", err, "status", status)
	}
}
