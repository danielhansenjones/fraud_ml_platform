package server

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
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

type PredictionLogger interface {
	Log(ctx context.Context, r predictions.Record) error
}

type Handler struct {
	featureClient FeatureGetter
	modelRunner   ModelScorer
	store         PredictionLogger
	threshold     float64
	logWg         sync.WaitGroup
}

func NewHandler(
	featureClient FeatureGetter,
	modelRunner ModelScorer,
	store PredictionLogger,
	threshold float64,
) *Handler {
	return &Handler{
		featureClient: featureClient,
		modelRunner:   modelRunner,
		store:         store,
		threshold:     threshold,
	}
}

func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":        "ok",
		"model_version": h.modelRunner.ModelVersion(),
	})
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
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.TransactionID == 0 {
		http.Error(w, "invalid request body: transaction_id required", http.StatusBadRequest)
		return
	}

	ctx := r.Context()

	lookupStart := time.Now()
	featureMap, err := h.featureClient.Get(ctx, req.TransactionID)
	lookupMs := float64(time.Since(lookupStart).Microseconds()) / 1000.0

	if errors.Is(err, features.ErrNotFound) {
		http.Error(w, "transaction features not found", http.StatusNotFound)
		return
	}
	if err != nil {
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

	metrics.ModelInferenceDuration.Observe(inferMs / 1000.0)

	fraudProb := float64(prob)
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
		_ = h.store.Log(context.Background(), predictions.Record{
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
	}()
}

// Must be called before store.Shutdown() to avoid orphaning buffered log records.
func (h *Handler) Drain() { h.logWg.Wait() }

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
