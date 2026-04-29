package test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/silentwraith/fraud_ml_platform/serving/internal/features"
)

type notFoundClient struct{}

func (c *notFoundClient) Get(_ context.Context, _ int64) (map[string]float32, error) {
	return nil, features.ErrNotFound
}

type knownFeatureClient struct {
	feats map[string]float32
}

func (c *knownFeatureClient) Get(_ context.Context, _ int64) (map[string]float32, error) {
	return c.feats, nil
}

// testableHandler wires test doubles for unit-level handler testing.
type testableHandler struct {
	runner *mockRunner
	client interface {
		Get(ctx context.Context, transactionID int64) (map[string]float32, error)
	}
	threshold float64
}

func newTestHandler(runner *mockRunner, client interface {
	Get(ctx context.Context, transactionID int64) (map[string]float32, error)
}, _ interface{}, threshold float64) *testableHandler {
	return &testableHandler{runner: runner, client: client, threshold: threshold}
}

func (h *testableHandler) Health(w http.ResponseWriter, r *http.Request) {
	encodeJSON(w, 200, map[string]string{
		"status":        "ok",
		"model_version": h.runner.ModelVersion(),
	})
}

func (h *testableHandler) Score(w http.ResponseWriter, r *http.Request) {
	var req struct {
		TransactionID int64 `json:"transaction_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.TransactionID == 0 {
		http.Error(w, "invalid request body: transaction_id required", http.StatusBadRequest)
		return
	}
	if h.client == nil {
		http.Error(w, "no client configured", http.StatusInternalServerError)
		return
	}
	featureMap, err := h.client.Get(r.Context(), req.TransactionID)
	if errors.Is(err, features.ErrNotFound) {
		http.Error(w, "transaction features not found", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, "feature lookup failed", http.StatusInternalServerError)
		return
	}
	prob, _, err := h.runner.Score(r.Context(), featureMap)
	if err != nil {
		http.Error(w, "inference failed", http.StatusInternalServerError)
		return
	}
	flagged := float64(prob) >= h.threshold
	encodeJSON(w, 200, map[string]interface{}{
		"transaction_id":    req.TransactionID,
		"fraud_probability": float64(prob),
		"flagged":           flagged,
		"model_version":     h.runner.ModelVersion(),
	})
}

func encodeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
