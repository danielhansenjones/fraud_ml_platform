package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/silentwraith/fraud_ml_platform/serving/internal/features"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/predictions"
)

// Test doubles

type stubGetter struct {
	feats map[string]float32
	err   error
}

func (s *stubGetter) Get(_ context.Context, _ int64) (map[string]float32, error) {
	return s.feats, s.err
}

type stubScorer struct {
	version string
	prob    float32
	err     error
}

func (s *stubScorer) Score(_ context.Context, _ map[string]float32) (float32, string, error) {
	return s.prob, "deadbeef", s.err
}

func (s *stubScorer) ModelVersion() string { return s.version }

type recordingLogger struct {
	logged chan predictions.Record
}

func (l *recordingLogger) Log(_ context.Context, r predictions.Record) error {
	select {
	case l.logged <- r:
	default:
	}
	return nil
}

type blockingLogger struct {
	unblock chan struct{}
	called  chan struct{}
}

func (l *blockingLogger) Log(_ context.Context, _ predictions.Record) error {
	close(l.called)
	<-l.unblock
	return nil
}

// Tests

func TestHealth_OK(t *testing.T) {
	h := NewHandler(&stubGetter{}, &stubScorer{version: "abc123"}, &recordingLogger{logged: make(chan predictions.Record, 1)}, 0.5)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	h.Health(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]string
	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))
	assert.Equal(t, "ok", resp["status"])
	assert.Equal(t, "abc123", resp["model_version"])
}

func TestScore_MissingTransactionID(t *testing.T) {
	h := NewHandler(&stubGetter{}, &stubScorer{version: "v1"}, &recordingLogger{logged: make(chan predictions.Record, 1)}, 0.5)
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`{}`))
	w := httptest.NewRecorder()
	h.Score(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestScore_InvalidBody(t *testing.T) {
	h := NewHandler(&stubGetter{}, &stubScorer{version: "v1"}, &recordingLogger{logged: make(chan predictions.Record, 1)}, 0.5)
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`not json`))
	w := httptest.NewRecorder()
	h.Score(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestScore_FeatureNotFound(t *testing.T) {
	h := NewHandler(
		&stubGetter{err: features.ErrNotFound},
		&stubScorer{version: "v1"},
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		0.5,
	)
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`{"transaction_id": 99}`))
	w := httptest.NewRecorder()
	h.Score(w, req)
	assert.Equal(t, http.StatusNotFound, w.Code)
}

func TestScore_ValidRequest(t *testing.T) {
	logger := &recordingLogger{logged: make(chan predictions.Record, 1)}
	h := NewHandler(
		&stubGetter{feats: map[string]float32{"f1": 1.0}},
		&stubScorer{version: "v1", prob: 0.8},
		logger,
		0.5,
	)
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`{"transaction_id": 42}`))
	w := httptest.NewRecorder()
	h.Score(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))
	assert.InDelta(t, 0.8, resp["fraud_probability"], 1e-5)
	assert.Equal(t, true, resp["flagged"])
	assert.Equal(t, "v1", resp["model_version"])
}

func TestScore_FeatureLookupError(t *testing.T) {
	h := NewHandler(
		&stubGetter{err: fmt.Errorf("redis timeout")},
		&stubScorer{version: "v1"},
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		0.5,
	)
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`{"transaction_id": 42}`))
	w := httptest.NewRecorder()
	h.Score(w, req)
	assert.Equal(t, http.StatusInternalServerError, w.Code)
}

func TestScore_InferenceError(t *testing.T) {
	h := NewHandler(
		&stubGetter{feats: map[string]float32{"f1": 1.0}},
		&stubScorer{err: fmt.Errorf("onnx session failed")},
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		0.5,
	)
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`{"transaction_id": 42}`))
	w := httptest.NewRecorder()
	h.Score(w, req)
	assert.Equal(t, http.StatusInternalServerError, w.Code)
}

func TestScore_DoesNotBlockOnLog(t *testing.T) {
	logger := &blockingLogger{
		unblock: make(chan struct{}),
		called:  make(chan struct{}),
	}
	h := NewHandler(
		&stubGetter{feats: map[string]float32{"f1": 1.0}},
		&stubScorer{version: "v1", prob: 0.1},
		logger,
		0.5,
	)
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`{"transaction_id": 42}`))
	w := httptest.NewRecorder()

	start := time.Now()
	h.Score(w, req)
	elapsed := time.Since(start)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Less(t, elapsed, 50*time.Millisecond, "Score must return before the blocking logger finishes")

	close(logger.unblock)
	select {
	case <-logger.called:
	case <-time.After(time.Second):
		t.Error("logger goroutine never ran")
	}
}
