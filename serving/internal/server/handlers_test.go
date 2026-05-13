package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
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

type stubReloader struct {
	called   bool
	gotModel string
	gotOrder string
	newVer   string
	err      error
}

func (s *stubReloader) Reload(modelPath, featureOrderPath string) (string, error) {
	s.called = true
	s.gotModel = modelPath
	s.gotOrder = featureOrderPath
	return s.newVer, s.err
}

type stubPinger struct{ err error }

func (s *stubPinger) Ping(_ context.Context) error { return s.err }

func okPinger() *stubPinger { return &stubPinger{} }

// Tests

func TestHealth_OK(t *testing.T) {
	h := NewHandler(&stubGetter{}, &stubScorer{version: "abc123"}, &stubReloader{}, &recordingLogger{logged: make(chan predictions.Record, 1)}, okPinger(), okPinger(), 0.5, "tok")
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	h.Health(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]string
	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))
	assert.Equal(t, "ok", resp["status"])
	assert.Equal(t, "abc123", resp["model_version"])
	assert.Equal(t, "ok", resp["redis"])
	assert.Equal(t, "ok", resp["postgres"])
}

func TestHealth_RedisDown(t *testing.T) {
	h := NewHandler(&stubGetter{}, &stubScorer{version: "v1"}, &stubReloader{}, &recordingLogger{logged: make(chan predictions.Record, 1)},
		&stubPinger{err: fmt.Errorf("redis timeout")}, okPinger(), 0.5, "tok")
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	h.Health(w, req)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	var resp map[string]string
	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))
	assert.Equal(t, "unhealthy", resp["status"])
	assert.Contains(t, resp["redis"], "redis timeout")
}

func TestHealth_PostgresDown(t *testing.T) {
	h := NewHandler(&stubGetter{}, &stubScorer{version: "v1"}, &stubReloader{}, &recordingLogger{logged: make(chan predictions.Record, 1)},
		okPinger(), &stubPinger{err: fmt.Errorf("connection refused")}, 0.5, "tok")
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	h.Health(w, req)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
}

func TestScore_MissingTransactionID(t *testing.T) {
	h := NewHandler(&stubGetter{}, &stubScorer{version: "v1"}, &stubReloader{}, &recordingLogger{logged: make(chan predictions.Record, 1)}, okPinger(), okPinger(), 0.5, "tok")
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`{}`))
	w := httptest.NewRecorder()
	h.Score(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestScore_InvalidBody(t *testing.T) {
	h := NewHandler(&stubGetter{}, &stubScorer{version: "v1"}, &stubReloader{}, &recordingLogger{logged: make(chan predictions.Record, 1)}, okPinger(), okPinger(), 0.5, "tok")
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`not json`))
	w := httptest.NewRecorder()
	h.Score(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestScore_FeatureNotFound(t *testing.T) {
	h := NewHandler(
		&stubGetter{err: features.ErrNotFound},
		&stubScorer{version: "v1"},
		&stubReloader{},
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		okPinger(), okPinger(),
		0.5,
		"tok",
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
		&stubReloader{},
		logger,
		okPinger(), okPinger(),
		0.5,
		"tok",
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
		&stubReloader{},
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		okPinger(), okPinger(),
		0.5,
		"tok",
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
		&stubReloader{},
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		okPinger(), okPinger(),
		0.5,
		"tok",
	)
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`{"transaction_id": 42}`))
	w := httptest.NewRecorder()
	h.Score(w, req)
	assert.Equal(t, http.StatusInternalServerError, w.Code)
}

func TestScore_NaNProbReturns500(t *testing.T) {
	h := NewHandler(
		&stubGetter{feats: map[string]float32{"f1": 1.0}},
		&stubScorer{version: "v1", prob: float32(math.NaN())},
		&stubReloader{},
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		okPinger(), okPinger(),
		0.5,
		"tok",
	)
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewBufferString(`{"transaction_id": 42}`))
	w := httptest.NewRecorder()
	h.Score(w, req)
	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, w.Body.String(), "non-finite")
}

func TestReload_RejectsWithoutToken(t *testing.T) {
	rel := &stubReloader{newVer: "newversion"}
	h := NewHandler(
		&stubGetter{}, &stubScorer{version: "old"}, rel,
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		okPinger(), okPinger(), 0.5, "secret",
	)
	body := `{"model_path":"/m.onnx","feature_order_path":"/f.json"}`
	req := httptest.NewRequest(http.MethodPost, "/admin/reload", bytes.NewBufferString(body))
	w := httptest.NewRecorder()
	h.Reload(w, req)
	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.False(t, rel.called)
}

func TestReload_AcceptsCorrectToken(t *testing.T) {
	rel := &stubReloader{newVer: "newversion"}
	h := NewHandler(
		&stubGetter{}, &stubScorer{version: "old"}, rel,
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		okPinger(), okPinger(), 0.5, "secret",
	)
	body := `{"model_path":"/m.onnx","feature_order_path":"/f.json"}`
	req := httptest.NewRequest(http.MethodPost, "/admin/reload", bytes.NewBufferString(body))
	req.Header.Set("X-Admin-Token", "secret")
	w := httptest.NewRecorder()
	h.Reload(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.True(t, rel.called)
	assert.Equal(t, "/m.onnx", rel.gotModel)
	assert.Equal(t, "/f.json", rel.gotOrder)
	var resp map[string]string
	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))
	assert.Equal(t, "newversion", resp["model_version"])
}

func TestReload_MissingFields(t *testing.T) {
	rel := &stubReloader{}
	h := NewHandler(
		&stubGetter{}, &stubScorer{version: "old"}, rel,
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		okPinger(), okPinger(), 0.5, "secret",
	)
	req := httptest.NewRequest(http.MethodPost, "/admin/reload", bytes.NewBufferString(`{}`))
	req.Header.Set("X-Admin-Token", "secret")
	w := httptest.NewRecorder()
	h.Reload(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.False(t, rel.called)
}

func TestReload_PropagatesReloaderError(t *testing.T) {
	rel := &stubReloader{err: fmt.Errorf("model file not found")}
	h := NewHandler(
		&stubGetter{}, &stubScorer{version: "old"}, rel,
		&recordingLogger{logged: make(chan predictions.Record, 1)},
		okPinger(), okPinger(), 0.5, "secret",
	)
	body := `{"model_path":"/missing.onnx","feature_order_path":"/f.json"}`
	req := httptest.NewRequest(http.MethodPost, "/admin/reload", bytes.NewBufferString(body))
	req.Header.Set("X-Admin-Token", "secret")
	w := httptest.NewRecorder()
	h.Reload(w, req)
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
		&stubReloader{},
		logger,
		okPinger(), okPinger(),
		0.5,
		"tok",
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
