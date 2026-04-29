package test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockRunner and mockFeatureClient are test doubles used in unit tests.

type mockRunner struct {
	version string
	prob    float32
	err     error
}

func (m *mockRunner) Score(_ context.Context, _ map[string]float32) (float32, string, error) {
	return m.prob, "deadbeef", m.err
}

func (m *mockRunner) ModelVersion() string { return m.version }

type mockStore struct{}

func (m *mockStore) Log(_ context.Context, _ interface{}) error { return nil }

func TestHealthReturns200(t *testing.T) {
	h := newTestHandler(&mockRunner{version: "abc123", prob: 0.1}, nil, nil, 0.5)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	h.Health(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]string
	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))
	assert.Equal(t, "ok", resp["status"])
	assert.Equal(t, "abc123", resp["model_version"])
}

func TestScoreMissingTransactionID(t *testing.T) {
	h := newTestHandler(&mockRunner{version: "v1", prob: 0.1}, nil, nil, 0.5)

	body := bytes.NewBufferString(`{}`)
	req := httptest.NewRequest(http.MethodPost, "/score", body)
	w := httptest.NewRecorder()
	h.Score(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestScoreInvalidBody(t *testing.T) {
	h := newTestHandler(&mockRunner{version: "v1", prob: 0.1}, nil, nil, 0.5)

	body := bytes.NewBufferString(`not json`)
	req := httptest.NewRequest(http.MethodPost, "/score", body)
	w := httptest.NewRecorder()
	h.Score(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestScoreUnknownTransactionID(t *testing.T) {
	h := newTestHandler(&mockRunner{version: "v1", prob: 0.1}, &notFoundClient{}, nil, 0.5)

	body := bytes.NewBufferString(`{"transaction_id": 99999}`)
	req := httptest.NewRequest(http.MethodPost, "/score", body)
	w := httptest.NewRecorder()
	h.Score(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
}

func TestScoreValidInput(t *testing.T) {
	runner := &mockRunner{version: "v1", prob: 0.23}
	client := &knownFeatureClient{feats: map[string]float32{"f1": 1.0}}
	h := newTestHandler(runner, client, nil, 0.5)

	body := bytes.NewBufferString(`{"transaction_id": 12345}`)
	req := httptest.NewRequest(http.MethodPost, "/score", body)
	w := httptest.NewRecorder()
	h.Score(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))
	assert.Contains(t, resp, "fraud_probability")
	assert.Contains(t, resp, "flagged")
	assert.Contains(t, resp, "model_version")
}
