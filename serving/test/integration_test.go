package test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Integration test: seeds miniredis with a known feature vector, calls /score,
// asserts the response shape and that the prediction probability is numeric.
// Postgres writes are tested via the store unit - full DB integration requires
// testcontainers, which is out of scope for CI without Docker.

func TestScoreIntegration(t *testing.T) {
	mr, err := miniredis.Run()
	require.NoError(t, err)
	defer mr.Close()

	// Seed a feature vector
	featureVec := map[string]interface{}{
		"f1": 1.5,
		"f2": -0.3,
		"f3": 0.0,
	}
	featureJSON, _ := json.Marshal(featureVec)
	mr.Set("fraud:features:42", string(featureJSON))

	featClient := newTestRedisClient(mr.Addr())
	runner := &mockRunner{version: "testv1", prob: 0.87}
	h := newTestHandler(runner, featClient, nil, 0.5)

	body := bytes.NewBufferString(`{"transaction_id": 42}`)
	req := httptest.NewRequest(http.MethodPost, "/score", body)
	w := httptest.NewRecorder()
	h.Score(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var resp map[string]interface{}
	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))

	prob, ok := resp["fraud_probability"].(float64)
	require.True(t, ok, "fraud_probability should be a float")
	assert.InDelta(t, 0.87, prob, 1e-5)
	assert.Equal(t, true, resp["flagged"])
	assert.Equal(t, "testv1", resp["model_version"])
}

func TestScoreIntegration_NotFound(t *testing.T) {
	mr, err := miniredis.Run()
	require.NoError(t, err)
	defer mr.Close()

	featClient := newTestRedisClient(mr.Addr())
	runner := &mockRunner{version: "testv1", prob: 0.1}
	h := newTestHandler(runner, featClient, nil, 0.5)

	body := bytes.NewBufferString(`{"transaction_id": 99999}`)
	req := httptest.NewRequest(http.MethodPost, "/score", body)
	w := httptest.NewRecorder()
	h.Score(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
}
