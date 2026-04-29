package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLogging_PassesThrough(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	Logging(inner).ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestRecovery_CatchesPanic(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("test panic")
	})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	Recovery(inner).ServeHTTP(w, req)
	assert.Equal(t, http.StatusInternalServerError, w.Code)
}
