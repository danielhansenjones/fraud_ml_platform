package server

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (sw *statusWriter) WriteHeader(code int) {
	sw.status = code
	sw.ResponseWriter.WriteHeader(code)
}

// knownRoutes mirrors the serving package's classifier so log labels stay
// bounded and consistent across services. Keep in sync with the
// mux.HandleFunc calls in cmd/server/main.go and handlers.go.
var knownRoutes = map[string]struct{}{
	"/score":        {},
	"/health":       {},
	"/admin/canary": {},
	"/admin/state":  {},
}

func classifyRoute(path string) string {
	if _, ok := knownRoutes[path]; ok {
		return path
	}
	return "unmatched"
}

func Logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Prometheus self-scrape would otherwise emit a log line per scrape.
		if r.URL.Path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}
		sw := &statusWriter{ResponseWriter: w, status: 200}
		start := time.Now()
		reqID := uuid.New().String()[:8]

		next.ServeHTTP(sw, r)

		slog.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"route", classifyRoute(r.URL.Path),
			"status", sw.status,
			"duration_ms", time.Since(start).Milliseconds(),
			"request_id", reqID,
		)
	})
}

func Recovery(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				slog.Error("panic recovered", "error", rec)
				http.Error(w, "internal server error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}
