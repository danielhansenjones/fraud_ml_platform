package server

import (
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/metrics"
)

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (sw *statusWriter) WriteHeader(code int) {
	sw.status = code
	sw.ResponseWriter.WriteHeader(code)
}

func Logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Prometheus self-scrape would otherwise emit a log line per scrape and
		// inflate the request_total counter for its own path, hiding real signal.
		if r.URL.Path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}
		sw := &statusWriter{ResponseWriter: w, status: 200}
		start := time.Now()
		reqID := uuid.New().String()[:8]

		next.ServeHTTP(sw, r)

		// Bound metric cardinality: only the routes we register get their own
		// label; everything else collapses into "unmatched".
		label := classifyRoute(r.URL.Path)

		slog.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"route", label,
			"status", sw.status,
			"duration_ms", time.Since(start).Milliseconds(),
			"request_id", reqID,
		)

		statusStr := strconv.Itoa(sw.status)
		metrics.HTTPRequestsTotal.With(prometheus.Labels{"path": label, "status": statusStr}).Inc()
		metrics.HTTPRequestDuration.With(prometheus.Labels{"path": label}).Observe(time.Since(start).Seconds())
	})
}

// knownRoutes must stay in sync with the mux.HandleFunc calls in
// cmd/server/main.go. Anything not listed here gets bucketed as "unmatched"
// in HTTP request metrics.
var knownRoutes = map[string]struct{}{
	"/health":       {},
	"/score":        {},
	"/admin/reload": {},
}

func classifyRoute(path string) string {
	if _, ok := knownRoutes[path]; ok {
		return path
	}
	return "unmatched"
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
