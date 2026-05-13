package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	RequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "router_requests_total",
		Help: "Total requests processed by the router.",
	}, []string{"decision", "upstream", "status"})

	RequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "router_request_duration_seconds",
		Help:    "Request latency by upstream.",
		Buckets: prometheus.DefBuckets,
	}, []string{"upstream"})

	ShadowDispatchesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "router_shadow_dispatches_total",
		Help: "Shadow dispatch outcomes.",
	}, []string{"outcome"})

	ShadowBufferSize = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "router_shadow_buffer_size",
		Help: "Current number of pending shadow dispatches in the buffer.",
	})

	CanaryTrafficPercent = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "router_canary_traffic_percent",
		Help: "Current canary challenger traffic percentage.",
	})

	UpstreamErrorsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "router_upstream_errors_total",
		Help: "Upstream errors by service and kind.",
	}, []string{"upstream", "kind"})

	ShadowInsertErrorsTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "router_shadow_insert_errors_total",
		Help: "shadow_comparisons INSERT failures from the OnComplete callback.",
	})
)
