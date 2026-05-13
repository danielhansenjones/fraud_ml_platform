package metrics

import "github.com/prometheus/client_golang/prometheus"

var (
	HTTPRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{Name: "http_requests_total", Help: "Total HTTP requests"},
		[]string{"path", "status"},
	)
	HTTPRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request duration",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"path"},
	)
	PredictionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{Name: "predictions_total", Help: "Total predictions made"},
		[]string{"outcome"},
	)
	FeatureLookupDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "feature_lookup_duration_seconds",
		Help:    "Redis feature lookup duration",
		Buckets: []float64{.0001, .0005, .001, .005, .01, .05, .1},
	})
	ModelInferenceDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "model_inference_duration_seconds",
		Help:    "ONNX model inference duration",
		Buckets: []float64{.0001, .0005, .001, .005, .01, .05, .1},
	})
	PredictionLogBufferSize = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "prediction_log_buffer_size",
		Help: "Current size of the prediction log buffer",
	})
	PredictionLogFlushErrors = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "prediction_log_flush_errors_total",
		Help: "Total prediction log flush errors",
	})
	PredictionLogDropped = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "prediction_log_dropped_total",
		Help: "Predictions dropped because the log buffer was full",
	})
	FeatureLookupErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "feature_lookup_errors_total",
			Help: "Feature lookup failures by reason",
		},
		[]string{"reason"},
	)
)

func Register(reg *prometheus.Registry) {
	reg.MustRegister(
		HTTPRequestsTotal,
		HTTPRequestDuration,
		PredictionsTotal,
		FeatureLookupDuration,
		ModelInferenceDuration,
		PredictionLogBufferSize,
		PredictionLogFlushErrors,
		PredictionLogDropped,
		FeatureLookupErrors,
	)
}
