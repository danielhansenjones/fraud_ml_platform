CREATE TABLE IF NOT EXISTS drift_alerts (
    alert_id UUID PRIMARY KEY,
    detector_name VARCHAR(64) NOT NULL,
    metric_name VARCHAR(64) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    reference_window_start TIMESTAMPTZ NOT NULL,
    reference_window_end TIMESTAMPTZ NOT NULL,
    severity VARCHAR(16) NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_alerts_created_at ON drift_alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_drift_alerts_severity ON drift_alerts(severity);
