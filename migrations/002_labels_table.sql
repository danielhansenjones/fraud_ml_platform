CREATE TABLE IF NOT EXISTS labels (
    transaction_id BIGINT PRIMARY KEY,
    is_fraud BOOLEAN NOT NULL,
    label_source VARCHAR(32) NOT NULL,
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_labels_available_at ON labels(available_at);
