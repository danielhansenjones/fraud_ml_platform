CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY,
    transaction_id BIGINT NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    fraud_probability DOUBLE PRECISION NOT NULL,
    flagged BOOLEAN NOT NULL,
    features_hash VARCHAR(64) NOT NULL,
    feature_lookup_ms DOUBLE PRECISION NOT NULL,
    model_inference_ms DOUBLE PRECISION NOT NULL,
    total_ms DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
