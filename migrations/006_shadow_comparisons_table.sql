CREATE TABLE IF NOT EXISTS shadow_comparisons (
    transaction_id BIGINT NOT NULL,
    champion_prediction_id UUID NOT NULL,
    challenger_prediction_id UUID NOT NULL,
    champion_probability DOUBLE PRECISION NOT NULL,
    challenger_probability DOUBLE PRECISION NOT NULL,
    champion_flagged BOOLEAN NOT NULL,
    challenger_flagged BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (transaction_id, created_at)
);

CREATE INDEX IF NOT EXISTS idx_shadow_comparisons_created_at ON shadow_comparisons(created_at);
