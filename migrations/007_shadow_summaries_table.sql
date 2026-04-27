CREATE TABLE IF NOT EXISTS shadow_summaries (
    summary_id UUID PRIMARY KEY,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    n_comparisons BIGINT NOT NULL,
    correlation DOUBLE PRECISION,
    disagreement_rate DOUBLE PRECISION,
    abs_diff_p50 DOUBLE PRECISION,
    abs_diff_p95 DOUBLE PRECISION,
    abs_diff_p99 DOUBLE PRECISION,
    confusion JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shadow_summaries_window ON shadow_summaries(window_start, window_end);
