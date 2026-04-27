CREATE TABLE IF NOT EXISTS canary_decisions (
    decision_id UUID PRIMARY KEY,
    champion_version VARCHAR(64) NOT NULL,
    challenger_version VARCHAR(64) NOT NULL,
    decision VARCHAR(16) NOT NULL CHECK (decision IN ('promote', 'rollback', 'continue', 'extend')),
    reason TEXT NOT NULL,
    metrics JSONB NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_canary_decisions_created_at ON canary_decisions(created_at);
