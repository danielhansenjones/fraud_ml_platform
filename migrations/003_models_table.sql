CREATE TABLE IF NOT EXISTS models (
    model_version VARCHAR(64) PRIMARY KEY,
    model_path VARCHAR(255) NOT NULL,
    onnx_sha256 VARCHAR(64) NOT NULL,
    role VARCHAR(16) NOT NULL CHECK (role IN ('champion', 'challenger', 'retired')),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_models_role ON models(role);
