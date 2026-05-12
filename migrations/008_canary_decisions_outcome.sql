ALTER TABLE canary_decisions
ADD COLUMN IF NOT EXISTS outcome VARCHAR(32) NOT NULL DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS idx_canary_decisions_outcome ON canary_decisions(outcome);

-- 'extend' was declared in 005 but never emitted by the evaluator. Drop it from
-- the CHECK so the column accurately describes what the code can write.
ALTER TABLE canary_decisions DROP CONSTRAINT IF EXISTS canary_decisions_decision_check;
ALTER TABLE canary_decisions
ADD CONSTRAINT canary_decisions_decision_check
CHECK (decision IN ('promote', 'rollback', 'continue'));
