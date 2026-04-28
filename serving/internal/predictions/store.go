package predictions

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/metrics"
)

type Record struct {
	PredictionID     uuid.UUID
	TransactionID    int64
	ModelVersion     string
	FraudProbability float64
	Flagged          bool
	FeaturesHash     string
	FeatureLookupMs  float64
	ModelInferenceMs float64
	TotalMs          float64
	CreatedAt        time.Time
}

type Store struct {
	pool          *pgxpool.Pool
	ch            chan Record
	flushInterval time.Duration
	done          chan struct{}
	wg            sync.WaitGroup
	flushFn       func([]Record)
}

func NewStore(pool *pgxpool.Pool, bufferSize int, flushIntervalMs int) *Store {
	s := &Store{
		pool:          pool,
		ch:            make(chan Record, bufferSize),
		flushInterval: time.Duration(flushIntervalMs) * time.Millisecond,
		done:          make(chan struct{}),
	}
	s.flushFn = s.flushPostgres
	s.wg.Add(1)
	go s.flusher()
	return s
}

func (s *Store) Log(ctx context.Context, r Record) error {
	metrics.PredictionLogBufferSize.Set(float64(len(s.ch)))
	select {
	case s.ch <- r:
		return nil
	default:
		// Buffer full; flush synchronously to avoid losing data
		s.flushFn([]Record{r})
		return nil
	}
}

func (s *Store) flusher() {
	defer s.wg.Done()
	ticker := time.NewTicker(s.flushInterval)
	defer ticker.Stop()

	var batch []Record
	for {
		select {
		case r := <-s.ch:
			batch = append(batch, r)
		case <-ticker.C:
			if len(batch) > 0 {
				s.flushFn(batch)
				batch = batch[:0]
			}
		case <-s.done:
			// Drain remaining
			for {
				select {
				case r := <-s.ch:
					batch = append(batch, r)
				default:
					if len(batch) > 0 {
						s.flushFn(batch)
					}
					return
				}
			}
		}
	}
}

func (s *Store) flushPostgres(batch []Record) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		slog.Error("prediction flush: begin tx", "err", err)
		metrics.PredictionLogFlushErrors.Inc()
		return
	}
	defer tx.Rollback(ctx)

	const q = `INSERT INTO predictions
		(prediction_id, transaction_id, model_version, fraud_probability, flagged,
		 features_hash, feature_lookup_ms, model_inference_ms, total_ms, created_at)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`

	for _, r := range batch {
		if _, err := tx.Exec(ctx, q,
			r.PredictionID, r.TransactionID, r.ModelVersion, r.FraudProbability,
			r.Flagged, r.FeaturesHash, r.FeatureLookupMs, r.ModelInferenceMs,
			r.TotalMs, r.CreatedAt,
		); err != nil {
			slog.Error("prediction flush: insert", "err", err)
			metrics.PredictionLogFlushErrors.Inc()
			return
		}
	}

	if err := tx.Commit(ctx); err != nil {
		slog.Error("prediction flush: commit", "err", err)
		metrics.PredictionLogFlushErrors.Inc()
	}
}

// Shutdown signals the flusher to drain and waits for it to complete.
func (s *Store) Shutdown() {
	close(s.done)
	s.wg.Wait()
}
