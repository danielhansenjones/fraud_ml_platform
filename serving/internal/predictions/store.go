package predictions

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
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

const (
	defaultRetryBackoff = 200 * time.Millisecond
	drainChunkSize      = 1000
)

type Store struct {
	pool          *pgxpool.Pool
	ch            chan Record
	flushInterval time.Duration
	retryBackoff  time.Duration
	done          chan struct{}
	wg            sync.WaitGroup
	flushFn       func([]Record) error
}

func NewStore(pool *pgxpool.Pool, bufferSize int, flushIntervalMs int) *Store {
	s := &Store{
		pool:          pool,
		ch:            make(chan Record, bufferSize),
		flushInterval: time.Duration(flushIntervalMs) * time.Millisecond,
		retryBackoff:  defaultRetryBackoff,
		done:          make(chan struct{}),
	}
	s.flushFn = s.flushPostgres
	s.wg.Add(1)
	go s.flusher()
	return s
}

// ErrBufferFull is returned by Log when the prediction buffer cannot accept
// the record. The caller should record a metric and continue; the score
// response has already been sent to the client.
var ErrBufferFull = errors.New("prediction log buffer full")

func (s *Store) Log(ctx context.Context, r Record) error {
	metrics.PredictionLogBufferSize.Set(float64(len(s.ch)))
	select {
	case s.ch <- r:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	default:
		// Drop on overflow rather than amplify load by spawning synchronous DB
		// flushes from the request path. Better to lose a few audit rows than
		// exhaust pgxpool under sustained overload.
		metrics.PredictionLogDropped.Inc()
		return ErrBufferFull
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
				s.flushWithRetry(batch)
				batch = batch[:0]
			}
		case <-s.done:
			// Drain in chunks so a large backlog isn't sent as one giant batch.
			for {
				select {
				case r := <-s.ch:
					batch = append(batch, r)
					if len(batch) >= drainChunkSize {
						s.flushWithRetry(batch)
						batch = batch[:0]
					}
				default:
					if len(batch) > 0 {
						s.flushWithRetry(batch)
					}
					return
				}
			}
		}
	}
}

func (s *Store) flushWithRetry(batch []Record) {
	err := s.flushFn(batch)
	if err == nil {
		return
	}
	slog.Warn("prediction flush failed, retrying once", "err", err, "n", len(batch))
	time.Sleep(s.retryBackoff)
	if err := s.flushFn(batch); err != nil {
		metrics.PredictionLogFlushErrors.Inc()
		slog.Error("prediction flush failed after retry; batch dropped", "err", err, "n", len(batch))
	}
}

func (s *Store) flushPostgres(batch []Record) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := s.pool.CopyFrom(
		ctx,
		pgx.Identifier{"predictions"},
		[]string{
			"prediction_id", "transaction_id", "model_version",
			"fraud_probability", "flagged", "features_hash",
			"feature_lookup_ms", "model_inference_ms", "total_ms", "created_at",
		},
		pgx.CopyFromSlice(len(batch), func(i int) ([]any, error) {
			r := batch[i]
			return []any{
				r.PredictionID, r.TransactionID, r.ModelVersion,
				r.FraudProbability, r.Flagged, r.FeaturesHash,
				r.FeatureLookupMs, r.ModelInferenceMs, r.TotalMs, r.CreatedAt,
			}, nil
		}),
	)
	return err
}

// Shutdown signals the flusher to drain and waits for it to complete.
func (s *Store) Shutdown() {
	close(s.done)
	s.wg.Wait()
}