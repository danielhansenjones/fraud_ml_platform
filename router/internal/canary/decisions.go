package canary

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type ShadowComparison struct {
	TransactionID          int64
	ChampionPredictionID   string
	ChallengerPredictionID string
	ChampionProbability    float64
	ChallengerProbability  float64
	ChampionFlagged        bool
	ChallengerFlagged      bool
}

type Store struct {
	db *pgxpool.Pool
}

func NewStore(db *pgxpool.Pool) *Store {
	return &Store{db: db}
}

func (s *Store) InsertShadowComparison(ctx context.Context, sc ShadowComparison) error {
	if s == nil {
		return nil
	}
	_, err := s.db.Exec(ctx, `
		INSERT INTO shadow_comparisons (
			transaction_id, champion_prediction_id, challenger_prediction_id,
			champion_probability, challenger_probability,
			champion_flagged, challenger_flagged, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`,
		sc.TransactionID,
		sc.ChampionPredictionID,
		sc.ChallengerPredictionID,
		sc.ChampionProbability,
		sc.ChallengerProbability,
		sc.ChampionFlagged,
		sc.ChallengerFlagged,
		time.Now().UTC(),
	)
	return err
}
