package server

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"strconv"
	"time"

	"github.com/silentwraith/fraud_ml_platform/router/internal/canary"
	"github.com/silentwraith/fraud_ml_platform/router/internal/metrics"
	"github.com/silentwraith/fraud_ml_platform/router/internal/routing"
	"github.com/silentwraith/fraud_ml_platform/router/internal/shadow"
	"github.com/silentwraith/fraud_ml_platform/router/internal/upstream"
)

type Handler struct {
	state      *routing.AtomicState
	champion   *upstream.Client
	challenger *upstream.Client
	dispatcher *shadow.Dispatcher
	store      *canary.Store
	adminToken string
}

func NewHandler(
	state *routing.AtomicState,
	champion, challenger *upstream.Client,
	dispatcher *shadow.Dispatcher,
	store *canary.Store,
	adminToken string,
) *Handler {
	return &Handler{
		state:      state,
		champion:   champion,
		challenger: challenger,
		dispatcher: dispatcher,
		store:      store,
		adminToken: adminToken,
	}
}

func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /score", h.handleScore)
	mux.HandleFunc("GET /health", h.handleHealth)
	mux.HandleFunc("POST /admin/canary", h.handleAdminCanary)
	mux.HandleFunc("GET /admin/state", h.handleAdminState)
}

type scoreRequest struct {
	TransactionID int64 `json:"transaction_id"`
}

func (h *Handler) handleScore(w http.ResponseWriter, r *http.Request) {
	var req scoreRequest
	r.Body = http.MaxBytesReader(w, r.Body, 4096)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.TransactionID == 0 {
		http.Error(w, "invalid request body: transaction_id required", http.StatusBadRequest)
		return
	}

	st := h.state.Load()
	rnd := rand.Float64()
	live, doShadow := routing.Decide(st, rnd)

	var liveClient *upstream.Client
	if live == routing.DecisionChallenger {
		liveClient = h.challenger
	} else {
		liveClient = h.champion
	}

	start := time.Now()
	ctx, cancel := context.WithTimeout(r.Context(), 500*time.Millisecond)
	defer cancel()

	resp, err := liveClient.Score(ctx, upstream.ScoreRequest{TransactionID: req.TransactionID})
	elapsed := time.Since(start).Seconds()

	upstreamName := string(live)
	metrics.RequestDuration.WithLabelValues(upstreamName).Observe(elapsed)

	if err != nil {
		ue, _ := err.(*upstream.UpstreamError)
		kind := "unknown"
		if ue != nil {
			kind = string(ue.Kind)
		}
		metrics.UpstreamErrorsTotal.WithLabelValues(upstreamName, kind).Inc()
		metrics.RequestsTotal.WithLabelValues(string(live), upstreamName, "error").Inc()
		http.Error(w, "upstream error", http.StatusServiceUnavailable)
		return
	}

	metrics.RequestsTotal.WithLabelValues(string(live), upstreamName, "200").Inc()

	w.Header().Set("X-Model-Version", resp.ModelVersion)
	w.Header().Set("X-Routing-Decision", string(live))
	w.Header().Set("Content-Type", "application/json")

	if doShadow {
		championResp := resp
		h.dispatcher.Dispatch(shadow.Task{
			TransactionID:        req.TransactionID,
			ChampionPredictionID: championResp.PredictionID,
			ChampionProbability:  championResp.FraudProbability,
			ChampionFlagged:      championResp.Flagged,
			OnComplete: func(challengerResp *upstream.ScoreResponse) {
				ctx2, cancel2 := context.WithTimeout(context.Background(), 2*time.Second)
				defer cancel2()
				err := h.store.InsertShadowComparison(ctx2, canary.ShadowComparison{
					TransactionID:          req.TransactionID,
					ChampionPredictionID:   championResp.PredictionID,
					ChallengerPredictionID: challengerResp.PredictionID,
					ChampionProbability:    championResp.FraudProbability,
					ChallengerProbability:  challengerResp.FraudProbability,
					ChampionFlagged:        championResp.Flagged,
					ChallengerFlagged:      challengerResp.Flagged,
				})
				if err != nil {
					metrics.ShadowInsertErrorsTotal.Inc()
					slog.Warn("shadow comparison insert failed", "err", err)
				}
			},
		})
	}

	json.NewEncoder(w).Encode(resp)
}

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 200*time.Millisecond)
	defer cancel()

	if err := h.champion.Health(ctx); err != nil {
		http.Error(w, fmt.Sprintf("champion unhealthy: %v", err), http.StatusServiceUnavailable)
		return
	}
	// If the challenger is taking live canary traffic, an unreachable challenger
	// is a user-visible problem and must fail readiness. Shadow-only failures
	// degrade comparison data, not user responses, so they are not gated here.
	st := h.state.Load()
	if st.CanaryEnabled && st.ChallengerTrafficPercent > 0 {
		if err := h.challenger.Health(ctx); err != nil {
			http.Error(w, fmt.Sprintf("challenger unhealthy: %v", err), http.StatusServiceUnavailable)
			return
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, `{"status":"ok"}`)
}

type adminCanaryRequest struct {
	Enabled                  bool `json:"enabled"`
	ChallengerTrafficPercent int  `json:"challenger_traffic_percent"`
	ShadowPercent            int  `json:"shadow_percent"`
}

func (h *Handler) handleAdminCanary(w http.ResponseWriter, r *http.Request) {
	if subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Admin-Token")), []byte(h.adminToken)) == 0 {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	var req adminCanaryRequest
	r.Body = http.MaxBytesReader(w, r.Body, 4096)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.ChallengerTrafficPercent < 0 || req.ChallengerTrafficPercent > 100 {
		http.Error(w, "challenger_traffic_percent must be 0-100", http.StatusBadRequest)
		return
	}
	if req.ShadowPercent < 0 || req.ShadowPercent > 100 {
		http.Error(w, "shadow_percent must be 0-100", http.StatusBadRequest)
		return
	}
	// Shadow only runs when canary is off (see routing.Decide). Reject the
	// combination at the API boundary so callers cannot set a value that the
	// router silently ignores.
	if req.Enabled && req.ShadowPercent > 0 {
		http.Error(w, "shadow_percent must be 0 when canary is enabled", http.StatusBadRequest)
		return
	}

	h.state.Store(routing.State{
		CanaryEnabled:            req.Enabled,
		ChallengerTrafficPercent: req.ChallengerTrafficPercent,
		ShadowPercent:            req.ShadowPercent,
	})

	metrics.CanaryTrafficPercent.Set(float64(req.ChallengerTrafficPercent))

	slog.Info("canary state updated",
		"enabled", req.Enabled,
		"challenger_pct", req.ChallengerTrafficPercent,
		"shadow_pct", req.ShadowPercent,
	)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(req)
}

func (h *Handler) handleAdminState(w http.ResponseWriter, r *http.Request) {
	st := h.state.Load()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"canary_enabled":             strconv.FormatBool(st.CanaryEnabled),
		"challenger_traffic_percent": strconv.Itoa(st.ChallengerTrafficPercent),
		"shadow_percent":             strconv.Itoa(st.ShadowPercent),
	})
}
