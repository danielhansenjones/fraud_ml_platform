package router_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/silentwraith/fraud_ml_platform/router/internal/canary"
	"github.com/silentwraith/fraud_ml_platform/router/internal/routing"
	"github.com/silentwraith/fraud_ml_platform/router/internal/server"
	"github.com/silentwraith/fraud_ml_platform/router/internal/shadow"
	"github.com/silentwraith/fraud_ml_platform/router/internal/upstream"
)

const testAdminToken = "test-token"

func mockModelServer(t *testing.T, version string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		var req struct {
			TransactionID int64 `json:"transaction_id"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"transaction_id":    req.TransactionID,
			"prediction_id":     "00000000-0000-0000-0000-000000000001",
			"fraud_probability": 0.12,
			"flagged":           false,
			"model_version":     version,
			"latency_ms":        1.0,
		})
	}))
}

func newTestHandler(t *testing.T, state routing.State) (*server.Handler, *httptest.Server, *httptest.Server) {
	t.Helper()
	champSrv := mockModelServer(t, "champion-v1")
	challSrv := mockModelServer(t, "challenger-v1")
	t.Cleanup(func() { champSrv.Close(); challSrv.Close() })

	champ := upstream.New("champion", champSrv.URL, 500)
	chall := upstream.New("challenger", challSrv.URL, 500)
	st := routing.NewAtomicState(state)
	disp := shadow.NewDispatcher(chall, 64, 2)
	t.Cleanup(func() { disp.Shutdown() })

	// nil store skips shadow comparison DB writes without needing a live Postgres connection.
	h := server.NewHandler(st, champ, chall, disp, (*canary.Store)(nil), testAdminToken)
	return h, champSrv, challSrv
}

func scoreRequest(t *testing.T, h *server.Handler, txID int64) *httptest.ResponseRecorder {
	t.Helper()
	body, _ := json.Marshal(map[string]int64{"transaction_id": txID})
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)
	mux.ServeHTTP(w, req)
	return w
}

func TestRouting_ChampionOnly_WhenCanaryDisabled(t *testing.T) {
	h, _, _ := newTestHandler(t, routing.State{CanaryEnabled: false, ShadowPercent: 0})
	w := scoreRequest(t, h, 1001)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	decision := w.Header().Get("X-Routing-Decision")
	if decision != "champion" {
		t.Errorf("expected champion routing, got %q", decision)
	}
}

func TestRouting_CanarySplit_StatisticallyCorrect(t *testing.T) {
	h, _, _ := newTestHandler(t, routing.State{CanaryEnabled: true, ChallengerTrafficPercent: 30, ShadowPercent: 0})

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	var champion, challenger int64
	const n = 10_000
	for i := range n {
		body, _ := json.Marshal(map[string]int64{"transaction_id": int64(i)})
		req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		switch w.Header().Get("X-Routing-Decision") {
		case "champion":
			atomic.AddInt64(&champion, 1)
		case "challenger":
			atomic.AddInt64(&challenger, 1)
		}
	}

	pct := float64(challenger) / n * 100
	if pct < 28 || pct > 32 {
		t.Errorf("expected ~30%% challenger, got %.1f%%", pct)
	}
}

func TestRouting_ShadowDrops_DoNotBlockForeground(t *testing.T) {
	h, _, challSrv := newTestHandler(t, routing.State{CanaryEnabled: false, ShadowPercent: 100})
	_ = challSrv

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	const n = 200
	start := time.Now()
	for i := range n {
		body, _ := json.Marshal(map[string]int64{"transaction_id": int64(i)})
		req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("request %d: expected 200, got %d", i, w.Code)
		}
	}
	elapsed := time.Since(start)
	if elapsed > 5*time.Second {
		t.Errorf("foreground requests took too long: %v", elapsed)
	}
}

func TestAdmin_RejectsWithoutToken(t *testing.T) {
	h, _, _ := newTestHandler(t, routing.State{})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{"enabled": true, "challenger_traffic_percent": 50, "shadow_percent": 100})
	req := httptest.NewRequest(http.MethodPost, "/admin/canary", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", w.Code)
	}
}

func TestAdmin_AcceptsCorrectToken(t *testing.T) {
	h, _, _ := newTestHandler(t, routing.State{})
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]any{"enabled": true, "challenger_traffic_percent": 20, "shadow_percent": 50})
	req := httptest.NewRequest(http.MethodPost, "/admin/canary", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Admin-Token", testAdminToken)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d; body: %s", w.Code, w.Body.String())
	}
}

func TestUpstreamTimeout_Returns503(t *testing.T) {
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer slow.Close()

	chall := upstream.New("challenger", slow.URL, 500)
	champ := upstream.New("champion", slow.URL, 50) // 50ms timeout
	st := routing.NewAtomicState(routing.State{CanaryEnabled: false, ShadowPercent: 0})
	disp := shadow.NewDispatcher(chall, 64, 2)
	defer disp.Shutdown()

	h := server.NewHandler(st, champ, chall, disp, (*canary.Store)(nil), testAdminToken)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]int64{"transaction_id": 1})
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503, got %d", w.Code)
	}
}

func TestDecide_ChampionOnly(t *testing.T) {
	st := routing.State{CanaryEnabled: false, ShadowPercent: 0}
	for i := range 100 {
		live, _ := routing.Decide(st, float64(i)/100.0)
		if live != routing.DecisionChampion {
			t.Fatalf("expected champion, got %s at r=%.2f", live, float64(i)/100.0)
		}
	}
}

func TestDecide_CanaySplit(t *testing.T) {
	st := routing.State{CanaryEnabled: true, ChallengerTrafficPercent: 50}
	champCount, challCount := 0, 0
	for i := range 1000 {
		live, _ := routing.Decide(st, float64(i)/1000.0)
		if live == routing.DecisionChampion {
			champCount++
		} else {
			challCount++
		}
	}
	if champCount != challCount {
		t.Errorf("expected 50/50 split, got champ=%d chall=%d", champCount, challCount)
	}
}

func TestDecide_ShadowFiredWhenCanaryOff(t *testing.T) {
	st := routing.State{CanaryEnabled: false, ShadowPercent: 100}
	for i := range 100 {
		_, doShadow := routing.Decide(st, float64(i)/100.0)
		if !doShadow {
			t.Errorf("expected shadow at r=%.2f, got false", float64(i)/100.0)
		}
	}
}

func TestScore_ContextCancelled(t *testing.T) {
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(500 * time.Millisecond):
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer slow.Close()

	champ := upstream.New("champion", slow.URL, 500)
	chall := upstream.New("challenger", slow.URL, 500)
	st := routing.NewAtomicState(routing.State{CanaryEnabled: false, ShadowPercent: 0})
	disp := shadow.NewDispatcher(chall, 64, 2)
	defer disp.Shutdown()

	h := server.NewHandler(st, champ, chall, disp, (*canary.Store)(nil), testAdminToken)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body, _ := json.Marshal(map[string]int64{"transaction_id": 1})
	req := httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	ctx, cancel := context.WithTimeout(req.Context(), 50*time.Millisecond)
	defer cancel()
	req = req.WithContext(ctx)

	w := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		mux.ServeHTTP(w, req)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handler did not return after context cancellation")
	}
}

var _ = fmt.Sprintf
