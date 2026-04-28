package routing

import (
	"sync/atomic"
)

type State struct {
	CanaryEnabled            bool
	ChallengerTrafficPercent int
	ShadowPercent            int
}

type AtomicState struct {
	ptr atomic.Pointer[State]
}

func NewAtomicState(initial State) *AtomicState {
	s := &AtomicState{}
	s.ptr.Store(&initial)
	return s
}

func (s *AtomicState) Load() State {
	return *s.ptr.Load()
}

func (s *AtomicState) Store(st State) {
	s.ptr.Store(&st)
}

type RouteDecision string

const (
	DecisionChampion   RouteDecision = "champion"
	DecisionChallenger RouteDecision = "challenger"
)

func Decide(st State, r float64) (live RouteDecision, shadow bool) {
	if st.CanaryEnabled && r*100 < float64(st.ChallengerTrafficPercent) {
		live = DecisionChallenger
	} else {
		live = DecisionChampion
	}

	if !st.CanaryEnabled && st.ShadowPercent > 0 {
		// Under canary both paths already run live, so shadow is only needed when canary is off.
		shadow = r*100 < float64(st.ShadowPercent)
	}

	return live, shadow
}
