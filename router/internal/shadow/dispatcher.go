package shadow

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/silentwraith/fraud_ml_platform/router/internal/metrics"
	"github.com/silentwraith/fraud_ml_platform/router/internal/upstream"
)

type Task struct {
	TransactionID        int64
	ChampionPredictionID string
	ChampionProbability  float64
	ChampionFlagged      bool
	OnComplete           func(resp *upstream.ScoreResponse)
}

type Dispatcher struct {
	challenger *upstream.Client
	ch         chan Task
	wg         sync.WaitGroup
	done       chan struct{}
}

func NewDispatcher(challenger *upstream.Client, bufferSize, workers int) *Dispatcher {
	d := &Dispatcher{
		challenger: challenger,
		ch:         make(chan Task, bufferSize),
		done:       make(chan struct{}),
	}
	for range workers {
		d.wg.Add(1)
		go d.worker()
	}
	return d
}

func (d *Dispatcher) Dispatch(t Task) bool {
	select {
	case d.ch <- t:
		metrics.ShadowDispatchesTotal.WithLabelValues("sent").Inc()
		metrics.ShadowBufferSize.Set(float64(len(d.ch)))
		return true
	default:
		metrics.ShadowDispatchesTotal.WithLabelValues("dropped").Inc()
		return false
	}
}

func (d *Dispatcher) Shutdown() {
	close(d.done)
	d.wg.Wait()
}

func (d *Dispatcher) worker() {
	defer d.wg.Done()
	for {
		select {
		case t, ok := <-d.ch:
			if !ok {
				return
			}
			metrics.ShadowBufferSize.Set(float64(len(d.ch)))
			d.execute(t)
		case <-d.done:
			// Drain buffered tasks before exiting so in-flight shadow calls aren't silently dropped.
			for {
				select {
				case t := <-d.ch:
					d.execute(t)
				default:
					return
				}
			}
		}
	}
}

func (d *Dispatcher) execute(t Task) {
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	resp, err := d.challenger.Score(ctx, upstream.ScoreRequest{TransactionID: t.TransactionID})
	if err != nil {
		slog.Warn("shadow dispatch failed", "err", err)
		return
	}
	if t.OnComplete != nil {
		t.OnComplete(resp)
	}
}
