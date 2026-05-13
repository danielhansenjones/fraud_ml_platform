package predictions

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestStore(bufferSize int, captured *[]Record, mu *sync.Mutex) *Store {
	s := &Store{
		ch:            make(chan Record, bufferSize),
		flushInterval: time.Hour,
		retryBackoff:  0,
		done:          make(chan struct{}),
	}
	s.flushFn = func(batch []Record) error {
		mu.Lock()
		*captured = append(*captured, batch...)
		mu.Unlock()
		return nil
	}
	s.wg.Add(1)
	go s.flusher()
	return s
}

func TestLog_NonBlocking(t *testing.T) {
	var captured []Record
	var mu sync.Mutex
	s := newTestStore(100, &captured, &mu)
	defer s.Shutdown()

	start := time.Now()
	err := s.Log(context.Background(), Record{TransactionID: 1})
	elapsed := time.Since(start)

	require.NoError(t, err)
	assert.Less(t, elapsed, 10*time.Millisecond)
}

func TestLog_BufferOverflow_DropsAndReportsError(t *testing.T) {
	var captured []Record
	var mu sync.Mutex
	// Buffer of 1, no time-based flush in this test interval. Once the channel
	// holds one record, the next Log must drop and surface ErrBufferFull
	// instead of synchronously flushing (which would amplify load).
	s := newTestStore(1, &captured, &mu)
	defer s.Shutdown()

	require.NoError(t, s.Log(context.Background(), Record{TransactionID: 1}))
	err := s.Log(context.Background(), Record{TransactionID: 2})
	assert.ErrorIs(t, err, ErrBufferFull)
}

func TestShutdown_DrainsBuffer(t *testing.T) {
	var captured []Record
	var mu sync.Mutex
	s := newTestStore(100, &captured, &mu)

	const n = 10
	for i := 0; i < n; i++ {
		require.NoError(t, s.Log(context.Background(), Record{TransactionID: int64(i)}))
	}

	done := make(chan struct{})
	go func() {
		s.Shutdown()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Shutdown did not return within timeout")
	}

	mu.Lock()
	defer mu.Unlock()
	assert.Len(t, captured, n, "all buffered records must be flushed on shutdown")
}

func TestFlush_RetriesOnceOnTransientError(t *testing.T) {
	var attempts int
	var captured []Record
	var mu sync.Mutex
	flushDone := make(chan struct{}, 1)

	s := &Store{
		ch:            make(chan Record, 10),
		flushInterval: 10 * time.Millisecond,
		retryBackoff:  0,
		done:          make(chan struct{}),
	}
	s.flushFn = func(batch []Record) error {
		mu.Lock()
		attempts++
		first := attempts == 1
		if !first {
			captured = append(captured, batch...)
		}
		mu.Unlock()
		if first {
			return errors.New("transient")
		}
		select {
		case flushDone <- struct{}{}:
		default:
		}
		return nil
	}
	s.wg.Add(1)
	go s.flusher()
	defer s.Shutdown()

	require.NoError(t, s.Log(context.Background(), Record{TransactionID: 7}))

	select {
	case <-flushDone:
	case <-time.After(2 * time.Second):
		t.Fatal("retry flush never succeeded")
	}

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 2, attempts, "expected one failure followed by one successful retry")
	assert.Len(t, captured, 1)
}
