package predictions

import (
	"context"
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
		done:          make(chan struct{}),
	}
	s.flushFn = func(batch []Record) {
		mu.Lock()
		*captured = append(*captured, batch...)
		mu.Unlock()
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

func TestLog_BufferOverflow_FlushesSync(t *testing.T) {
	var captured []Record
	var mu sync.Mutex
	// Buffer of 1: second call overflows and flushes synchronously
	s := newTestStore(1, &captured, &mu)
	defer s.Shutdown()

	require.NoError(t, s.Log(context.Background(), Record{TransactionID: 1}))
	require.NoError(t, s.Log(context.Background(), Record{TransactionID: 2}))

	mu.Lock()
	count := len(captured)
	mu.Unlock()
	assert.GreaterOrEqual(t, count, 1)
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
