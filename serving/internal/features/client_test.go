package features

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestClient(addr string) *Client {
	return &Client{rdb: redis.NewClient(&redis.Options{Addr: addr})}
}

func TestGet_Found(t *testing.T) {
	mr, err := miniredis.Run()
	require.NoError(t, err)
	defer mr.Close()

	raw := map[string]interface{}{"f1": 1.5, "f2": -0.3, "f3": nil}
	b, _ := json.Marshal(raw)
	mr.Set("fraud:features:42", string(b))

	c := newTestClient(mr.Addr())
	got, err := c.Get(context.Background(), 42)

	require.NoError(t, err)
	assert.InDelta(t, float32(1.5), got["f1"], 1e-6)
	assert.InDelta(t, float32(-0.3), got["f2"], 1e-6)
	assert.Equal(t, float32(0), got["f3"]) // nil -> 0
}

func TestGet_NotFound(t *testing.T) {
	mr, err := miniredis.Run()
	require.NoError(t, err)
	defer mr.Close()

	c := newTestClient(mr.Addr())
	_, err = c.Get(context.Background(), 99999)

	assert.ErrorIs(t, err, ErrNotFound)
}

func TestGet_InvalidJSON(t *testing.T) {
	mr, err := miniredis.Run()
	require.NoError(t, err)
	defer mr.Close()

	mr.Set("fraud:features:1", "not-json")
	c := newTestClient(mr.Addr())
	_, err = c.Get(context.Background(), 1)

	assert.Error(t, err)
	assert.NotErrorIs(t, err, ErrNotFound)
}
