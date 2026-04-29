package test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"
	"github.com/silentwraith/fraud_ml_platform/serving/internal/features"
)

// testRedisClient wraps a real redis.Client pointed at miniredis for integration tests.
type testRedisClient struct {
	rdb *redis.Client
}

func newTestRedisClient(addr string) *testRedisClient {
	return &testRedisClient{
		rdb: redis.NewClient(&redis.Options{Addr: addr}),
	}
}

func (c *testRedisClient) Get(ctx context.Context, transactionID int64) (map[string]float32, error) {
	key := fmt.Sprintf("fraud:features:%d", transactionID)
	val, err := c.rdb.Get(ctx, key).Result()
	if errors.Is(err, redis.Nil) {
		return nil, features.ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
	if err := json.Unmarshal([]byte(val), &raw); err != nil {
		return nil, err
	}
	out := make(map[string]float32, len(raw))
	for k, v := range raw {
		switch tv := v.(type) {
		case float64:
			out[k] = float32(tv)
		case nil:
			out[k] = 0
		}
	}
	return out, nil
}
