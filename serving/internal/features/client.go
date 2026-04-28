package features

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"
)

var ErrNotFound = errors.New("transaction features not found")

type Client struct {
	rdb *redis.Client
}

func NewClient(host, port string) *Client {
	return &Client{
		rdb: redis.NewClient(&redis.Options{
			Addr: fmt.Sprintf("%s:%s", host, port),
		}),
	}
}

func (c *Client) Get(ctx context.Context, transactionID int64) (map[string]float32, error) {
	key := fmt.Sprintf("fraud:features:%d", transactionID)
	val, err := c.rdb.Get(ctx, key).Result()
	if errors.Is(err, redis.Nil) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("redis get %s: %w", key, err)
	}

	var raw map[string]any
	if err := json.Unmarshal([]byte(val), &raw); err != nil {
		return nil, fmt.Errorf("unmarshal features for %d: %w", transactionID, err)
	}

	features := make(map[string]float32, len(raw))
	for k, v := range raw {
		switch tv := v.(type) {
		case float64:
			features[k] = float32(tv)
		case nil:
			features[k] = 0
		default:
			return nil, fmt.Errorf("unexpected feature type for key %q: %T", k, v)
		}
	}
	return features, nil
}

func (c *Client) Ping(ctx context.Context) error {
	return c.rdb.Ping(ctx).Err()
}

func (c *Client) Close() error {
	return c.rdb.Close()
}
