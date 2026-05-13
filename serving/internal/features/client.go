package features

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

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

	nan := float32(math.NaN())
	features := make(map[string]float32, len(raw))
	for k, v := range raw {
		switch tv := v.(type) {
		case float64:
			if math.IsNaN(tv) {
				features[k] = nan
			} else {
				features[k] = float32(tv)
			}
		case nil:
			// Train and serve agree on the missing-value policy: NaN. The ONNX
			// TreeEnsembleClassifier carries the booster's missing-branch routing.
			features[k] = nan
		default:
			return nil, fmt.Errorf("unexpected feature type for key %q: %T", k, v)
		}
	}
	return features, nil
}

func (c *Client) Ping(ctx context.Context) error {
	return c.rdb.Ping(ctx).Err()
}

// SampleFeatureNames returns the feature key set from one arbitrary
// fraud:features:* row so callers can detect Redis/model schema drift before
// completing a hot swap; without this check a swap to a model that expects
// names not present in Redis would fail every Score with a feature lookup
// error.
func (c *Client) SampleFeatureNames(ctx context.Context) ([]string, error) {
	iter := c.rdb.Scan(ctx, 0, "fraud:features:*", 1).Iterator()
	if !iter.Next(ctx) {
		if err := iter.Err(); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}
		return nil, ErrNotFound
	}
	val, err := c.rdb.Get(ctx, iter.Val()).Result()
	if err != nil {
		return nil, fmt.Errorf("get %s: %w", iter.Val(), err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(val), &raw); err != nil {
		return nil, fmt.Errorf("unmarshal sample %s: %w", iter.Val(), err)
	}
	names := make([]string, 0, len(raw))
	for k := range raw {
		names = append(names, k)
	}
	return names, nil
}

func (c *Client) Close() error {
	return c.rdb.Close()
}
