package upstream

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type ErrorKind string

const (
	KindTimeout  ErrorKind = "timeout"
	KindUpstream ErrorKind = "upstream_error"
)

type UpstreamError struct {
	Kind    ErrorKind
	Message string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("%s: %s", e.Kind, e.Message)
}

type ScoreRequest struct {
	TransactionID int64 `json:"transaction_id"`
}

type ScoreResponse struct {
	TransactionID    int64   `json:"transaction_id"`
	PredictionID     string  `json:"prediction_id"`
	FraudProbability float64 `json:"fraud_probability"`
	Flagged          bool    `json:"flagged"`
	ModelVersion     string  `json:"model_version"`
	LatencyMS        float64 `json:"latency_ms"`
}

type Client struct {
	httpClient *http.Client
	baseURL    string
	name       string
}

func New(name, baseURL string, timeoutMS int) *Client {
	return &Client{
		httpClient: &http.Client{Timeout: time.Duration(timeoutMS) * time.Millisecond},
		baseURL:    baseURL,
		name:       name,
	}
}

func (c *Client) Name() string { return c.name }

func (c *Client) Score(ctx context.Context, req ScoreRequest) (*ScoreResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, &UpstreamError{Kind: KindUpstream, Message: err.Error()}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/score", bytes.NewReader(body))
	if err != nil {
		return nil, &UpstreamError{Kind: KindUpstream, Message: err.Error()}
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		kind := KindUpstream
		if isTimeout(err) {
			kind = KindTimeout
		}
		return nil, &UpstreamError{Kind: kind, Message: err.Error()}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, &UpstreamError{Kind: KindUpstream, Message: fmt.Sprintf("status %d: %s", resp.StatusCode, b)}
	}

	var sr ScoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, &UpstreamError{Kind: KindUpstream, Message: err.Error()}
	}
	return &sr, nil
}

func (c *Client) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("upstream %s health returned %d", c.name, resp.StatusCode)
	}
	return nil
}

func isTimeout(err error) bool {
	if err == nil {
		return false
	}
	type timeoutErr interface{ Timeout() bool }
	if te, ok := err.(timeoutErr); ok {
		return te.Timeout()
	}
	return false
}
