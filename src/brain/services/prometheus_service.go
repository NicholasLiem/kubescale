package services

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

type PrometheusService struct {
	baseURL string
	client  *http.Client
}

type PrometheusResponse struct {
	Status string `json:"status"`
	Data   struct {
		ResultType string `json:"resultType"`
		Result     []struct {
			Metric map[string]string `json:"metric"`
			Value  []interface{}     `json:"value,omitempty"`
			Values [][]interface{}   `json:"values,omitempty"`
		} `json:"result"`
	} `json:"data"`
}

func NewPrometheusService(prometheusURL string) *PrometheusService {
	return &PrometheusService{
		baseURL: prometheusURL,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (p *PrometheusService) QueryRange(query string, startTime, endTime time.Time, step string) (*PrometheusResponse, error) {
	params := url.Values{}
	params.Add("query", query)
	params.Add("start", fmt.Sprintf("%d", startTime.Unix()))
	params.Add("end", fmt.Sprintf("%d", endTime.Unix()))
	params.Add("step", step)

	url := fmt.Sprintf("%s/api/v1/query_range?%s", p.baseURL, params.Encode())

	resp, err := p.client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to query Prometheus: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Prometheus query failed with status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var prometheusResp PrometheusResponse
	if err := json.Unmarshal(body, &prometheusResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &prometheusResp, nil
}

func (p *PrometheusService) Query(query string) (*PrometheusResponse, error) {
	params := url.Values{}
	params.Add("query", query)

	url := fmt.Sprintf("%s/api/v1/query?%s", p.baseURL, params.Encode())

	resp, err := p.client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to query Prometheus: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Prometheus query failed with status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var prometheusResp PrometheusResponse
	if err := json.Unmarshal(body, &prometheusResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &prometheusResp, nil
}

func (p *PrometheusService) GetAvailableMetrics() ([]string, error) {
	url := fmt.Sprintf("%s/api/v1/label/__name__/values", p.baseURL)

	resp, err := p.client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to get metrics: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var result struct {
		Status string   `json:"status"`
		Data   []string `json:"data"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metrics: %w", err)
	}

	return result.Data, nil
}
