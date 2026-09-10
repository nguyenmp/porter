package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
)

// ProviderEnv is the environment variable that sets which endpoints may answer
// a model request. See ParseProvider for its two forms.
const ProviderEnv = "PORTER_PROVIDER"

// ProviderPreferences is the `provider` object of a Chat Completions request.
// It holds the gateway's routing rules, which decide which endpoint answers.
// One model usually has many endpoints: providers that serve it at different
// precisions. A gateway that does not know the field ignores it, so leaving it
// unset changes nothing.
//
// The fields follow OpenRouter
// (https://openrouter.ai/docs/features/provider-routing). Every field is
// optional. For evals, the two that matter are Only (name one endpoint) and
// AllowFallbacks (false, so a run measures that endpoint and nothing else).
type ProviderPreferences struct {
	// Order lists provider slugs to try, in this order. Every slug after the
	// first acts as a fallback, so Order on its own still allows other
	// providers.
	Order []string `json:"order,omitempty"`
	// Only allows just these provider slugs. Nothing else may answer.
	Only []string `json:"only,omitempty"`
	// Ignore skips these provider slugs.
	Ignore []string `json:"ignore,omitempty"`
	// AllowFallbacks lets the gateway use a backup provider when the chosen one
	// is unavailable. False makes the request fail instead of going elsewhere.
	AllowFallbacks *bool `json:"allow_fallbacks,omitempty"`
	// RequireParameters keeps only providers that support every parameter the
	// request sends, tools among them. Use it to keep a run that calls tools
	// away from an endpoint that cannot call them.
	RequireParameters *bool `json:"require_parameters,omitempty"`
	// Quantizations keeps only these precisions, for example ["fp8"]. One
	// provider often serves a model at several precisions, so its name alone
	// may not pick out one endpoint: DeepInfra serves gpt-oss-120b at both bf16
	// and fp8, and the router picks between them.
	Quantizations []string `json:"quantizations,omitempty"`
	// Sort orders providers by price, throughput, or latency.
	Sort *ProviderSort `json:"sort,omitempty"`
	// DataCollection is "allow" or "deny". "deny" keeps out providers that may
	// store the request.
	DataCollection string `json:"data_collection,omitempty"`
	// ZeroDataRetention keeps requests on endpoints that do not retain data.
	ZeroDataRetention *bool `json:"zdr,omitempty"`
	// EnforceDistillableText keeps requests on models that allow text
	// distillation.
	EnforceDistillableText *bool `json:"enforce_distillable_text,omitempty"`
	// PreferredMinThroughput and PreferredMaxLatency move endpoints that miss
	// the threshold to the end of the list. Such endpoints can still answer.
	PreferredMinThroughput *ProviderThreshold `json:"preferred_min_throughput,omitempty"`
	PreferredMaxLatency    *ProviderThreshold `json:"preferred_max_latency,omitempty"`
	// MaxPrice caps the price this request may pay.
	MaxPrice *MaxPrice `json:"max_price,omitempty"`
}

// ProviderSort is the `sort` preference: a plain string ("price", "throughput",
// or "latency"), or an object that names a field and a partition to sort
// within.
type ProviderSort struct {
	By        string `json:"by,omitempty"`
	Partition string `json:"partition,omitempty"`
}

// UnmarshalJSON accepts either the string or the object form. Anything else is
// an error instead of a value dropped without a word: a routing rule read
// wrongly changes which endpoint answers, and the run still looks fine.
func (s *ProviderSort) UnmarshalJSON(data []byte) error {
	var by string
	if err := json.Unmarshal(data, &by); err == nil {
		s.By = by
		return nil
	}
	var obj struct {
		By        string `json:"by"`
		Partition string `json:"partition"`
	}
	if err := strictUnmarshal(data, &obj); err != nil {
		return fmt.Errorf("sort must be a string or an object with by and partition: %w", err)
	}
	s.By, s.Partition = obj.By, obj.Partition
	return nil
}

// ProviderThreshold is a performance threshold: a plain number, which applies
// to the median (p50), or an object that names any of the four percentiles.
// Each percentile covers the last five minutes.
type ProviderThreshold struct {
	P50 *float64 `json:"p50,omitempty"`
	P75 *float64 `json:"p75,omitempty"`
	P90 *float64 `json:"p90,omitempty"`
	P99 *float64 `json:"p99,omitempty"`
}

// UnmarshalJSON accepts either the number or the object form.
func (t *ProviderThreshold) UnmarshalJSON(data []byte) error {
	var n float64
	if err := json.Unmarshal(data, &n); err == nil {
		t.P50 = &n
		return nil
	}
	type plain ProviderThreshold
	var p plain
	if err := strictUnmarshal(data, &p); err != nil {
		return fmt.Errorf("threshold must be a number or an object with p50/p75/p90/p99: %w", err)
	}
	*t = ProviderThreshold(p)
	return nil
}

// MaxPrice caps what a request may cost. Prompt and Completion are US dollars
// per million tokens. Request and Image are charges per request and per image.
type MaxPrice struct {
	Prompt     *float64 `json:"prompt,omitempty"`
	Completion *float64 `json:"completion,omitempty"`
	Request    *float64 `json:"request,omitempty"`
	Image      *float64 `json:"image,omitempty"`
}

// ParseProvider reads the value of PORTER_PROVIDER. An empty value means no
// preference: the gateway picks the endpoint. OpenRouter spreads requests
// across providers and favors the cheapest.
//
// Two forms are accepted:
//
//   - A JSON object with the ProviderPreferences fields, for example
//     {"only":["deepinfra/fp8"],"allow_fallbacks":false,"quantizations":["fp8"]}
//     An unknown field is an error, not ignored, so a typo cannot change where
//     requests go without a word.
//   - A comma-separated list of provider slugs, for example "deepinfra/fp8".
//     This shorthand pins those endpoints and turns fallbacks off, which is
//     what an eval wants: the run measures them or fails, and never measures
//     another endpoint without saying so.
func ParseProvider(value string) (*ProviderPreferences, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil, nil
	}
	if !strings.HasPrefix(value, "{") {
		var only []string
		for _, slug := range strings.Split(value, ",") {
			if slug = strings.TrimSpace(slug); slug != "" {
				only = append(only, slug)
			}
		}
		if len(only) == 0 {
			return nil, fmt.Errorf("%s: no provider slug in %q", ProviderEnv, value)
		}
		noFallbacks := false
		return &ProviderPreferences{Only: only, AllowFallbacks: &noFallbacks}, nil
	}
	var prefs ProviderPreferences
	if err := strictUnmarshal([]byte(value), &prefs); err != nil {
		if strings.Contains(err.Error(), "unknown field") {
			return nil, fmt.Errorf("%s: %w (accepted fields: %s)",
				ProviderEnv, err, strings.Join(preferenceFieldNames(), ", "))
		}
		return nil, fmt.Errorf("%s: %w", ProviderEnv, err)
	}
	return &prefs, nil
}

// strictUnmarshal decodes JSON into v and rejects fields v does not define, so
// a misspelled field fails instead of being dropped.
func strictUnmarshal(data []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	if err := dec.Decode(new(json.RawMessage)); err != io.EOF {
		return fmt.Errorf("unexpected trailing data after the JSON value")
	}
	return nil
}

// preferenceFieldNames lists the JSON field names of ProviderPreferences, so an
// unknown-field error can name the fields that do exist. It reads the struct
// tags, so the list always matches the struct.
func preferenceFieldNames() []string {
	t := reflect.TypeOf(ProviderPreferences{})
	names := make([]string, 0, t.NumField())
	for i := 0; i < t.NumField(); i++ {
		name, _, _ := strings.Cut(t.Field(i).Tag.Get("json"), ",")
		if name != "" {
			names = append(names, name)
		}
	}
	return names
}
