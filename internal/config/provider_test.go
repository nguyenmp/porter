package config

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseProviderEmpty(t *testing.T) {
	for _, in := range []string{"", "   "} {
		p, err := ParseProvider(in)
		if err != nil || p != nil {
			t.Fatalf("ParseProvider(%q) = %v, %v; want nil, nil", in, p, err)
		}
	}
}

// TestParseProviderSlugShorthand covers the eval's common case: naming
// endpoints pins them and turns fallbacks off, so a run measures those
// endpoints or fails, rather than measuring another one without saying so.
func TestParseProviderSlugShorthand(t *testing.T) {
	p, err := ParseProvider("deepinfra/fp8")
	if err != nil {
		t.Fatalf("ParseProvider: %v", err)
	}
	if len(p.Only) != 1 || p.Only[0] != "deepinfra/fp8" {
		t.Fatalf("Only = %v, want [deepinfra/fp8]", p.Only)
	}
	if p.AllowFallbacks == nil || *p.AllowFallbacks {
		t.Fatalf("AllowFallbacks = %v, want false", p.AllowFallbacks)
	}

	p, err = ParseProvider(" deepinfra/fp8 , novita/fp8 ,, ")
	if err != nil {
		t.Fatalf("ParseProvider list: %v", err)
	}
	if len(p.Only) != 2 || p.Only[0] != "deepinfra/fp8" || p.Only[1] != "novita/fp8" {
		t.Fatalf("Only = %v, want [deepinfra/fp8 novita/fp8]", p.Only)
	}
}

func TestParseProviderObject(t *testing.T) {
	p, err := ParseProvider(`{"only":["deepinfra/fp8"],"allow_fallbacks":false,"quantizations":["fp8"],"require_parameters":true}`)
	if err != nil {
		t.Fatalf("ParseProvider: %v", err)
	}
	if len(p.Only) != 1 || p.Only[0] != "deepinfra/fp8" {
		t.Fatalf("Only = %v", p.Only)
	}
	if p.AllowFallbacks == nil || *p.AllowFallbacks {
		t.Fatalf("AllowFallbacks = %v, want false", p.AllowFallbacks)
	}
	if p.RequireParameters == nil || !*p.RequireParameters {
		t.Fatalf("RequireParameters = %v, want true", p.RequireParameters)
	}
	if len(p.Quantizations) != 1 || p.Quantizations[0] != "fp8" {
		t.Fatalf("Quantizations = %v", p.Quantizations)
	}
}

// TestParseProviderRejectsUnknownField keeps a misspelled field loud. An
// ignored typo means a run that routed somewhere else while reporting success,
// which is exactly what an eval must not do.
func TestParseProviderRejectsUnknownField(t *testing.T) {
	_, err := ParseProvider(`{"onnly":["deepinfra/fp8"]}`)
	if err == nil {
		t.Fatal("ParseProvider accepted an unknown field")
	}
	if !strings.Contains(err.Error(), "unknown field") ||
		!strings.Contains(err.Error(), "only") {
		t.Fatalf("error should name the unknown field and the accepted ones, got: %v", err)
	}
	if !strings.Contains(err.Error(), ProviderEnv) {
		t.Fatalf("error should name %s, got: %v", ProviderEnv, err)
	}
}

func TestParseProviderRejectsBadJSON(t *testing.T) {
	for _, in := range []string{`{`, `{"only":`, `{"sort":"throughput"} trailing`} {
		if _, err := ParseProvider(in); err == nil {
			t.Fatalf("ParseProvider(%q) succeeded, want an error", in)
		}
	}
}

func TestProviderSortStringAndObject(t *testing.T) {
	p, err := ParseProvider(`{"sort":"throughput"}`)
	if err != nil {
		t.Fatalf("ParseProvider: %v", err)
	}
	if p.Sort == nil || p.Sort.By != "throughput" || p.Sort.Partition != "" {
		t.Fatalf("Sort = %+v, want By=throughput", p.Sort)
	}

	p, err = ParseProvider(`{"sort":{"by":"price","partition":"model"}}`)
	if err != nil {
		t.Fatalf("ParseProvider: %v", err)
	}
	if p.Sort == nil || p.Sort.By != "price" || p.Sort.Partition != "model" {
		t.Fatalf("Sort = %+v, want By=price Partition=model", p.Sort)
	}

	if _, err := ParseProvider(`{"sort":{"buy":"price"}}`); err == nil {
		t.Fatal("ParseProvider accepted an unknown field inside sort")
	}
	if _, err := ParseProvider(`{"sort":5}`); err == nil {
		t.Fatal("ParseProvider accepted a number as sort")
	}
}

func TestProviderThresholdNumberAndObject(t *testing.T) {
	p, err := ParseProvider(`{"preferred_min_throughput":120}`)
	if err != nil {
		t.Fatalf("ParseProvider: %v", err)
	}
	if p.PreferredMinThroughput == nil || p.PreferredMinThroughput.P50 == nil ||
		*p.PreferredMinThroughput.P50 != 120 {
		t.Fatalf("PreferredMinThroughput = %+v, want p50=120", p.PreferredMinThroughput)
	}

	p, err = ParseProvider(`{"preferred_max_latency":{"p50":1,"p90":3,"p99":5}}`)
	if err != nil {
		t.Fatalf("ParseProvider: %v", err)
	}
	lat := p.PreferredMaxLatency
	if lat == nil || lat.P50 == nil || *lat.P50 != 1 || lat.P90 == nil || *lat.P90 != 3 ||
		lat.P99 == nil || *lat.P99 != 5 {
		t.Fatalf("PreferredMaxLatency = %+v, want p50=1 p90=3 p99=5", lat)
	}

	if _, err := ParseProvider(`{"preferred_max_latency":{"p95":1}}`); err == nil {
		t.Fatal("ParseProvider accepted an unknown percentile")
	}
}

func TestParseProviderMaxPrice(t *testing.T) {
	p, err := ParseProvider(`{"max_price":{"prompt":1,"completion":2}}`)
	if err != nil {
		t.Fatalf("ParseProvider: %v", err)
	}
	if p.MaxPrice == nil || p.MaxPrice.Prompt == nil || *p.MaxPrice.Prompt != 1 ||
		p.MaxPrice.Completion == nil || *p.MaxPrice.Completion != 2 {
		t.Fatalf("MaxPrice = %+v, want prompt=1 completion=2", p.MaxPrice)
	}
}

// TestProviderJSONRoundTrip checks the parsed object marshals to the field
// names the gateway expects, because that is what leaves the process.
func TestProviderJSONRoundTrip(t *testing.T) {
	p, err := ParseProvider(`deepinfra/fp8`)
	if err != nil {
		t.Fatalf("ParseProvider: %v", err)
	}
	got, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	want := `{"only":["deepinfra/fp8"],"allow_fallbacks":false}`
	if string(got) != want {
		t.Fatalf("Marshal = %s, want %s", got, want)
	}
}

// TestEnvReadsProvider checks the variable is read, and that Validate reports a
// bad value instead of the first request failing on it.
func TestEnvReadsProvider(t *testing.T) {
	t.Setenv(ProviderEnv, "deepinfra/fp8")
	cfg := Env()
	if cfg.Provider == nil || len(cfg.Provider.Only) != 1 {
		t.Fatalf("Env() Provider = %+v, want only [deepinfra/fp8]", cfg.Provider)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	t.Setenv(ProviderEnv, `{"onnly":[]}`)
	cfg = Env()
	if cfg.Validate() == nil {
		t.Fatal("Validate accepted a bad PORTER_PROVIDER value")
	}
}
