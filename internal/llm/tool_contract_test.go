package llm

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
)

func TestAddToolContract(t *testing.T) {
	defs := []Tool{
		{
			Type: "function",
			Function: Function{
				Name:        "shell",
				Description: "run a command",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"command": map[string]any{"type": "string"},
					},
					"required": []string{"command"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "bare",
				Description: "no required args at all",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		},
	}
	out := AddToolContract(defs)
	// The description leads both the property list and the required list; the
	// timeout trails; the tool's own arguments keep their place between them.
	want := map[string][]string{
		"shell": {ArgPorterDescription, "command", ArgPorterTimeout},
		"bare":  {ArgPorterDescription, ArgPorterTimeout},
	}
	for _, d := range out {
		params := d.Function.Parameters
		props := params["properties"].(orderedProps)
		for _, name := range []string{ArgPorterDescription, ArgPorterTimeout} {
			if _, ok := props[name]; !ok {
				t.Errorf("%s: missing %s property", d.Function.Name, name)
			}
		}
		// The schema is what we send to the provider, so it must stay valid
		// JSON — and its property order is the point of this test.
		raw, err := json.Marshal(d)
		if err != nil {
			t.Errorf("%s: schema no longer marshals: %v", d.Function.Name, err)
			continue
		}
		if keys := propertiesKeys(t, raw); !reflect.DeepEqual(keys, want[d.Function.Name]) {
			t.Errorf("%s: properties order = %v, want %v", d.Function.Name, keys, want[d.Function.Name])
		}
		req := params["required"].([]string)
		if !reflect.DeepEqual(req, want[d.Function.Name]) {
			t.Errorf("%s: required list = %v, want %v", d.Function.Name, req, want[d.Function.Name])
		}
	}
}

// propertiesKeys returns the keys of the first "properties" object in a
// marshaled tool definition, in the order the JSON lists them. It exists so
// tests can assert the order the provider sees; decoding into a map would
// destroy that order.
func propertiesKeys(t *testing.T, raw []byte) []string {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	for {
		tok, err := dec.Token()
		if err != nil {
			t.Fatalf("scan tool schema: %v", err)
		}
		key, ok := tok.(string)
		if !ok || key != "properties" {
			continue
		}
		if tok, err := dec.Token(); err != nil {
			t.Fatalf("read properties object: %v", err)
		} else if d, ok := tok.(json.Delim); !ok || d != '{' {
			t.Fatalf("properties value = %v, want an object", tok)
		}
		var keys []string
		for dec.More() {
			tok, err := dec.Token()
			if err != nil {
				t.Fatalf("read property name: %v", err)
			}
			keys = append(keys, tok.(string))
			skipValue(t, dec)
		}
		dec.Token() // the object's closing '}'
		return keys
	}
}

// skipValue consumes one JSON value from dec, an object or array in full.
func skipValue(t *testing.T, dec *json.Decoder) {
	t.Helper()
	tok, err := dec.Token()
	if err != nil {
		t.Fatalf("skip value: %v", err)
	}
	if d, ok := tok.(json.Delim); ok {
		switch d {
		case '{':
			for dec.More() {
				skipValue(t, dec) // key
				skipValue(t, dec) // value
			}
		case '[':
			for dec.More() {
				skipValue(t, dec)
			}
		}
		dec.Token() // closing delimiter
	}
}

func TestAddToolContractDoesNotDuplicate(t *testing.T) {
	defs := AddToolContract([]Tool{{
		Type: "function",
		Function: Function{
			Name: "s",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}})
	// Applying it twice (a caller that rebuilds defs per request and decorates
	// again) must not duplicate required entries.
	twice := AddToolContract(defs)
	got := twice[0].Function.Parameters["required"].([]string)
	seen := map[string]int{}
	for _, r := range got {
		seen[r]++
	}
	for r, n := range seen {
		if n != 1 {
			t.Errorf("required %q appears %d times after a double apply: %v", r, n, got)
		}
	}
}
