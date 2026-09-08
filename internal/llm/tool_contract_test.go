package llm

import (
	"encoding/json"
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
	for _, d := range out {
		params := d.Function.Parameters
		props := params["properties"].(map[string]any)
		if _, ok := props[ArgPorterDescription]; !ok {
			t.Errorf("%s: missing %s property", d.Function.Name, ArgPorterDescription)
		}
		if _, ok := props[ArgPorterTimeout]; !ok {
			t.Errorf("%s: missing %s property", d.Function.Name, ArgPorterTimeout)
		}
		var req []string
		for _, r := range params["required"].([]string) {
			if r == ArgPorterDescription || r == ArgPorterTimeout {
				req = append(req, r)
			}
		}
		if len(req) != 2 {
			t.Errorf("%s: required list = %v, want both porter fields", d.Function.Name, params["required"])
		}
		// The schema must stay valid JSON (it is what we send to the provider).
		if _, err := json.Marshal(d); err != nil {
			t.Errorf("%s: schema no longer marshals: %v", d.Function.Name, err)
		}
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
