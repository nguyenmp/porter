package recall

import (
	"strings"
	"testing"
)

// header is the exact shape timingNote generates; the tests below vary its
// surroundings to pin down when a header is stripped and when it is not.
const header = "[replied 2026-09-04 19:06:23 UTC, took 2.2s]"

func TestStripRepliedPrefix(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"header then blank lines", header + "\n\nGreat question.", "Great question."},
		{"header then single newline", "[replied 2026-09-04 19:06:23 UTC]\nDone.", "Done."},
		{"header then spaces same line", header + " Great question.", "Great question."},
		{"header then tabs and newline", header + "\t\nGreat question.", "Great question."},
		{"header alone", header, ""},
		{"header plus trailing spaces", header + "   ", ""},
		{"leading newline before header", "\n" + header + "\nGreat question.", "Great question."},
		{"leading blank line before header", "\n\n" + header + "\nGreat question.", "Great question."},
		{"duration variants", "[replied 2026-09-04 19:06:23 UTC, took 45s]\nA", "A"},
		{"duration minutes", "[replied 2026-09-04 19:06:23 UTC, took 3m 12s]\nA", "A"},
		{"duration hours", "[replied 2026-09-04 19:06:23 UTC, took 1h 5m]\nA", "A"},
		{"fractional over ten", "[replied 2026-09-04 19:06:23 UTC, took 10.0s]\nA", "A"},
		{"empty", "", ""},
		{"whitespace only", "   ", "   "},

		// Not headers: must pass through untouched.
		{"bracket glued to text", header + "Sure.", header + "Sure."},
		{"prose replied", "[replied to your question] Sure.", "[replied to your question] Sure."},
		{"bad duration word", "[replied 2026-09-04 19:06:23 UTC, took ages] Sure.", "[replied 2026-09-04 19:06:23 UTC, took ages] Sure."},
		{"bad duration shape", "[replied 2026-09-04 19:06:23 UTC, took 2.2] Sure.", "[replied 2026-09-04 19:06:23 UTC, took 2.2] Sure."},
		{"truncated stamp", "[replied 2026-09-04] Sure.", "[replied 2026-09-04] Sure."},
		{"wrong stamp literal", "[replied 2026-09-04 19:06:23 GMT] Sure.", "[replied 2026-09-04 19:06:23 GMT] Sure."},
		{"mid-content", "First.\n\n" + header, "First.\n\n" + header},
		{"leading text", "Sure, " + header, "Sure, " + header},
		{"missing bracket", "[replied 2026-09-04 19:06:23 UTC, took 2.2s\nSure.", "[replied 2026-09-04 19:06:23 UTC, took 2.2s\nSure."},
		{"wrong prefix", "[answered 2026-09-04 19:06:23 UTC, took 2.2s] Sure.", "[answered 2026-09-04 19:06:23 UTC, took 2.2s] Sure."},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := StripRepliedPrefix(tt.in); got != tt.want {
				t.Errorf("StripRepliedPrefix(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// TestStripRepliedPrefixStripsOnlyLeadingHeader: a second "[replied ...]" later
// in the content is real text (an echo or quote), not the imitation of the
// leading note, and must survive.
func TestStripRepliedPrefixStripsOnlyLeadingHeader(t *testing.T) {
	in := header + "\nSee also: " + header
	if got := StripRepliedPrefix(in); got != "See also: "+header {
		t.Errorf("got %q, want the second header kept", got)
	}
}

func TestRepliedScrubberFeed(t *testing.T) {
	t.Run("header split across deltas", func(t *testing.T) {
		s := NewRepliedScrubber()
		got := ""
		for _, d := range []string{"[rep", "lied 2026-09-04 19:06:23 ", "UTC, took 2.2s]", "\n\nThe answer.", " More."} {
			got += s.Feed(d)
		}
		if got != "The answer. More." {
			t.Errorf("streamed text = %q, want %q", got, "The answer. More.")
		}
	})
	t.Run("header then newline then text in later deltas", func(t *testing.T) {
		s := NewRepliedScrubber()
		if got := s.Feed(header); got != "" {
			t.Errorf("first feed = %q, want held", got)
		}
		if got := s.Feed("\n\n"); got != "" {
			t.Errorf("whitespace feed = %q, want swallowed", got)
		}
		if got := s.Feed("Done."); got != "Done." {
			t.Errorf("text feed = %q, want %q", got, "Done.")
		}
		if got := s.Feed("  Follow-up."); got != "  Follow-up." {
			t.Errorf("post-resolution feed = %q, want passthrough", got)
		}
	})
	t.Run("clean reply released immediately", func(t *testing.T) {
		s := NewRepliedScrubber()
		if got := s.Feed("Hello"); got != "Hello" {
			t.Errorf("first feed = %q, want immediate release", got)
		}
		if got := s.Feed(" world"); got != " world" {
			t.Errorf("later feed = %q, want passthrough", got)
		}
	})
	t.Run("reply with leading blank lines", func(t *testing.T) {
		s := NewRepliedScrubber()
		if got := s.Feed("\n\nHello"); got != "\n\nHello" {
			t.Errorf("got %q, want leading whitespace released with the text", got)
		}
	})
	t.Run("prose that starts like a header", func(t *testing.T) {
		s := NewRepliedScrubber()
		in := "[replied to your question] Sure."
		if got := s.Feed(in); got != in {
			t.Errorf("got %q, want %q untouched", got, in)
		}
	})
	t.Run("stamp that goes bad mid-stream", func(t *testing.T) {
		s := NewRepliedScrubber()
		if got := s.Feed("[replied 2"); got != "" {
			t.Errorf("first feed = %q, want held", got)
		}
		in := "0xx] body"
		if got := s.Feed(in); got != "[replied 20xx] body" {
			t.Errorf("got %q, want the opening released untouched once invalid", got)
		}
	})
	t.Run("cap passes a pathological opening through", func(t *testing.T) {
		s := NewRepliedScrubber()
		var fed strings.Builder
		fed.WriteString("[replied 2026-09-04 19:06:23 UTC, took ")
		out := ""
		// Feed one delta at a time so the scrubber sees each chunk; once the
		// buffer passes the cap it must release everything untouched.
		for _, c := range fed.String() {
			out += s.Feed(string(c))
		}
		big := strings.Repeat("9", repliedScrubCap)
		out += s.Feed(big)
		if out != fed.String()+big {
			t.Errorf("cap release mismatch: got %d bytes, want %d", len(out), len(fed.String())+len(big))
		}
		if !strings.HasPrefix(out, "[replied ") {
			t.Errorf("cap release = %q, want the original opening kept", out)
		}
	})
	t.Run("empty deltas", func(t *testing.T) {
		s := NewRepliedScrubber()
		if got := s.Feed(""); got != "" {
			t.Errorf("empty feed = %q, want empty", got)
		}
	})
}
