package recall

// Replied-header scrubbing.
//
// The model-view projection (ProjectModelView) annotates committed assistant
// messages with a timing note ("[replied 2026-09-04 19:06:23 UTC, took 2.2s]")
// prepended to the message content on the outgoing copy of history. Because
// the note sits inside the content field of the very role the model is about
// to write, models sometimes imitate it and open their own reply with the same
// line. That imitation is stored as real assistant text, where it compounds:
// the next request's projection prepends another real note in front of it, and
// the stored copy keeps teaching the pattern.
//
// The real note is never stored — it is added fresh by the projection on every
// request — so a "[replied ...]" header in model-authored content is, by
// construction, a copy of porter's own note rather than text the model meant
// to write. This file strips exactly that header from assistant output as it
// streams in (RepliedScrubber, fed each message_delta) and as it is assembled
// (StripRepliedPrefix, on the decoder's final message), so neither the live
// view nor the committed history ever shows the imitation.
//
// The matcher is deliberately strict: it accepts only the two shapes timingNote
// generates — "[replied <UTC stamp>]" and "[replied <UTC stamp>, took
// <duration>]" — appearing at the start of the content (after optional leading
// whitespace) and followed by whitespace, a newline, or the end of the content.
// Anything else — a mid-content echo, a quote in prose, "[replied to ...]" —
// passes through untouched. The one unavoidable ambiguity is a reply that
// genuinely opens by quoting porter's own header verbatim; that is
// indistinguishable from the imitation and is stripped too.

// isHeaderSpace reports whether c is whitespace the note's surrounding layout
// uses: the model-view projection places the note on its own line, and a copy
// may trail it with spaces, tabs, or blank lines before the real reply starts.
func isHeaderSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

// repliedHeaderState classifies how a text buffer begins relative to the
// "[replied ...]" grammar.
type repliedHeaderState int

const (
	// repliedNo: the text can never begin with a replied header; pass it
	// through untouched.
	repliedNo repliedHeaderState = iota
	// repliedPartial: the text ends inside a possible header (or in the
	// whitespace before one); more input may still complete a header.
	repliedPartial
	// repliedDone: the text is a complete header (plus any trailing
	// whitespace) with no real content after it yet.
	repliedDone
	// repliedYes: the text begins with a complete header followed by real
	// content; rest holds that content with the header and its trailing
	// whitespace stripped.
	repliedYes
)

// scanRepliedHeader classifies the leading text of s against the replied-note
// grammar. rest is meaningful only for repliedYes.
func scanRepliedHeader(s string) (state repliedHeaderState, rest string) {
	i := 0
	for i < len(s) && isHeaderSpace(s[i]) {
		i++
	}
	if i >= len(s) {
		// Whitespace only so far: a header may still follow.
		return repliedPartial, ""
	}
	end := repliedAfter(s, i)
	if end == -2 {
		return repliedPartial, ""
	}
	if end == -1 {
		return repliedNo, ""
	}
	// A complete header runs s[i:end]. What follows the closing bracket
	// decides: the note always ends its line, so a bracket glued to text
	// ("...took 2.2s]Sure") is not the note's shape and nothing is stripped.
	j := end
	if j < len(s) && !isHeaderSpace(s[j]) {
		return repliedNo, ""
	}
	for j < len(s) && isHeaderSpace(s[j]) {
		j++
	}
	if j >= len(s) {
		return repliedDone, ""
	}
	return repliedYes, s[j:]
}

// repliedAfter reports how s[i:] relates to the header grammar, where i is the
// offset of the first character after any leading whitespace. It returns the
// offset just past the closing ']' when s[i:] begins with a complete header;
// -1 when s[i:] can never form a header; or -2 when s[i:] ends inside a
// possible header (a plausible prefix that more input could complete).
func repliedAfter(s string, i int) int {
	n := len(s)
	lit := "[replied "
	for k := 0; k < len(lit); k++ {
		if i >= n {
			return -2
		}
		if s[i] != lit[k] {
			return -1
		}
		i++
	}
	// "2026-09-04 19:06:23 UTC": fixed shape; a '0' in the template stands
	// for any digit (utcStamp formats with zero-padded fields).
	const stamp = "0000-00-00 00:00:00 UTC"
	for k := 0; k < len(stamp); k++ {
		if i >= n {
			return -2
		}
		e := stamp[k]
		if e == '0' {
			if s[i] < '0' || s[i] > '9' {
				return -1
			}
		} else if s[i] != e {
			return -1
		}
		i++
	}
	if i >= n {
		return -2
	}
	if s[i] == ']' {
		return i + 1
	}
	// ", took " + a duration in durText form ("2.2s", "45s", "3m 12s", "1h 5m").
	lit = ", took "
	for k := 0; k < len(lit); k++ {
		if i >= n {
			return -2
		}
		if s[i] != lit[k] {
			return -1
		}
		i++
	}
	// Leading digits (at least one).
	if i >= n {
		return -2
	}
	if s[i] < '0' || s[i] > '9' {
		return -1
	}
	for i < n && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	// Optional ".d" fractional seconds (durText's %.1fs form).
	if i < n && s[i] == '.' {
		i++
		if i >= n {
			return -2
		}
		if s[i] < '0' || s[i] > '9' {
			return -1
		}
		i++
	}
	// Unit: s, m, or h.
	if i >= n {
		return -2
	}
	if s[i] != 's' && s[i] != 'm' && s[i] != 'h' {
		return -1
	}
	i++
	// Optional second component ("3m 12s", "1h 5m").
	if i < n && s[i] == ' ' {
		i++
		if i >= n {
			return -2
		}
		if s[i] < '0' || s[i] > '9' {
			return -1
		}
		for i < n && s[i] >= '0' && s[i] <= '9' {
			i++
		}
		if i >= n {
			return -2
		}
		if s[i] != 's' && s[i] != 'm' && s[i] != 'h' {
			return -1
		}
		i++
	}
	// Closing bracket.
	if i >= n {
		return -2
	}
	if s[i] != ']' {
		return -1
	}
	return i + 1
}

// StripRepliedPrefix removes a hallucinated "[replied ...]" header from the
// start of model-authored content, returning the content with the header (and
// any whitespace around it) gone. It returns s unchanged when s does not begin
// with a complete header. The agent applies it to the decoder's assembled
// message, which is the authoritative text of every committed assistant reply.
func StripRepliedPrefix(s string) string {
	state, rest := scanRepliedHeader(s)
	switch state {
	case repliedYes:
		return rest
	case repliedDone:
		// The whole content was the header (plus trailing whitespace).
		return ""
	default:
		return s
	}
}

// repliedScrubCap bounds how much leading text RepliedScrubber holds back
// before forcing a decision. A real header is at most ~60 characters, so this
// only ever trips on a pathological opening that never resolves; genuine
// replies that cannot be a header are released after the first few characters
// (the matcher rejects at the first impossible byte), so the cap is a safety
// valve, not a latency target.
const repliedScrubCap = 160

// RepliedScrubber withholds the opening of a streamed assistant reply until it
// can tell a hallucinated "[replied ...]" header apart from real content, so
// the header is dropped before it reaches the live view or the reply buffer.
// Feed must be called with each streamed content delta in order; it returns
// the text to forward for that delta ("" while the opening is held or is being
// swallowed). A reply that begins with anything other than the header is
// released immediately and passes through untouched from then on.
type RepliedScrubber struct {
	buf  string // held opening, released once the leading text resolves
	done bool   // resolved: pass everything through from here on
}

// NewRepliedScrubber returns a scrubber ready to consume a reply's deltas.
func NewRepliedScrubber() *RepliedScrubber {
	return &RepliedScrubber{}
}

// Feed consumes one content delta and returns the text to forward: the delta
// itself when the scrubber has resolved and the opening is clean, the resolved
// remainder when a header was stripped, or "" while the opening is held (it
// may yet become a header) or swallowed (a complete header's trailing
// whitespace). Once the opening resolves either way the scrubber is done and
// every later delta passes through unchanged.
func (s *RepliedScrubber) Feed(delta string) string {
	if s.done {
		return delta
	}
	s.buf += delta
	state, rest := scanRepliedHeader(s.buf)
	switch state {
	case repliedYes:
		// A header followed by real content: strip the header, release the
		// content, and never look at the opening again.
		s.buf = ""
		s.done = true
		return rest
	case repliedNo:
		// The opening can never be a header: release it untouched.
		out := s.buf
		s.buf = ""
		s.done = true
		return out
	case repliedDone:
		// A complete header with only whitespace after it: it is junk, so
		// swallow it and keep waiting for real content. A pathological run
		// of whitespace past the cap commits to the strip.
		if len(s.buf) >= repliedScrubCap {
			s.buf = ""
			s.done = true
		}
		return ""
	default: // repliedPartial
		// Still plausibly a header: hold. If the opening runs past the cap
		// without resolving, treat it as real content and pass it through.
		if len(s.buf) >= repliedScrubCap {
			out := s.buf
			s.buf = ""
			s.done = true
			return out
		}
		return ""
	}
}
