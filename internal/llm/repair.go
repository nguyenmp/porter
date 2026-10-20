package llm

// InterruptedToolResult is the result recorded for a tool call that produced
// none, because the user stopped the turn or the server restarted while the
// call was still pending. The model reads this text on its next request, so it
// says what happened and what to check before retrying.
const InterruptedToolResult = "[interrupted: no result was recorded for this tool call. The app stopped or the server restarted while the tool was running, so it may have finished, stopped partway, or never started. Check the working directory before you retry.]"

// RepairToolCalls returns msgs with a result for every tool call, so the slice
// is safe to send to a provider.
//
// Chat-completions providers reject history that breaks the pairing between an
// assistant message and its results: a message that asks for three tool calls
// must be followed by three role-"tool" messages, one per call. The agent
// commits the assistant message first and each result as its call finishes, so
// anything that ends the process in between — a Stop, a crash, a restart —
// leaves stored history that no provider accepts, and every later request on
// that chat fails with an error about the missing tool result.
//
// The repair happens on read. Each call without a result gets
// InterruptedToolResult, placed after the results that do exist and in the
// order of the calls. A role-"tool" message that answers no call is dropped,
// because providers reject those too. Messages that are already paired pass
// through unchanged, and a second run changes nothing, so more than one caller
// can apply it.
func RepairToolCalls(msgs []ChatMessage) []ChatMessage {
	return RepairToolCallsWith(msgs,
		func(m ChatMessage) ChatMessage { return m },
		func(m ChatMessage) ChatMessage { return m })
}

// RepairToolCallsWith does the work of RepairToolCalls for a caller whose rows
// wrap a ChatMessage — the database's Message, which also carries a bus seq. get
// returns a row's message, and build makes a row from a message.
func RepairToolCallsWith[T any](rows []T, get func(T) ChatMessage, build func(ChatMessage) T) []T {
	// declared holds every call id an assistant message asked for.
	declared := make(map[string]bool)
	for _, row := range rows {
		for _, tc := range get(row).ToolCalls {
			declared[tc.ID] = true
		}
	}
	out := make([]T, 0, len(rows))
	changed := false
	for i := 0; i < len(rows); i++ {
		m := get(rows[i])
		if m.Role == "tool" {
			// A result with no assistant message in front of it. The pairing
			// cannot be rebuilt from here, so drop it. Results that do belong
			// to a call are collected by the run scan below.
			changed = true
			continue
		}
		out = append(out, rows[i])
		if m.Role != "assistant" || len(m.ToolCalls) == 0 {
			continue
		}
		// Collect the run of results that follows the assistant message: every
		// role-"tool" message up to the next message of another role.
		answered := make(map[string]bool, len(m.ToolCalls))
		j := i + 1
		for ; j < len(rows) && get(rows[j]).Role == "tool"; j++ {
			id := get(rows[j]).ToolCallID
			if id == "" || answered[id] || !declared[id] {
				// A second result for one call, or a result for a call nothing
				// declared: a provider rejects both, so drop it.
				changed = true
				continue
			}
			answered[id] = true
			out = append(out, rows[j])
		}
		i = j - 1
		// Answer every call the run did not. They go last, so the results that
		// did arrive keep their completion order.
		for _, tc := range m.ToolCalls {
			if answered[tc.ID] {
				continue
			}
			answered[tc.ID] = true
			changed = true
			out = append(out, build(interruptedResult(tc.ID)))
		}
	}
	if !changed {
		return rows
	}
	return out
}

// interruptedResult is the role-"tool" message built in place of a result that
// never arrived. It is marked cancelled so history and the UI show the run as
// aborted rather than as a tool that returned that text.
func interruptedResult(id string) ChatMessage {
	m := ToolResult(id, InterruptedToolResult)
	m.Cancelled = true
	return m
}
