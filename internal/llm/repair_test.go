package llm

import (
	"reflect"
	"testing"
)

// assistantAsking builds an assistant message that asks for one tool call per id.
func assistantAsking(ids ...string) ChatMessage {
	calls := make([]ToolCall, 0, len(ids))
	for _, id := range ids {
		calls = append(calls, ToolCall{ID: id, Type: "function", Function: ToolFunction{Name: "shell", Arguments: "{}"}})
	}
	return AssistantMessage("", "", calls)
}

// roles describes a history as compact strings: "user", "assistant", and
// "tool:<id>" for a tool result.
func roles(msgs []ChatMessage) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == "tool" {
			out = append(out, "tool:"+m.ToolCallID)
			continue
		}
		out = append(out, m.Role)
	}
	return out
}

func TestRepairToolCallsLeavesPairedHistoryAlone(t *testing.T) {
	history := []ChatMessage{
		UserMessage("run two things"),
		assistantAsking("call_1", "call_2"),
		ToolResult("call_1", "first"),
		ToolResult("call_2", "second"),
		UserMessage("thanks"),
	}
	got := RepairToolCalls(history)
	if !reflect.DeepEqual(got, history) {
		t.Errorf("paired history changed:\n got %v\nwant %v", roles(got), roles(history))
	}
}

func TestRepairToolCallsFillsMissingResults(t *testing.T) {
	history := []ChatMessage{
		UserMessage("run three things"),
		assistantAsking("call_1", "call_2", "call_3"),
		ToolResult("call_1", "first finished"),
		UserMessage("and then?"),
	}
	got := RepairToolCalls(history)

	want := []string{"user", "assistant", "tool:call_1", "tool:call_2", "tool:call_3", "user"}
	if !reflect.DeepEqual(roles(got), want) {
		t.Fatalf("roles = %v, want %v", roles(got), want)
	}
	if got[2].Content != "first finished" {
		t.Errorf("existing result was rewritten: %q", got[2].Content)
	}
	for _, i := range []int{3, 4} {
		if got[i].Content != InterruptedToolResult {
			t.Errorf("result %d = %q, want the interrupted marker", i, got[i].Content)
		}
		if !got[i].Cancelled {
			t.Errorf("result %d should be marked cancelled so the UI shows an aborted run", i)
		}
	}
}

func TestRepairToolCallsAnswersCallsWithNoResultsAtAll(t *testing.T) {
	// A restart between the assistant message and the first result.
	history := []ChatMessage{
		UserMessage("go"),
		assistantAsking("call_1", "call_2"),
		UserMessage("still there?"),
	}
	got := RepairToolCalls(history)

	want := []string{"user", "assistant", "tool:call_1", "tool:call_2", "user"}
	if !reflect.DeepEqual(roles(got), want) {
		t.Fatalf("roles = %v, want %v", roles(got), want)
	}
}

func TestRepairToolCallsDropsResultsThatAnswerNoCall(t *testing.T) {
	t.Run("result with no assistant message", func(t *testing.T) {
		history := []ChatMessage{
			UserMessage("go"),
			ToolResult("call_9", "orphan"),
			UserMessage("next"),
		}
		want := []string{"user", "user"}
		if got := roles(RepairToolCalls(history)); !reflect.DeepEqual(got, want) {
			t.Errorf("roles = %v, want %v", got, want)
		}
	})

	t.Run("second result for one call", func(t *testing.T) {
		history := []ChatMessage{
			UserMessage("go"),
			assistantAsking("call_1"),
			ToolResult("call_1", "first"),
			ToolResult("call_1", "duplicate"),
			UserMessage("next"),
		}
		got := RepairToolCalls(history)
		want := []string{"user", "assistant", "tool:call_1", "user"}
		if !reflect.DeepEqual(roles(got), want) {
			t.Fatalf("roles = %v, want %v", roles(got), want)
		}
		if got[2].Content != "first" {
			t.Errorf("kept result = %q, want the first result", got[2].Content)
		}
	})
}

func TestRepairToolCallsIsIdempotent(t *testing.T) {
	broken := []ChatMessage{
		UserMessage("go"),
		assistantAsking("call_1", "call_2"),
		ToolResult("call_1", "done"),
	}
	once := RepairToolCalls(broken)
	twice := RepairToolCalls(once)
	if !reflect.DeepEqual(once, twice) {
		t.Errorf("second repair changed the history:\n once  %v\n twice %v", roles(once), roles(twice))
	}
}

// row is a ChatMessage with a sequence number, like the database's Message.
type row struct {
	Seq uint64
	Msg ChatMessage
}

func TestRepairToolCallsWithKeepsRowData(t *testing.T) {
	rows := []row{
		{Seq: 1, Msg: UserMessage("go")},
		{Seq: 2, Msg: assistantAsking("call_1", "call_2")},
		{Seq: 3, Msg: ToolResult("call_1", "done")},
	}
	got := RepairToolCallsWith(rows,
		func(r row) ChatMessage { return r.Msg },
		func(m ChatMessage) row { return row{Msg: m} })

	want := []uint64{1, 2, 3, 0}
	if len(got) != len(want) {
		t.Fatalf("got %d rows, want %d", len(got), len(want))
	}
	for i, seq := range want {
		if got[i].Seq != seq {
			t.Errorf("row %d seq = %d, want %d", i, got[i].Seq, seq)
		}
	}
}
