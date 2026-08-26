# porter

**porter** is [Mark Nguyen](https://github.com/nguyenmp/)'s ideal LLM agent, and an experiment to improve agentic workflows.

I want an agent that is:

- **Simple:** single-user, minimal dependencies, fast, minimal configuration, sane defaults, no LLM jargon.
- **Everywhere:** chat from any device, and run code on any backend. Start a task on my laptop, then walk away — keep sending messages from my phone, and hand execution to a server or the cloud when the laptop is off. This is the killer feature: fluid handoff between local, remote, and cloud execution.
- **Transparent:** you see what an agent did and what the model actually received (memories, notes, todo), with realtime feedback on what's being sent or waited on. Nothing is magic — the full toolset is visible, nothing is snuck into context, and it never waits on tools you didn't ask for. You can even keep chatting with a subagent after it finishes.
- **Efficient:** runs quickly and cuts token cost in deliberate ways, then shows you the cost up front, can be run on a tiny VPS

Above all, it works with my use cases: work and personal, coding and research.

## Features

### Conversations

A tree of turns. Each turn has a single parent and can fork into many children.

- A `session_id` defines the top level chat, whereas a `turn_id` represents a node in the chat tree.  Thus, loading a `session_id` shows the chat leading to the most recent turn.  You can operate on turn_ids like forking, or direct-linking.
- Forking or rewinding only affects the conversation, never the world state (files, network calls, side effects).
- Link and share a specific conversation, fork, or turn.

### CLI interface

- With no TTY, output as JSONL.
- TTY REPL mode: stdout shows input/output, stderr shows JSONL. Fork by targeting an old `turn_uuid`.

### Handoff

- An **execution provider** runs code: the local laptop via an agent process, a cloud provider like AWS EC2/EKS, or nsjail/subprocess on the server.
- **Single-writer-per-session**. The server is the serialization point. When another client (e.g. phone) connects, it pulls old data from the DB but streams live from a bus.
  - Conversation → sync the DB + network calls
  - Filesystem → sync via git check pointing on a working branch
  - Processes are not sync'ed

### Cost & metrics

- Estimated cost per message, per turn, per chat, per subagent — before sending, based on input cost, including resume-from-history
- Show both cached and uncached estimates where behavior is hard to predict; pull prices from upstream (OpenRouter, etc)
- Keep the hidden thought token out of resubmission
- Drop full tool call output from chat history after the model has already read it. Keep only a `tool_output` field (full | head | tail | summary | omit) with size metadata (bytes/lines), plus a `read_output` tool to show it again when needed.
- Concurrent compaction to reduce chat history, with long term recall like `read_output` but for memory

### Performance

- **git worktree is slow** on my giant repo (GBs). Want a hot-cache of worktrees, a faster approach like a filesystem clone, or an alternative to worktrees.
- Tokens/sec (encode and decode), tool run time, output size, time to first/last token, time spent connecting
- Should not use a boatload of RAM or CPU -- I'm offloading so much work to GPUs in data centers

## Building & running

We build and run through Docker so the pinned Go version is used for portable development and runtime

1. `cp .env.example .env` and set the env vars.
2. `make test` / `make vet` to verify
3. `make server` — starts the server (owns the LLM connection + tool execution) in one terminal
4. `make repl` — starts the interactive REPL client in another, or `make run PROMPT="say hi"` for one-shot

### Persisted sessions

The server owns conversation state, and since the SQLite-backed history roadmap item it persists every session — its creation time, full committed history (including tool timing, reasoning, and tool calls), and the bus position clients resume from — to a SQLite database at `porter.db` in the working directory. When running via `make server` (which mounts the repo at `/app`) or `porter-macos` from the repo, the file lands in the repo and is gitignored, exactly like `porter.log`. Because committed state lives in the database, sessions survive server restarts: a restarted server loads them back and the sidebar, `/view`, and SSE resume all keep working.

### Quiet REPL logs

In the REPL the human-readable conversation goes to stdout while the structured JSONL event stream (plus progress lines) goes to stderr. When run interactively through a container, both land on the same terminal and the JSONL can get noisy. Set `PORTER_LOG=/path/to/porter.log` in `.env` to redirect that stream to a file instead — the REPL then only prints the conversation. When running via `make shell` (which mounts the repo at `/app`), a relative path like `porter.log` is written into the repo and gitignored.

## Why I'm building this

I'm tired of my existing agents. I juggle three styles:

- **Local terminal** (claude-code / OpenCode)
- **Cloud chat** (claude.ai / Open WebUI)
- **Cloud agents** (internal)

General pain points:

- TUIs are garbage UIs
- Weird defaults and configuration are annoying. Open WebUI won't add Web Search MCP to your default chat — you must add it to each model individually. It doesn't default to Native Function Calling, so I have to remember to enable it every time.
- No dynamic MCP. Both tools keep static lists of MCPs. I have to add each MCP to both OpenCode (local) and OpenWebUI (cloud). Add too many and OpenCode defaults all of them on, bloating context. Open WebUI lets you toggle manually, but I'd rather the system figure out which tools are relevant dynamically, or let me force them on or off.  Only claude-code has dynamic MCP support.
- They're slow. Opencode takes 3 seconds to launch
- Claude hides what's happening — I want to see commands streaming.
- Claude.ai is too locked down, often hiding the thoughts and web fetches
- I want to jump to a previous message or view a tree of all the forks I took
- Token usage and cost should be upfront
- Open WebUI doesn't resume properly on mobile when backgrounded and reopened
- Open WebUI sometimes times out resolving my MCPs
- Claude's local skills aren't synced to the cloud automatically
- Open WebUI frequently hard-refreshes to an empty chat when switching between chats
- I'm happy to use Claude because my work pays for it, but I would never personally
- Open WebUI idles at 512MB RAM, using 25% of my VPS which is already beefy for a hobby.  They recommend 1-4GB RAM, wtf.  It even uses 5% of my CPU while idle, constantly.
- [Open WebUI docs are pure AI slop](https://docs.openwebui.com/troubleshooting/performance/#4-sqlite-memory-footprint-on-constrained-containers).  Also a lot of these configurations should be automatic, the app can self-detect the conditions that require these settings to be changed.

## Technical decisions

- **Tech stack:** Go with Chi server + HTMX directly, SQLite (via the pure-Go `modernc.org/sqlite` driver, keeping the static build). Minimal dependencies.
- **Models:** support OpenAI-compatible APIs. I already run a LiteLLM proxy and have OpenRouter access.
- **Permissions:** YOLO for now.
- **Auth:** only when needed; single-user.

## Glossary

Terms used throughout this README, grouped by the part of the system they belong to.

### The model (what the LLM produces and sees)

- **Event** — purely what the LLM produced: streamed text, reasoning, a finished message, token usage, and the tool calls it requests. Real-time and ephemeral.
- **Delta** — a streamed chunk of message or reasoning text (`message_delta` / `reasoning_delta`).
- **View** — *(planned)* the projection of History a consumer is given. The model's view can be trimmed or summarized (with recall tools to expand back); the user's view is the human-readable form. Today the model receives History directly.

### The conversation (shared vocabulary)

- **Conversation / Session** — one chat, the top-level unit of state (`session_id`). Owns the history and the pacing; the server is its single writer.
- **Message** — one entry in a conversation: what you said, what the assistant said, a tool call, or a tool's result. The building block of history.

### The agent loop (the turn engine)

- **Turn** — from one thing you say to the model's final answer; can span several LLM round-trips (asking for a tool, seeing the result). Bounded by a `turn_completed` marker.
- **Turn id** — identifies a turn; the future fork/rewind target (`turn_id`/`turn_uuid`).
- **Tool result** — a system-side fact: the agent ran a tool and got this outcome. Not from the model.

### The server (state + bus)

- **History** — the raw, append-only, full-fidelity record of everything committed in a session, persisted to SQLite. Ground truth; never rewritten in place.
- **Queue** — the server's per-session to-do list of user messages; worked through one turn at a time, in order.
- **Bus / event log** — the ordered stream a subscriber watches. Committed messages and turn markers are logged for replay in the same order History is built; live Events and tool results stream on top, real-time only.
- **Envelope** — one line on the bus; the union of everything a subscriber can receive: an Event, a tool result, a committed message, a turn completion, or a resync signal.
- **Position (seq)** — the counter on the bus; a client says "I've seen up to here" and resumes exactly there.
- **Commit** — the server appending a message to History (and the bus) the moment it's produced.

### Clients (rendering + commands)

- **Command** — what a client sends instead of its own copy of history: create a session, append a user message. *(Planned: stop, fork.)*
- **Poll / Subscribe** — poll = fetch History; subscribe = watch the bus for new lines.
- **Resync** — the bus telling a client its position is too old; refetch History and resubscribe.

## Roadmap

**Where I start:** a minimal self-coding agent, used to improve itself
**Where I'll end:** a fast, efficient web UI

Build order:

- [x] 1. One-shot CLI (JSONL output)
- [x] 2. TTY REPL
- [x] 3. Tool-call shell (file edits + network calls)
- [x] 4. Server-owned conversation state (stateless clients poll history and subscribe to an event bus; commands instead of full-history resubmission)
- [ ] Web UI (render from history poll + event bus, send commands)
- [x] SQLite-backed history (persist the server-owned session state)
- [ ] Handoff / async execution
- [ ] Metrics & performance (tokens/sec, tool timing, worktree cache)
- [ ] Token cost management (`tool_output` trimming, `read_output`, budget before send)
- [ ] Dynamic MCP

**As-needed (ad hoc, when the need arises):**

- Forks / resume
- Memory
- TODO
- Skills
- Auth
- Share / link (conversation, fork, turn)
- Chat with a subagent after it finishes
- Subagents

**Non-goals:** IDE integration, mobile apps, native desktop apps, advanced TUI, plugins.

## Inspiration

- [Building an Advanced Agentic Harness](https://data4sci.com/blog/building-an-advanced-agentic-harness) — inspiration, not a roadmap. Aiming for simple and fast, I'll add complexity as I need it.
- **Dirge** — interesting but missing fundamentals (skills)
- **OpenCode** — shows context budget and cost; UI could improve
- **Claude Code** — started it all
- **Cursor** — easy desktop/cloud/IDE switching; Elon makes me sad
- **Pi** — fast but doesn't run in the cloud
- **Open WebUI** — okay, but settings are complicated and slow to save; some rough 0.9.0–2 versions

### For me to read

- [ ] [TUIs Are an Abomination: Rethinking AI Interfaces](https://streamzero.com/blog/posts/deep-dives-tools-technologies-architectures/ux-for-ai)
- Blake Crosley
  - [ ] [Chat Is the Wrong Interface for AI Agents](https://blakecrosley.com/blog/chat-is-the-wrong-interface)
  - [ ] [Agents Need Supervision Surfaces](https://blakecrosley.com/blog/agents-need-supervision-surfaces)
  - [ ] [Agentic Design Is Control Surface Design](https://blakecrosley.com/blog/agentic-design-control-surface)
- [ ] [The Terminal: A 1970s UI for a 2026 Problem](https://nimbalyst.com/blog/the-terminal-is-a-1970s-interface-for-a-2026-problem/)
- [ ] [Why I moved coding-agent work out of the terminal](https://dev.to/dilless/why-i-moved-coding-agent-work-out-of-the-terminal-bb0)
- [ ] [I built a clean Web UI for Claude Code agents because the terminal was killing me](https://dev.to/ngxba/i-built-a-clean-web-ui-for-claude-code-agents-because-the-terminal-was-killing-me-rn-46fk)
- [ ] [Terminal Black Screen vs. Desktop Client](https://clawlite.ai/blog/terminal-black-screen-vs-desktop-client-for-autonomous-agents)
- Hacker News
  - [ ] [Ask HN: Why are Gemini CLI and Claude Code TUIs so terrible?](https://news.ycombinator.com/item?id=46286057)
  - [ ] [I don't get why terminal agents are so popular of late](https://news.ycombinator.com/item?id=44736541)
  - [ ] [I find it strange how most of these terminal-based AI coding agents have ended up with flashy TUIs](https://news.ycombinator.com/item?id=44737008)
  - [ ] [Ask HN: GUI or TUI for Coding Harness?](https://news.ycombinator.com/item?id=47771230)
- [ ] [Pickuma's review of Claude Code's Agent View](https://pickuma.com/for-dev/claude-code-agent-view-review/)
- [ ] [Desktop GUI vs Terminal TUI: how I choose](https://kunpeng-ai.com/en/blog/gui-vs-tui-ai-coding-agent-workflow/)
- [ ] [forgecode's discussion on TUIs](https://github.com/tailcallhq/forgecode/discussions/2506)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms.

## License

This project is licensed under the GNU Affero General Public License v3.0 — see the [LICENSE](LICENSE) file for details.
