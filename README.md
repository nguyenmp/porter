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
- Reconnect to a previous chat: `porter --session <id>` starts the REPL attached to that session — it replays the committed history and appends new turns to it instead of creating a new session. The one-shot path takes the same flag (`porter --session <id> "prompt"`, or `PORTER_SESSION`), and the session id is always printed on startup (`session <id>`).

### Handoff

- An **execution provider** runs code: the local laptop via an agent process, a cloud provider like AWS EC2/EKS, or nsjail/subprocess on the server.
- **Single-writer-per-session**. The server is the serialization point. When another client (e.g. phone) connects, it pulls old data from the DB but streams live from a bus.
  - Conversation → sync the DB + network calls
  - Filesystem → sync via git check pointing on a working branch
  - Processes are not sync'ed
### Execution context & skills

- The execution provider reports where it runs — system (so the model knows to
  use curl vs wget, or macOS BSD vs GNU userland), working directory, files
  there, and the skills it can load — when it connects. The server injects this
  as a system-message prefix on every request, so the model knows where commands
  will run without guessing. A multi-repo sandbox lists files from each repo,
  prefixed with the repo's directory name (`porter/go.mod`) and bounded per
  repo, so the model sees the shape of every repo it can work in.
- Skills live at `<root>/.agents/skills/*/SKILL.md` or
  `<root>/.*/skills/*/SKILL.md` (any hidden dir's skills subdir), found under
  the repo root (`git rev-parse --show-toplevel`) or the user root (`~/`),
  deduplicated by name (repo wins). They are exposed to the model as a single
  **load_skill** tool whose description lists every skill; calling it returns
  the full SKILL.md body. A multi-repo sandbox searches every sandboxed repo
  (each worktree is a repo root), so each repo's committed skills load.
- CLI tools the provider declares available ride along the environment: a
  curated manifest at `~/.porter/clis.json` (`{"clis": {"gt": "Graphite stack
  management (submit, split, restack, up/down)"}}`) lists what the model can
  run via the **shell** tool. If it's in the manifest it's assumed to exist;
  a missing binary surfaces as a normal "command not found". CLIs are a
  discoverability hint, not tools of their own.
- Provider status is real-time: an `exec_status` bus envelope and
  `GET /api/sessions/{id}/exec/status` show the active provider, where it runs,
  and every provider that could run this session's tools.
- **Choosing where commands run (web).** A session can have several providers
  connected at once — the server process itself (always available, listed as
  **local**) plus any execution clients (e.g. a REPL on a laptop). A bar below
  the chat shows where commands run; opening it lists every provider (name,
  system, working directory) and clicking one switches execution via
  `POST /api/sessions/{id}/exec/select`. A connecting client takes over
  automatically, matching the original behavior; after that you can switch
  freely — including back to a provider you switched away from, which stays
  connected (it simply receives no tool calls while deselected). Switching
  takes effect on the next message, and commits a short role-`system` notice so
  the model sees the environment change. A provider's environment context is
  injected fresh with each request, so the model always knows where commands
  will run.

### MCP (Model Context Protocol)

MCP servers are configured in `porter.mcp.json` (gitignored; see
`porter.mcp.json.example`). At startup the server fetches each configured
server's tool list (streamable HTTP transport, bearer-token auth) and caches
it in memory; a server that fails to respond is reported rather than blocking
startup. The model sees exactly two tools no matter how many servers or tools
are configured:

- **FindMCP** — its description lists every configured server (name,
  description, tool count, load status); calling it lists a server's tools
  (or all servers), filtered by an optional substring, with `full=true` for
  full descriptions and input schemas.
- **CallMCP** — calls one tool on one server and returns the text result. It
  supports cancellation like any other tool (aborting the HTTP request).

MCP calls run where the credentials live. Server-configured servers are
served by the server process and never cross the exec channel, so server-side
credentials stay server-side. An execution host (e.g. the laptop) can also
host MCP servers — typically ones only reachable from that machine, like a
corporate-VPN-only gateway — by configuring `~/.porter/porter.mcp.json` on
the host. The host reports its servers with its environment; the server lists
them in FindMCP (`hosted on <host>`) and routes a CallMCP for one down the
host's exec channel, where the host's own local hub executes it — so those
credentials never leave the host. A host-owned server is available while that
host is the session's active provider, matching how `shell` behaves on it.

Auth is either a static bearer token (`"auth": {"type": "bearer", "token":
"..."}`) or OAuth 2.0 (`"auth": {"type": "oauth"}`) for servers that only
accept OAuth (e.g. Retool's MCP). `scope` is optional — omit it to use the
server's default scopes, or set it (e.g. `"scope": "mcp:read"`) to request
least privilege. OAuth uses dynamic client registration and the
authorization-code flow with PKCE over an ephemeral loopback redirect: run
`porter mcp login <server-name>` on the machine that can reach the server —
it opens a browser, and stores tokens in `~/.porter/mcp/tokens.json` (0600).
The host daemon never opens a browser; it reads and refreshes stored tokens,
and `porter mcp logout <server-name>` revokes them. Only tools are used;
resources and prompts are ignored.
Connections are stateless (no persistent stream), so
`notifications/list_changed` and `ping` are never received; the only
per-server state kept is the streamable-HTTP session id. The legacy SSE
transport is future work.

### Cost & metrics

- Estimated cost per message, per turn, per chat, per subagent — before sending, based on input cost, including resume-from-history
- Show both cached and uncached estimates where behavior is hard to predict; pull prices from upstream (OpenRouter, etc)
- Keep the hidden thought token out of resubmission
- Trim large tool call output in the model's view only: a tool result larger than 1.5 KB is sent to the model as a head + tail slice with a size header, and a `read_output(call_id, offset, max_bytes)` tool loads any byte window of the original (omit `max_bytes` to read the rest in one call). The full output is still stored in History and rendered in the UI — frontend and DB hold the full data; only the model's context is trimmed. A `tool_output` metadata field (total_bytes, shown_bytes, truncated, recall) rides on the committed message and the bus so the UI can render a badge, and `read_output` results are persisted as a short placeholder so the window bytes are never duplicated in the database. *(Future: drop full output from History once the model has read it, with `read_output` served from the DB.)*
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

### Archiving sessions

The web sidebar keeps every session in one list, newest first. To keep the
list focused on what's still relevant, the chat page carries an **Archive**
button (top-right, always visible) for the current session. Archiving folds the
session into a collapsible **Archived** folder below the active list — collapsed
by default, with a count in its header so the folder's contents stay
discoverable. It is purely organizational: history, running turns, and
streaming are unaffected, and the session survives restarts either way.

Chatting with an archived session pulls it out of archive automatically:
sending any message unarchives it server-side (the web UI, REPL, and scripts
all get this for free, since they share the append path), so an old session
you come back to quietly returns to the active list. The backend surface is
`POST /api/sessions/{id}/archive` and `POST /api/sessions/{id}/unarchive`, both
idempotent.

### Cancelling a running task

Long-running tools (tests, builds, `tail -f`) render live in the web UI with an
elapsed timer, and each running tool block carries a **Cancel** button. Clicking
it stops the command and ends the turn: a locally-run tool's whole process group
is killed, a remotely-run tool's connected execution client is signalled to stop
its command, and the partial output is committed to history marked *cancelled*
so a reload (and the model, on the next turn) sees the run was aborted rather
than completed. The backend surface is `POST /api/sessions/{id}/cancel/{call_id}`.

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
- **View** — the projection of History a consumer is given. The model's view trims large tool results to a head + tail slice (with a `read_output` recall tool to expand back); the user's view is the full human-readable form. Other trims and summaries are future work.

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

- **Execution provider** — where a session's commands run. The local server
  process (always available) or a connected execution client, e.g. the REPL on
  a laptop. An **execution host** (`make host`) is a persistent agent that can
  provision these per-chat: given one or more repo paths it creates a sandbox
  container holding a git worktree per repo (each its own branch) so multiple
  chats can work on the same repos independently — and a chat can work across
  several repos at once, or the same repo twice on different branches to
  compare them — and serves that sandbox as the chat's provider. A session can have several connected at once; one is **active** and
  receives the tool calls. The web picker switches the active provider; a
  deselected client stays connected, so it can be picked again without
  reconnecting.
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
- [x] Web UI (render from history poll + event bus, send commands)
- [x] SQLite-backed history (persist the server-owned session state)
- [ ] Handoff / async execution
  - [x] Execution context selector for the web UI (choose where commands run:
        local server or any connected client; switching takes effect on the
        next message)
  - [x] Execution Host (`make host`): a persistent agent on a machine that
        provisions a per-chat sandbox — a working directory, or git worktrees
        on one or more shared repos (each chat gets its own branch per repo) —
        and serves it as that chat's execution provider; archiving a sandboxed
        chat releases its worktrees
- [ ] Metrics & performance (tokens/sec, tool timing, worktree cache)
- [x] Tool output trimming (`tool_output` head+tail model view, `read_output` recall) — full output kept in History/DB, only the model view trimmed
- [ ] Token budget before send
- [x] Dynamic MCP

**As-needed (ad hoc, when the need arises):**

- Forks / resume — tree view of the conversation, click to fork or rewind
- Cost / token display — per-message totals (per-turn and per-session already rendered)
- Memory
- TODO
- [x] Skills (discovery + load_skill)
- Auth — single-user token gating the whole UI
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
