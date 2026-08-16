# porter

I'm tired of my agents.  I have three styles of agents:
* Local terminal (claude-code)
* Cloud Chat (on my phone)
* Cloud Agents (Please do some work )

General pain points:
* I think TUIs are garbage UIs
* Weird defaults and configuration is annoying.  Open WebUI you can’t add Web Search MCP to your default chat, you must add it to individual models, one at a time.  Makes trying new models annoying.  Also doesn’t default to Native Function Calling.  Super annoying to remember to enable it each time.
* No dynamic MCP.  Both have static lists of MCPs available.  Means you need to add an MCP to both OpenCode for local and OpenWebUI for cloud.  But also if you add a bunch, OpenCode defaults to on which bloats the context.  Open WebUI is manually checked on or off (with some defaults) but that’s annoying, I wish the system dynamically figured out which tools are relevant and added them, or I could force on or off.
* They're slow.  Opencode takes 3 seconds to launch.  Claude really likes to hide what's going on under the hood.  I want to see commands streaming
* Claude is too locked down (can't fetch stuff)
* I want to jump to my previous message, or view a tree of all the forks I took
* Token usage and cost should be upfront
* Open WebUI doesn't resume properly on mobile when backgrounded and then reopened
* Open WebUI sometimes times out resolving my MCPs
* Claude does not sync my local skills aren't sync'ed to the cloud automatically
* Open WebUI frequently hard-refreshes to an empty chat when switching between chats

## Goals

bootstrap self-coding agent ASAP

Build a simple agent (cli chat, then toolcall shell for editing files and doing network calls) and expand as needed.  End goal is a local web UI that's fast and efficient

Tech Stack: Go w/ Chi server + HTMX directly, sqlite (but start with jsonl at first), minimal dependency

Support OpenAI-compatible API -- I already have LiteLLM proxy running and have access to OpenRouter.

Eventually my biggest goal is handoff between local CLI and web server w/ synced settings, config, history, skills, MCPs, ability to run agents locally and then async them onto the server maybe letting it run commands in a jail or container, or start on server and continue in local laptop.  For example, I might start a task on my laptop, then walk away (locked but not shut down) to grab some snacks.  I want to open my phone and check the status and issue more messages as needed.  These should be able to execute local shell toolcalls on my laptop, but the message is initiated from my phone.  Alternatively, if my laptop is turned off, I'd like to be able to run commands on the server or in AWS or something.  This gets complciated if I want to work with a repository (or multiple repositories) because I'll need to sync the dirty state as work is done.  Perhaps this should be done by git committing and pushing a working branch?  I think overlal, this means we need to have an "execution provider" that gives execution ability.  That could be my local laptop (perhaps through an agent?), a cloud provider like AWS EC2 or EKS that I give access to, or nsjail/subprocess on my server (risky?).  Handoff also requires single-writer-per-session or else we get total ordering of events wrong.  The server can be the serialization point, with a mutex or DB write lock or something.  When another client like my phone connects, we pull the old data from DB, but stream live from a bus, not polling the DB.  Also, syncing the chat history and syncing the filesystem are separate.  We cannot sync processes.  Chat history is easy, we sync the DB and make network calls, but syncing the file system will be through git or something, maybe checkpointing.

Estimated cost before sending a message would be cool.  Just purely based on input cost, rather than output cost.  Then each turn could have a cost attribution, so that each user sent message can show how muhc cost was accured due to all the agent work in response.  Then the chat can have a cost, as well as split by subagents.  We can estimate with char_count / 4 = token_count, and solidify with usage response key.  Source prices from upstream (OpenRouter, etc).  Note this excludes turns cause it’ll probably be cached, and excludes future tool calls cause that’s unpredictable.  We’re just looking at initial payload cost, or the cost of resuming a chat which has a lot of history.  Also must be differentiated by model/provider since each has a different cache and TTL.  Since it may be hard to predict cached vs uncached behavior, we should show both, as well as a predictor if it's cached or not based on model + provider + time since last message.

Permissions, yolo for now.

Also subagents can be chatted with after they're done.  That's not a thing right now.

MCP later.

I'd like to expose performance metrics like tokens/sec upload, download, how long tools take to run, output size.  Time to first token, time to last token, time spent connecting, including tool calls.

Log everything, make the inner workings discoverable, not magic.  Make the UI transparent so things are discoverable.  For example, being able to just see the full toolset.  I'm sad when "memory" and "notes" are automatically included and it messes up my context, where I want something clean.  Or when it's waiting on enabling an MCP when I don't need it.

Auth when needed, but single-user only.

First step is a one-shot CLI that takes a prompt and runs it with a response.  The CLI with no TTY outputs as jsonl.  A next iteration would output session_id (generated or provided) as well as a turn_uuid. Then subsequent runs can use the same session_id to resume from the most recent turn_uuid, or the user could specify a specific turn_uuid to “fork” the conversation. This inheritly follows the tree method.  This is the basis of a conversation.  Then we could support a TTY mode that acts as a REPL: stdout shows the input and output, stderr shows jsonl updates.  This supports forking by targeting an old turn_uuid and continuing from there.  So any parent can have multiple descendants, but each descendant has a single parent.  Forking/rewinding only affects the conversation, not the world state (files, network calls, other side-effects).

Link and share to a specific conversation/fork/turn

Optimizations:
* remove thought token from being resubmitted?
* remove tool call output after and store it in a file? remove it from the chat history, because the model has already read it and thought about in it’s turn. So when the user sends a new message, it’ll rewrite the chat history without the full tool outputs.  Add a field tool_results: full | head_tail | summary | omit with metadata on size (bytes, lines) to hint at what’s happening, and add a read_output tool call to re-show the output as needed.
* Add timestamp to messages?
* (Breaks cache in the short term but keeps cache in the long term)
* concurrent compacting (checkpointing) 
* git worktree is hella slow (Claude and git) on my giant repo (GBs).  I'd be nice if there was a hot-cache of worktrees, or a faster way to do worktrees like FS clone instead, or do something else that isn't worktree if it's too slow

Non-goals: IDE integration, mobile apps, advanced TUI, plugins
## Inspiration

https://data4sci.com/blog/building-an-advanced-agentic-harness <- inspiration, not roadmap. I'm aiming for simple and fast, not complicated.  I'll add complicated as I need.

Dirge very interesting but lacks a lot of fundamentals I need to do my job, like skills.

OpenCode shows context budget and $-cost, but the UI could be improved.

Claude Code started it all.  

Cursor easily lets you jump between desktop, cloud, and the IDE.  Elon makes me sad.

Pi I hear is fast but doesn't run in cloud.

Open WebUI is okay.  Settings are so complicated.  I sometimes save settings and it takes forever.  There were some really bad versions in 0.9.0-2 where a lot of stuff was just broken.

For me to read:
- [ ] [TUIs Are an Abomination: Rethinking AI Interfaces](https://streamzero.com/blog/posts/deep-dives-tools-technologies-architectures/ux-for-ai)
- [ ] Blake Crosley
	- [ ] [Chat Is the Wrong Interface for AI Agents](https://blakecrosley.com/blog/chat-is-the-wrong-interface)
	- [ ] [Agents Need Supervision Surfaces](https://blakecrosley.com/blog/agents-need-supervision-surfaces)
	- [ ] [Agentic Design Is Control Surface Design](https://blakecrosley.com/blog/agentic-design-control-surface)
- [ ] [The Terminal: A 1970s UI for a 2026 Problem (nimbalyst.com)](https://nimbalyst.com/blog/the-terminal-is-a-1970s-interface-for-a-2026-problem/)
- [ ] [Why I moved coding-agent work out of the terminal (ArchCode, dev.to)](https://dev.to/dilless/why-i-moved-coding-agent-work-out-of-the-terminal-bb0)
- [ ] [I built a clean Web UI for Claude Code agents because the terminal was killing me (dev.to)](https://dev.to/ngxba/i-built-a-clean-web-ui-for-claude-code-agents-because-the-terminal-was-killing-me-rn-46fk)
- [ ] [Terminal Black Screen vs. Desktop Client (ClawLite)](https://clawlite.ai/blog/terminal-black-screen-vs-desktop-client-for-autonomous-agents)
- [ ] Hacker News:
	- [ ] [Ask HN: Why are Gemini CLI and Claude Code TUIs so terrible?](https://news.ycombinator.com/item?id=46286057)
	- [ ] [I don't get why terminal agents are so popular of late](https://news.ycombinator.com/item?id=44736541)
	- [ ] [I find it strange how most of these terminal-based AI coding agents have ended up with flashy TUIs](https://news.ycombinator.com/item?id=44737008)
	- [ ] [Ask HN: GUI or TUI for Coding Harness?](https://news.ycombinator.com/item?id=47771230)
- [ ] [Pickuma's review of Claude Code's Agent View](https://pickuma.com/for-dev/claude-code-agent-view-review/)
- [ ] [Desktop GUI vs Terminal TUI: how I choose (Kunpeng AI Lab)](https://kunpeng-ai.com/en/blog/gui-vs-tui-ai-coding-agent-workflow/)
- [ ] [forgecode's discussion on TUIs](https://github.com/tailcallhq/forgecode/discussions/2506)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms.

## License

This project is licensed under the GNU Affero General Public License v3.0 — see the [LICENSE](LICENSE) file for details.
