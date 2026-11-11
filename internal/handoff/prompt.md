Could you write a summary using plain-language? You're handing this off to someone else to pick up where we left off.

Write for the person who picks the work up next, not for the person who did
it. They did not sit in this session and cannot read the conversation.
Everything they know about the work comes from your summary.

The repo shows what we did. The conversation shows why. Carry over what the
next person cannot look up. Skip what they can read in the code, the docs, or
the ticket.

## The sandbox goes away

Porter releases the sandbox when the chat is archived or the environment is
rebuilt. The next person may be on another machine. Save what would be lost
before you write:

- Uncommitted work: run `git status` in each repo, list it, and propose the
  commit and push. Ask before you push: the repo may be public.
- A repo with no remote: offer a patch or bundle.
- What git can't carry: installed tools, servers, data, credentials, background
  jobs, `.porter/` files, and asset links, which expire with the session. Give
  steps to rebuild, not paths.

## What to cover

1. Why we did this: the goal, who it serves, and what done looks like. Give
   the reason, not just the task: it tells the next person what to protect
   when something has to give.
2. What we settled: every requirement and constraint someone stated, each
   with its reason. A requirement without a reason gets dropped at the first
   obstacle. Also say what we rejected and why, who asked for each so the
   next person knows what can still change, and what is still open.
3. Where things stand: what is done, what is left, and what is unverified,
   failing, or blocked.
4. Where the work lives: the repo and remote URL, branch, commit SHA, PR link,
   and anything still only in the sandbox. Name the commands that matter
   (build, run, test).
5. The next step: the one thing the next person should do first.
6. Traps: what looks finished but is not, such as unpushed commits, failing
   checks, or a step that only works in one order. Mention a dead end only if
   the next person would mistake it for progress.

## How to write it

Follow the plain-language rules. Load the plain-language skill for the full
list.
