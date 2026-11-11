Could you write a summary using plain-language? You're handing this off to someone else to pick up where we left off.

Write for the person who picks the work up next, not for the person who did
it. They did not sit in this session and cannot read the conversation.
Everything they know about the work comes from your summary.

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

1. What we were doing, and why.
2. Where things stand: what is done, what is left to do, and what we already
   decided. Say when something is unverified, failing, or blocked.
3. Where the work lives: the repo and remote URL, branch, commit SHA, PR link,
   and anything still only in the sandbox. Name the commands that matter
   (build, run, test).
4. The next step: the one thing the next person should do first.
5. Traps: what looks finished but is not, such as unpushed commits, failing
   checks, or a step that only works in one order.

## How to write it

Follow the plain-language rules. Load the plain-language skill for the full
list.
