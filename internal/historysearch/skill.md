# Searching the chat history

Every conversation this server has held is in `/data/porter.db`, a SQLite
database whose message text carries a full-text index. Load this skill when the
user remembers something from an earlier chat: a decision, a phrase they used,
an error a command printed. They usually recall the idea and not the exact
words, so try a second phrasing before you report that nothing matched.

## Read only

Open the file with `-readonly`. It is the live database behind the chat you are
in: a write or a stray `.dump` can damage it. Never run UPDATE,
DELETE, INSERT, DROP, VACUUM, or load it without `-readonly`.

```sh
sqlite3 -readonly /data/porter.db "<SQL>"
```

If the shell says `/data/porter.db` does not exist, this chat's commands are not
running on the server. Stop searching and tell the user to switch the chat's
execution environment to local, which is where the database lives.

If `-readonly` reports `unable to open database file`, the server is not running
or its lock files are missing. Report that and stop; never retry without
`-readonly`.

## Schema

```
sessions(id, created_at, name, archived_at_epoch_ms)
  one row per chat. created_at is epoch milliseconds. name is the custom title
  and is often empty, so fall back to the chat's first user message.

messages(session_id, seq, role, content, ...)
  every message in order. role is user, assistant, tool, or system. content is
  the text; in a tool message it is the command's output, which can be
  enormous, so never select it without a filter and a limit. The reasoning,
  images, tool_calls, and token columns are not worth reading.

messages_fts(content)
  the full-text index over messages.content. Join it back by rowid.
```

The other tables (queries, variants, session_sandboxes, providers) are
bookkeeping and rarely answer a question about what was said.

## The query

```sh
sqlite3 -readonly /data/porter.db "
select 'session_' || s.id as chat,
       coalesce(nullif(s.name, ''), substr(coalesce((
         select m2.content from messages m2
         where m2.session_id = s.id and m2.role = 'user'
         order by m2.seq limit 1), ''), 1, 80)) as title,
       datetime(s.created_at / 1000, 'unixepoch') as day,
       m.role,
       snippet(messages_fts, 0, '[', ']', '…', 12) as hit
from messages_fts
join messages m on m.rowid = messages_fts.rowid
join sessions s on s.id = m.session_id
where messages_fts match '\"caddy\"' and m.role in ('user','assistant')
order by bm25(messages_fts)
limit 20;"
```

Replace `'\"caddy\"'` with the search terms. `bm25()` ranks the closest match
first, and `snippet()` returns about twelve words around it with the hit in
brackets. Drop columns you do not need.

## Writing the search terms

- Quote each word and join them with a space to require all of them:
  `'\"caddy\" \"tls\"'`.
- Put the words inside one pair of quotes to match them as a phrase:
  `'\"caddy rule\"'`.
- Add `*` after a quoted word to match a prefix, which is what finds a
  half-remembered word: `'\"cadd\"*'`.
- Double an inner quote, and quote or leave out `(`, `-`, `:`, and `NEAR`:
  unquoted, they are syntax, and a malformed query is an error rather than a
  miss.

When nothing matches, drop the least certain word, or try a prefix. When a
query fills all twenty rows, add words rather than raising `limit`.

## Reporting a hit

Name the chat `session_<id>` and include the link `/?session=session_<id>`, so
the user can open it. Quote a line of the snippet and give the day it was
written.

Add `'tool'` to the role filter when the answer is probably in a command's
output. To read more around a hit, select the neighbors:

```sh
sqlite3 -readonly /data/porter.db "
select seq, role, substr(content, 1, 800) from messages
where session_id = 1675 and seq between 40 and 60 order by seq;"
```
