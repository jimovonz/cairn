# Code Graph Navigation

Every project you work in may have a pre-built code graph (`.code-review-graph/graph.db`) maintained by `code-review-graph`. The `cairn-graph` CLI queries it with zero LLM cost and is faster and structurally aware compared to grep or file reads.

## When to use cairn-graph instead of grep or Read

Prefer `cairn-graph` for:
- Finding where a symbol is defined (instead of `grep -rn SYMBOL`)
- Finding who calls a function or what a function calls
- Estimating blast radius before an edit
- Getting full context (body + callers + tests) for a symbol before changing it

Read a file when you need the actual implementation or full file content.

## Commands

```
cairn-graph --location SYMBOL         # where SYMBOL is defined (file:line)
cairn-graph --callers SYMBOL          # functions that call SYMBOL
cairn-graph --callees SYMBOL          # functions SYMBOL calls
cairn-graph --impact SYMBOL           # one-line blast radius (callers:N tests:M files:F)
cairn-graph --context-pack SYMBOL     # body + callers + tests + related memories
cairn-graph --tests SYMBOL            # tests that cover SYMBOL
cairn-graph --file-context FILE       # all symbols, signatures, fan-in/out, risk tail for FILE
cairn-graph --summary                 # repo-level module/flow/hub summary
cairn-graph --orientation             # same as --summary, alias
```

`cairn-graph` is installed in the cairn venv — it is resolved automatically by name.

## When the graph is active

- Session start includes a `<code_graph_orientation>` block if the repo has a built graph.
- Before each Read/Edit, a `CAIRN GRAPH for <file>` block appears with structural context.
- If neither appears, run `cairn-graph --summary` to check availability.
- When you grep for a code symbol and a graph is present, a one-time reminder will fire.

## Graph coverage

Not all repos have a graph — it is built on first session contact (background) and refreshed hourly. If `cairn-graph --summary` returns nothing, the graph is still building or the repo has not been indexed yet.
