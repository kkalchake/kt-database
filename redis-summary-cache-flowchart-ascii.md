# Redis Summary Cache — Full Logic Flowchart

The flowchart itself, readable in any markdown viewer or plain-text editor. No rendering required.

```
                    ┌─────────────────────────────────────────┐
                    │  UI request:                            │
                    │  (app_id, latest_index, notes[])        │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │  HGETALL summary:app:{app_id}           │  [REDIS]
                    │  (wrapped in try/except)                │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                              ◇──────────────────◇
                              │  Redis read OK?  │
                              ◇──────────────────◇
                          No /                    \ Yes
                            /                      \
                           ▼                        ▼
      ┌──────────────────────────────┐    ◇──────────────────────────◇
      │ INCR stats:summary:error     │    │  Entry exists AND        │
      │ Treat as cache miss          │    │  upto_index ==           │
      │ (writes become best-effort)  │    │  latest_index?           │
      └──────────────┬───────────────┘    ◇──────────────────────────◇
                     │                 Yes /                \ No
                     │                    /                  \
                     │                   ▼                    ▼
                     │   ┌───────────────────────┐  ┌──────────────────────────┐
                     │   │ INCR stats:summary:hit│  │ INCR stats:summary:miss  │
                     │   └───────────┬───────────┘  │ (or stale_miss if entry  │  [REDIS]
                     │               │              │  existed)                │
                     │               ▼              └────────────┬─────────────┘
                     │   ╔═══════════════════════╗               │
                     │   ║ RETURN CACHED SUMMARY ║               ▼
                     │   ║ (fast path — no LLM)  ║  ┌──────────────────────────┐
                     │   ╚═══════════════════════╝  │ SET lock:summary:app:{id}│
                     │                              │ {worker_id} NX EX 60     │  [REDIS]
                     │                              └────────────┬─────────────┘
                     │                                           │
                     │                                           ▼
                     │                                ◇────────────────────◇
                     │                                │   Lock acquired?   │
                     │                                ◇────────────────────◇
                     │                            Yes /                    \ No
                     │                               /                      \
                     │                              ▼                        ▼
                     │              ◇──────────────────────◇      ◇──────────────────────◇
                     │              │ Full rebuild?        │      │ Stale summary        │
                     │              │ (increment_count>=15 │      │ exists?              │
                     │              │  OR no cached entry) │      ◇──────────────────────◇
                     │              ◇──────────────────────◇   Yes /              \ No
                     │           Yes /              \ No          /                \
                     │              /                \           ▼                  ▼
                     ▼             ▼                  ▼   ╔═══════════════╗  ┌───────────────┐
        ┌──────────────────┐  ┌──────────────────────┐   ║ RETURN STALE  ║  │ Wait ~200ms,  │
        │ LLM input =      │  │ LLM input =          │   ║ SUMMARY with  ║  │ re-check      │
        │ ALL notes        │  │ old summary + notes  │   ║ refreshing:   ║  │ cache         │
        │ (full rebuild)   │  │ where index >        │   ║ true          ║  └───────┬───────┘
        └────────┬─────────┘  │ upto_index           │   ╚═══════════════╝          │
                 │            │ (incremental)        │                              ▼
                 │            └──────────┬───────────┘                    ╔═════════════════╗
                 │                       │                                ║ RETURN          ║
                 └───────────┬───────────┘                                ║ AVAILABLE       ║
                             │                                            ║ RESULT          ║
                             ▼                                            ╚═════════════════╝
                ┌─────────────────────────┐
                │ Call LLM:               │  [LLM]
                │ generate summary        │
                └────────────┬────────────┘
                             │
                             ▼
                   ◇───────────────────◇
                   │  LLM succeeded?   │
                   ◇───────────────────◇
               Yes /                   \ No
                  /                     \
                 ▼                       ▼
┌────────────────────────────────┐  ┌──────────────────────────────┐
│ Pipeline (best-effort — skip   │  │ Release lock                 │
│ on Redis error, never fail     │  │ (Lua: DEL only if            │  [REDIS]
│ the request):                  │  │  value == worker_id)         │
│  • HSET text,                  │  └──────────────┬───────────────┘
│    upto_index = latest_index   │                 │
│  • HINCRBY increment_count 1   │                 ▼
│    (reset to 0 on full rebuild)│      ◇──────────────────────◇
│  • EXPIRE 24h                  │      │ Stale summary exists? │
└────────────────┬───────────────┘      ◇──────────────────────◇
                 │                   Yes /                \ No
                 ▼                      /                  \
┌────────────────────────────────┐    ▼                    ▼
│ Release lock                   │  ╔═══════════════════╗ ╔══════════════╗
│ (Lua: DEL only if              │  ║ RETURN STALE      ║ ║ RETURN ERROR ║
│  value == worker_id)  [REDIS]  │  ║ SUMMARY labeled   ║ ║ TO UI        ║
└────────────────┬───────────────┘  ║ with its age      ║ ╚══════════════╝
                 │                  ╚═══════════════════╝
                 ▼
   ╔═════════════════════════════╗
   ║ RETURN FRESH SUMMARY TO UI  ║
   ╚═════════════════════════════╝
```

## Legend

```
┌───────┐  process step                ◇───────◇  decision (Yes/No branches)
╚═══════╝  terminal — response to UI   [REDIS]    Redis operation
                                       [LLM]      LLM call
```

Note: the "treat as cache miss" branch (Redis unavailable, far left) flows down and joins the **full rebuild** path — with no readable cache there is nothing to do incrementally, and the subsequent cache write is best-effort.

## The six terminals

| # | Terminal | When |
|---|---|---|
| 1 | Return cached summary (fast path) | Cache hit: `upto_index == latest_index` — no LLM call |
| 2 | Return stale summary, `refreshing: true` | Lock held by another worker; stale copy exists |
| 3 | Return available result | Lock held; no stale copy; brief wait then best available |
| 4 | Return fresh summary | Regeneration succeeded (the main miss path) |
| 5 | Return stale summary labeled with age | LLM failed; stale copy exists |
| 6 | Return error to UI | LLM failed; nothing cached to fall back on |
