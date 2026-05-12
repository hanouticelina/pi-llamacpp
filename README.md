# pi-llamacpp

A [pi](https://github.com/badlogic/pi-mono) extension that exposes your local
`llama.cpp` cache as a `llama-cpp` provider inside pi. From pi's perspective it
looks like any OpenAI-compatible cloud provider — same `/model` picker, same
chat flow — except the "API" is a `llama-server` running on `127.0.0.1`.

## Run it

```bash
brew install llama.cpp                              # llama-server binary
npm install -g @mariozechner/pi-coding-agent        # pi itself

# Seed the cache with at least one GGUF (any of these work):
llama-server -hf unsloth/Qwen3-30B-A3B-GGUF:Q8_0    # downloads + exits with Ctrl-C
# or hf download <repo> <file> into ~/Library/Caches/llama.cpp

# Run pi with this extension:
pi -e git:github.com/hanouticelina/pi-local
#  or: pi install git:github.com/hanouticelina/pi-local   (one-time)
#      pi                                                 (subsequent runs)
```

Inside pi: `/model` → pick a `llama-cpp` entry → first message loads the GGUF
and streams. `/model` again to swap; the server stays up.

### Environment variables

| Var | Default | Effect |
| --- | --- | --- |
| `PI_LLAMACPP_MODELS_DIR` | platform default (see below) | Directory `llama-server --models-dir` scans for GGUFs. |
| `PI_LLAMACPP_CONTEXT` | `32768` | Server-wide `-c` ctx-size cap. Larger native contexts are clamped. |

Platform default for `PI_LLAMACPP_MODELS_DIR`:

- macOS: `~/Library/Caches/llama.cpp`
- Linux: `$XDG_CACHE_HOME/llama.cpp` (fallback `~/.cache/llama.cpp`)
- Windows: `%LOCALAPPDATA%/llama.cpp`

This is llama.cpp's own cache — populated automatically by `llama-server -hf …`.

---

## How it works

### How pi loads the extension

`package.json` declares:

```json
"pi": { "extensions": ["./src/index.ts"] }
```

Pi reads this on startup and calls the default export — `pilocalExtension(pi)`
from `src/index.ts` — passing in an `ExtensionAPI` for registering providers
and subscribing to events.

### Boot sequence (`src/index.ts`)

1. **Reap orphans** (`llama.ts:35`). If a prior pi run was `kill -9`'d, our
   process-exit hooks never fired and a `llama-server` is still holding RAM
   and the cache port. We persist its PID to `~/.cache/pi-llamacpp/server.pid`
   at spawn time and SIGTERM it on next launch.
2. **Resolve the models dir** — `$PI_LLAMACPP_MODELS_DIR` wins, else the
   platform default.
3. **Spawn one router server** (`llama.ts:172`):
   - Find a free ephemeral port (open `:0`, read back, close).
   - `spawn("llama-server", ["--models-dir", dir, "-c", ctxCap, "--host",
     "127.0.0.1", "--port", port, "-ngl", "999", "--spec-default", "--temp",
     "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.00"])`.
   - Write the child's PID to disk; register it in `liveChildren`; install
     signal handlers (`SIGINT/SIGTERM/SIGHUP`/`exit`) so we kill it on any pi
     exit path.
   - Poll `/health` until 200 (up to 120 s).
4. **Discover the catalog** — `GET /v1/models` → array of `{ id, … }`. Each
   becomes a `ProviderModelConfig`.
5. **Register the provider**:

   ```ts
   pi.registerProvider("llama-cpp", {
     baseUrl: `${server.baseUrl}/v1`,
     apiKey: "none",
     api: "openai-completions",
     models: […],
   });
   ```

   This puts the models into pi's `/model` picker and routes any chat request
   for them through `127.0.0.1:<port>`.
6. **Subscribe to shutdown** — on `session_shutdown` we call `server.stop()`
   → SIGTERM, escalating to SIGKILL after 3 s.

### Runtime — what happens when you chat

- `/model` lists the discovered models.
- You pick one. Pi stores it as the active model in its session state.
- Sending a message: pi builds an OpenAI-format request body
  (`{ model: "<id>", messages: […], stream: true, …}`) and POSTs it to
  `http://127.0.0.1:<port>/v1/chat/completions`.
- The router server matches `<id>`, **loads the GGUF on first hit** (because
  `--models-autoload` defaults on), holds up to `--models-max` (default 4) in
  memory, and streams tokens back.
- Switching models is purely an OpenAI-API thing — pi just changes the
  `model` field on subsequent requests. No spawn/stop, no server churn.

### Lifecycle and crash safety (`src/llama.ts`)

Three independent layers, because pi can die in ugly ways:

- **Graceful path** — `session_shutdown` → `stop()` → SIGTERM with SIGKILL
  fallback.
- **Signal path** — handlers for `SIGINT/SIGTERM/SIGHUP` SIGTERM every child
  in `liveChildren`. The `"exit"` event runs a synchronous SIGKILL as last
  resort.
- **Cross-process path** — PID file at `~/.cache/pi-llamacpp/server.pid`.
  Next pi launch's `reapOrphanServer` probes with `kill 0`, SIGTERMs if
  alive, escalates to SIGKILL after 2 s. Only touches PIDs *we* wrote —
  other `llama-server` processes on the system are untouched.

Stderr is tailed for the last 5 lines and surfaced if the child exits during
startup, so spawn failures don't disappear into the void.

### File structure

```
src/
  index.ts   # pi entrypoint: spawn router, fetch /v1/models, register provider
  llama.ts   # process lifecycle: spawn, health check, orphan reap, signal handlers
```

No build step — pi loads TypeScript at runtime via jiti.
