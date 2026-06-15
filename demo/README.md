# 🕸️ VeriGraph Demo

An interactive, single-file web UI that visualizes how `VeriGraphAgent` builds an
**executable evidence DAG** in real time: the model streams its reasoning, each
`<code_interpreter>` block runs in a persistent sandbox, and the evidence graph
(raw data → interpreter variables → computed results → natural-language claims)
is rendered live as `bind` / `infer` claims are created.

<div align="center">
  <em>source → artifact → cell/span → claim, rendered as you watch.</em>
</div>

## 🔧 Requirements

The demo reuses the agent core in [`../src`](../src), so run it from inside the
repository. Install the inference dependencies first:

```bash
pip install openai transformers numpy pandas tqdm json-repair
# optional, only for the Web research mode fallback fetchers:
pip install requests beautifulsoup4
```

The backend talks to an **OpenAI Responses API** endpoint with reasoning
enabled, so point it at a Responses-API-compatible reasoning model.

## 🏃 Run

```bash
# from the repository root
export OPENAI_API_KEY=sk-...        # or set VERIGRAPH_API_KEY
python3 demo/server.py
```

Then open <http://127.0.0.1:7867>.

For an offline, no-network walkthrough (canned trace, no model calls):

```bash
VERIGRAPH_MOCK=1 python3 demo/server.py
```

## 🧪 Evidence modes

The UI exposes four modes:

| Mode | Needs a file | External dependencies |
| --- | --- | --- |
| **Local data** | ✅ a CSV/table | None — works out of the box |
| **Hybrid data + web/API** | ✅ | Optional web/finance config (below) |
| **Web research** | ❌ | A search/browse endpoint *or* `SERPER_API_KEY` |
| **Financial API** | ❌ | A FinSight source tree, or simulated data |

**Local data** is fully self-contained and is the recommended starting point —
just upload a CSV (or use the bundled
[`sample_data/`](sample_data/) tables) and ask a question.

The **Web** and **Financial** modes call out to external services and are
strictly opt-in. They are disabled unless you provide your own configuration
(see below); the demo never ships with hardcoded third-party endpoints.

### Web research (bring your own search)

Provide an OpenAI-compatible search/browse HTTP service, or a Serper key:

```bash
export VERIGRAPH_SEARCH_ENDPOINT=https://your-search-host/search
export VERIGRAPH_BROWSE_ENDPOINT=https://your-search-host/browse
# or, to use Serper as a fallback:
export SERPER_API_KEY=...
```

### Financial API

Point the demo at a FinSight-style source tree that exposes
`tools.financial.*` / `tools.macro.*` modules:

```bash
export VERIGRAPH_FINSIGHT_SRC=/path/to/FinSight/src
```

If you don't have one, run the finance mode against deterministic **simulated**
market data instead — useful for offline presentations:

```bash
VERIGRAPH_MOCK_FINANCE=1 python3 demo/server.py
```

External web/API calls are snapshotted under each run workspace as evidence
artifacts; claims grounded with `bind_from(...)` are exported with their source,
timestamp, content hash, and row/span locator.

## 📤 Outputs

Each completed run writes artifacts under `output/<run_id>/`:

- `trace.json` — full streamed trace
- `graph.json` — the evidence DAG
- `report.json` / `report.md` — the cited final report

`output/latest.json` points to the most recent run. Both `runs/` and `output/`
are git-ignored.

## ⚙️ Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `VERIGRAPH_MODEL` | `gpt-5` | Responses-API reasoning model id. |
| `VERIGRAPH_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint. |
| `VERIGRAPH_API_KEY` | `$OPENAI_API_KEY` | API key for the endpoint. |
| `VERIGRAPH_REASONING_EFFORT` | `medium` | `low` / `medium` / `high`. |
| `VERIGRAPH_DEMO_HOST` / `VERIGRAPH_DEMO_PORT` | `127.0.0.1` / `7867` | Bind address. |
| `VERIGRAPH_AGENT_SRC` | `../src` | Location of the agent core. |
| `VERIGRAPH_MOCK` | unset | Offline canned trace, no model calls. |
| `VERIGRAPH_MOCK_FINANCE` | unset | Deterministic simulated finance data. |
| `VERIGRAPH_FINSIGHT_SRC` | unset | FinSight source tree for live finance tools. |
| `VERIGRAPH_SEARCH_ENDPOINT` / `VERIGRAPH_BROWSE_ENDPOINT` | unset | Web search/browse service. |
| `VERIGRAPH_ENV_FILE` | unset | Optional `key=value` file preloaded into the environment. |

See `python3 demo/server.py` startup logs for the resolved endpoint and model.
