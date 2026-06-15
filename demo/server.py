#!/usr/bin/env python3
from __future__ import annotations

import cgi
import copy
import html
import json
import mimetypes
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from openai import BadRequestError, OpenAI

try:
    import httpx
except Exception:  # pragma: no cover - openai usually brings httpx with it.
    httpx = None  # type: ignore[assignment]


ROOT_DIR = Path(__file__).resolve().parents[1]
# VeriGraph agent core lives under <repo>/src.
AGENT_SRC_DIR = Path(os.getenv("VERIGRAPH_AGENT_SRC", str(ROOT_DIR / "src")))
# Optional: FinSight source tree for the Financial API mode. Leave unset to use
# the deterministic simulated finance data (VERIGRAPH_MOCK_FINANCE=1) instead.
_FINSIGHT_SRC_ENV = os.getenv("VERIGRAPH_FINSIGHT_SRC", "").strip()
FINSIGHT_SRC_DIR = Path(_FINSIGHT_SRC_ENV) if _FINSIGHT_SRC_ENV else None
# Optional: a key=value env file with shared credentials/endpoints to preload.
_ENV_FILE = os.getenv("VERIGRAPH_ENV_FILE", "").strip()
ENV_FILE = Path(_ENV_FILE) if _ENV_FILE else None
# Optional: bring-your-own search/browse HTTP endpoints for the Web research mode.
# When unset, the Web research mode falls back to any locally available search
# tools (e.g. Serper via SERPER_API_KEY) and is otherwise disabled.
SEARCH_ENDPOINT = os.getenv("VERIGRAPH_SEARCH_ENDPOINT", "").strip()
BROWSE_ENDPOINT = os.getenv("VERIGRAPH_BROWSE_ENDPOINT", "").strip()
SEARCH_PROVIDER = os.getenv("VERIGRAPH_SEARCH_PROVIDER", "google")
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SAMPLE_DATA = Path(__file__).resolve().parent / "sample_data" / "merchant_fee_transactions.csv"
QUESTION_FILE = Path(__file__).resolve().parent / "sample_data" / "question.txt"
CHINESE_FINANCE_CASE_FILE = Path(__file__).resolve().parent / "sample_data" / "chinese_finance_case.txt"
NVIDIA_OUTLOOK_CASE_FILE = Path(__file__).resolve().parent / "sample_data" / "nvidia_outlook_case.txt"
TENCENT_MOAT_CASE_FILE = Path(__file__).resolve().parent / "sample_data" / "tencent_moat_case.txt"

if str(AGENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC_DIR))

from agents.core.code_executor import CodeExecutor  # noqa: E402
from agents.resources.init_workspaces import VERIGRAPH_INIT_CODE  # noqa: E402
from agents.resources.prompts import VERIGRAPH_PROMPT  # noqa: E402


FALLBACK_QUERY = (
    "Produce a comprehensive data analysis report that identifies optimal fee optimization "
    "strategies for merchants by analyzing payment transaction patterns, fee rule applicability, "
    "and merchant characteristics. Include merchant-specific recommendations with estimated savings."
)


def load_default_query() -> str:
    if NVIDIA_OUTLOOK_CASE_FILE.exists():
        text = NVIDIA_OUTLOOK_CASE_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    if QUESTION_FILE.exists():
        text = QUESTION_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    return FALLBACK_QUERY


DEFAULT_QUERY = load_default_query()


def load_sample_cases() -> List[Dict[str, str]]:
    cases = []
    if TENCENT_MOAT_CASE_FILE.exists():
        query = TENCENT_MOAT_CASE_FILE.read_text(encoding="utf-8").strip()
        if query:
            cases.append({"id": "tencent_moat", "label": "Tencent moat trajectory (HK/US + web)", "mode": "finance", "query": query})
    if NVIDIA_OUTLOOK_CASE_FILE.exists():
        query = NVIDIA_OUTLOOK_CASE_FILE.read_text(encoding="utf-8").strip()
        if query:
            cases.append({"id": "nvda_outlook", "label": "NVIDIA forward outlook (finance + web)", "mode": "finance", "query": query})
    if CHINESE_FINANCE_CASE_FILE.exists():
        query = CHINESE_FINANCE_CASE_FILE.read_text(encoding="utf-8").strip()
        if query:
            cases.append({"id": "zh_finance_aapl_sp500", "label": "中文混合证据样例", "mode": "finance", "query": query})
    return cases

DEMO_VERIGRAPH_PROMPT = VERIGRAPH_PROMPT + """

### Demo Runtime Contract
- This section overrides any earlier generic guidance about having 50 coding rounds or doing exhaustive exploration.
- The Python interpreter is persistent across turns. Continue from variables shown in Current Environment Space; do not restart the analysis, reload/refetch the same data, or redefine large tables unless the prior value is missing or wrong.
- Emit at most one `<code_interpreter>...</code_interpreter>` block per assistant turn.
- Keep visible `<think>` content concise: state the next analytical move, not a full private chain of thought.
- Keep each assistant turn compact: no more than 70 lines of Python, no broad exploratory prose, and print only 3-8 compact lines of stdout.
- Turn 1 should only inspect what is available (schema/row counts for local data, or one initial fetch for finance/web mode) and bind at most 1-2 grounding claims. Do not attempt the full report in turn 1.
- In later turns, create claims in small batches of 3-5. Use vectorized pandas transformations and named intermediate tables rather than one-off scalar calculations.
- The demo should visibly exercise computation, not only bind/infer. Build and reuse named variables (e.g. `price_table`, `returns_summary`, `peer_compare`, `margin_trend`, `macro_snapshot`, `news_summary`, `fee_model`, `merchant_summary`) appropriate to the question.
- Each evidence-grounded analytical step should produce a small named table/record first, then a `bind` from that single data object into one atomic claim.
- Claims should cover comparisons and thresholds, not only raw totals or single quotes. Examples: NVDA vs peer return, gross-margin trend (expanding vs compressing), latest CPI/rates level, hyperscaler capex direction, top vs median merchant, actual vs optimal ACI, domestic vs international.
- Keep atomic claims granular. Do not bundle unrelated facts into one `bind`; create one claim per comparison, one claim per driver, and one claim per recommendation/risk.
- Keep derived `infer` claims similarly focused: one infer claim should support one report sentence, not an entire section.
- After a successful `submit_answer(...)`, do not write additional prose; the demo renderer will compose the cited report from submitted claims.

### Mode-specific budget
- Local-data mode (single CSV): finish in roughly 4-6 assistant turns. Target 10-14 atomic `bind` claims and 4-6 derived `infer` claims.
- Finance / web / hybrid mode: this is a research-style task and SHOULD be long. The user query will usually be one short sentence (e.g. "analyze company X's moat" or "outlook for stock Y") — it is YOUR job to expand it into a full analyst workflow. Aim for 18-26 assistant turns; do NOT call `submit_answer` before turn 15 unless evidence coverage is already complete (see below).
- Across those turns mix tool calls liberally — at minimum 6-10 financial-API fetches and 4-6 web fetches:
  - Price action: fetch the target ticker plus 2-3 relevant peers and the relevant index (e.g. S&P 500 for US, Hang Seng for HK). Compute returns and volatility over a comparable window.
  - Fundamentals: pull the latest income statement, balance sheet, and cash flow statement. Compute margin trend, revenue mix change, FCF, and balance-sheet health.
  - Macro backdrop: fetch 2-3 macro indicators relevant to the geography (e.g. us_rates / us_cpi / us_market_index for US-listed names; broad index for HK/A-share names).
  - News & catalysts: run `web_search` on the question's specific themes (recent earnings/guidance, key product or franchise, regulatory backdrop, competitive moves) and `open_page` on the most useful 4-6 results. Spend a few turns computing derived tables (returns, vol, margin trends, peer ratios) before binding the corresponding atomic claims.
- Target 18-28 atomic `bind` claims (each grounded in retrieved data via `evidence_refs` to the relevant artifact) and 6-10 derived `infer` claims (covering price action, fundamentals, macro backdrop, news catalysts, individual risks, and the net stance / verdict that the user asked about). Do not stop after one or two fetches; do not pile multiple comparisons into a single composite — break them apart.
- For external evidence: every atomic claim derived from a financial API or web page MUST include `evidence_refs` pointing at the corresponding artifact id, with a useful `locator` (e.g. column name, row date, or quoted phrase). Without `evidence_refs` the claim is not verifiable.
- Be selective with web search: only `open_page` results that you actually use to ground a claim. Discovery search snippets that turn out irrelevant should NOT be bound to claims — they will be filtered from the final graph automatically. Aim for high signal: 4-6 truly useful pages, not a dozen marginal ones.
- Coverage check before `submit_answer`: confirm you have (a) ≥1 main-ticker price-action claim AND ≥1 peer-comparison claim, (b) ≥2 fundamental claims (margin / revenue mix / cash), (c) ≥2 macro claims, (d) ≥2 news/catalyst claims, (e) ≥2 explicit risk or counter-evidence claims with citations. Only when all five buckets are filled, synthesize 4-8 composite `infer` claims (one per major thesis / risk) and call `submit_answer([...])` with the strongest finals.
- Language: write claims, reasoning, and the final report in the SAME language as the user query (Chinese question → Chinese claims and Chinese report; English question → English).
"""


EXTERNAL_EVIDENCE_PROMPT = """

### External Evidence Tools for Web/API Modes
- This section overrides the earlier generic network rule only for the registered evidence functions below. Do not use `requests`, `httpx`, `yfinance`, `akshare`, shell commands, or arbitrary network calls directly.
- External data must enter the graph as an `EvidenceArtifact`. Use the registered functions so each source is snapshotted, hashed, timestamped, and exported into the evidence graph.
- Available functions:
  - `web_search(query, top_k=5)`: discover candidate pages through the configured search API. Search-result snippets are discovery evidence; for important factual claims, call `open_page(url, task=...)` first.
  - `open_page(url_or_search_artifact, task="")`: fetch a page/PDF snapshot through the configured browse API as text evidence. Alias: `browse(...)`.
  - `fetch_stock_price(stock_code, market="US", period="1y")`: fetch OHLCV price history.
  - `fetch_stock_profile(stock_code, market="US")`: fetch company profile data.
  - `fetch_stock_metrics(stock_code, market="US")`: fetch valuation/profitability metrics.
  - `fetch_balance_sheet(stock_code, market="US")`, `fetch_income_statement(...)`, `fetch_cash_flow(...)`: fetch financial statements.
  - `fetch_us_market_index(index_symbol="^GSPC", period="1y")`: fetch US index OHLCV data.
  - `fetch_us_macro(indicator, start=None)`: indicator is one of `cpi`, `gdp`, `unemployment`, `rates`.
  - `fetch_finsight(tool_name, **kwargs)`: generic adapter for the FinSight tools.
  - `open_search_result(search_result, task="")`: explicit helper equivalent to `open_page(search_result, task=...)`.
  - `get_evidence_catalog()`: inspect fetched artifacts.
- Each returned artifact has `.data`, `.source`, `.fetched_at`, `.snapshot_path`, `.content_hash`, and `.ref(...)`.
- For search results, you may pass the artifact directly into `open_page`, e.g. `results = web_search("..."); page = open_page(results[0], task="...")`. You do not need to manually copy the URL.
- The Python environment is persistent across turns. Reuse existing variables shown in Current Environment Space; do not reload or refetch the same artifact in later turns unless a previous call failed or the value is missing.
- When a claim uses external evidence, ground it with `bind_from(artifact_or_ref, template, **values)` or call `attach_evidence(claim, artifact_or_ref)` after `bind`. Use `_row`, `_columns`, `_locator`, or `_snippet` keyword arguments in `bind_from` when you can identify the exact table row, field, or text span.
- For live market/API data, include the relevant observation date or fetched snapshot time in bound claims when it affects interpretation.
- Search broadly enough to answer the question, but keep the demo compact: fetch at most 3-5 external artifacts unless the user asks for deeper research.
"""


EXTERNAL_EVIDENCE_INIT_CODE = r'''
import asyncio as _ev_asyncio
import hashlib as _ev_hashlib
import importlib as _ev_importlib
import re as _ev_re
import time as _ev_time
from dataclasses import dataclass as _ev_dataclass, field as _ev_field, asdict as _ev_asdict
from urllib.parse import urlsplit as _ev_urlsplit, urlunsplit as _ev_urlunsplit

_FINSIGHT_SRC = "<<FINSIGHT_SRC_PLACEHOLDER>>"
_BC_SEARCH_ENDPOINT = "<<SEARCH_ENDPOINT_PLACEHOLDER>>"
_BC_BROWSE_ENDPOINT = "<<BROWSE_ENDPOINT_PLACEHOLDER>>"
_BC_SEARCH_PROVIDER = "<<SEARCH_PROVIDER_PLACEHOLDER>>"
if _FINSIGHT_SRC and os.path.exists(_FINSIGHT_SRC) and _FINSIGHT_SRC not in sys.path:
    sys.path.insert(0, _FINSIGHT_SRC)


def _ev_json_safe(value, max_len=500):
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "...[truncated]"
        return value
    try:
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
    except Exception:
        pass
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": list(value.shape),
            "columns": [str(c) for c in list(value.columns)[:24]],
            "head": value.head(3).astype(str).to_dict(orient="records"),
        }
    if isinstance(value, pd.Series):
        return {"type": "Series", "len": int(len(value)), "name": str(value.name), "head": value.head(5).astype(str).tolist()}
    if isinstance(value, dict):
        return {str(k): _ev_json_safe(v, max_len=max_len) for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple, set)):
        return [_ev_json_safe(v, max_len=max_len) for v in list(value)[:40]]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _ev_json_safe(value.to_dict(), max_len=max_len)
        except Exception:
            pass
    text = repr(value)
    return text if len(text) <= max_len else text[:max_len] + "...[truncated]"


def _ev_hash_payload(data):
    try:
        if isinstance(data, pd.DataFrame):
            payload = data.to_json(orient="split", date_format="iso", default_handler=str)
        else:
            payload = json.dumps(_ev_json_safe(data, max_len=2000), sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = repr(data)
    return _ev_hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


@_ev_dataclass
class EvidenceRef:
    artifact_id: str
    locator: str = ""
    row: object = None
    columns: list = _ev_field(default_factory=list)
    snippet: str = ""

    def to_dict(self):
        return {
            "artifact_id": self.artifact_id,
            "locator": self.locator,
            "row": _ev_json_safe(self.row),
            "columns": [str(c) for c in (self.columns or [])],
            "snippet": _truncate_text(self.snippet, 360),
        }


@_ev_dataclass
class EvidenceArtifact:
    evidence_id: str
    tool: str
    name: str
    description: str
    source: str
    params: dict
    fetched_at: str
    content_hash: str
    snapshot_path: str
    evidence_type: str = "external"
    data_preview: object = None
    data: object = None

    def ref(self, locator="", row=None, columns=None, snippet=""):
        return EvidenceRef(
            artifact_id=self.evidence_id,
            locator=str(locator or ""),
            row=row,
            columns=list(columns or []),
            snippet=str(snippet or ""),
        )

    def to_frame(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.copy()
        if isinstance(self.data, list):
            return pd.DataFrame(self.data)
        if isinstance(self.data, dict):
            try:
                return pd.DataFrame(self.data)
            except Exception:
                return pd.DataFrame([self.data])
        return pd.DataFrame([{"value": str(self.data)}])

    def to_dict(self):
        return {
            "id": self.evidence_id,
            "tool": self.tool,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "params": _ev_json_safe(self.params),
            "fetched_at": self.fetched_at,
            "content_hash": self.content_hash,
            "snapshot_path": self.snapshot_path,
            "evidence_type": self.evidence_type,
            "data_preview": self.data_preview,
        }

    @property
    def url(self):
        if isinstance(self.data, dict):
            for key in ("url", "link", "canonical_url", "page_url"):
                if self.data.get(key):
                    return str(self.data.get(key))
        match = _ev_re.search(r"https?://\S+", str(self.source or ""))
        return match.group(0).rstrip(".,);]") if match else ""

    @property
    def text(self):
        return str(self.data if self.data is not None else "")

    def __repr__(self):
        suffix = f", url={self.url}" if self.url else ""
        return f"EvidenceArtifact(id={self.evidence_id}, tool={self.tool}, name={self.name}{suffix})"


class EvidenceArtifactManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.snapshot_dir = os.path.join(root_dir, "_evidence_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.counter = 0
        self.artifacts = {}

    def _snapshot(self, artifact_id, data):
        base = os.path.join(self.snapshot_dir, artifact_id)
        try:
            if isinstance(data, pd.DataFrame):
                path = base + ".csv"
                data.to_csv(path, index=False)
            elif isinstance(data, (dict, list)):
                path = base + ".json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(_ev_json_safe(data, max_len=5000), f, ensure_ascii=False, indent=2, default=str)
            else:
                path = base + ".txt"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(str(data))
        except Exception as exc:
            path = base + ".error.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"Snapshot failed: {exc}\n{repr(data)[:5000]}")
        return path

    def register(self, tool, params, data, source="", name="", description="", evidence_type="external"):
        self.counter += 1
        artifact_id = f"src{self.counter}"
        content_hash = _ev_hash_payload(data)
        snapshot_path = self._snapshot(artifact_id, data)
        artifact = EvidenceArtifact(
            evidence_id=artifact_id,
            tool=str(tool or "external_tool"),
            name=str(name or tool or artifact_id),
            description=str(description or ""),
            source=str(source or ""),
            params=dict(params or {}),
            fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            content_hash=content_hash,
            snapshot_path=snapshot_path,
            evidence_type=str(evidence_type or "external"),
            data_preview=_ev_json_safe(data),
            data=data,
        )
        self.artifacts[artifact_id] = artifact
        print(f"[Evidence][OK] {artifact_id}: {artifact.name} ({artifact.tool})")
        return artifact

    def get(self, artifact_id):
        return self.artifacts.get(str(artifact_id))

    def export(self):
        return [artifact.to_dict() for artifact in self.artifacts.values()]

    def state_summary(self):
        if not self.artifacts:
            return "No external evidence artifacts fetched yet."
        lines = []
        for artifact in list(self.artifacts.values())[-12:]:
            source = artifact.source.replace("\n", " ")[:100]
            lines.append(f"[{artifact.evidence_id}] {artifact.tool}: {artifact.name} | source={source} | hash={artifact.content_hash}")
        return "\n".join(lines)


_evidence_mgr = EvidenceArtifactManager(data_dir)


def _ev_run_async(coro):
    try:
        _ev_asyncio.get_running_loop()
    except RuntimeError:
        return _ev_asyncio.run(coro)
    raise RuntimeError("Evidence tools must be called from synchronous code blocks, not inside async_main().")


def _ev_load_class(module_name, class_name):
    module = _ev_importlib.import_module(module_name)
    return getattr(module, class_name)


def _ev_tool_result_data(result):
    return getattr(result, "data", result)


def _ev_tool_result_source(result):
    source = getattr(result, "source", "") or ""
    link = getattr(result, "link", "") or ""
    if link and link not in source:
        source = (source + "\n" + link).strip()
    return source


def _ev_register_tool_results(tool_name, params, results, evidence_type="api"):
    artifacts = []
    for result in results or []:
        data = _ev_tool_result_data(result)
        name = getattr(result, "name", "") or tool_name
        description = getattr(result, "description", "") or ""
        source = _ev_tool_result_source(result)
        artifacts.append(_evidence_mgr.register(tool_name, params, data, source, name, description, evidence_type))
    return artifacts


def _ev_is_missing_data(value):
    if value is None:
        return True
    if isinstance(value, pd.DataFrame):
        return value.empty
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _ev_is_missing_artifact(artifact):
    if isinstance(artifact, list):
        return not artifact or all(_ev_is_missing_artifact(item) for item in artifact)
    if isinstance(artifact, EvidenceArtifact):
        return _ev_is_missing_data(artifact.data)
    return _ev_is_missing_data(artifact)


def _mock_business_dates(period="1mo"):
    period = str(period or "1mo").lower()
    days = 23
    if period.endswith("mo"):
        try:
            days = max(8, int(period[:-2]) * 23)
        except Exception:
            days = 23
    elif period.endswith("y"):
        try:
            days = max(23, int(period[:-1]) * 252)
        except Exception:
            days = 252
    elif period.endswith("d"):
        try:
            days = max(5, int(period[:-1]))
        except Exception:
            days = 23
    end = pd.Timestamp.today().normalize()
    return pd.bdate_range(end=end, periods=days)


def _mock_ohlcv(symbol, period="1mo", base=100.0, total_return=0.03, volatility=0.012, volume_base=10_000_000):
    dates = _mock_business_dates(period)
    n = len(dates)
    x = np.linspace(0, 1, n)
    wave = np.sin(np.linspace(0, np.pi * 3, n)) * volatility * base
    close = base * (1 + total_return * x) + wave
    open_ = close * (1 + np.cos(np.linspace(0, np.pi * 2, n)) * 0.002)
    high = np.maximum(open_, close) * 1.006
    low = np.minimum(open_, close) * 0.994
    volume = (volume_base * (1 + 0.08 * np.sin(np.linspace(0, np.pi * 4, n)))).astype(int)
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": np.round(open_, 2),
        "High": np.round(high, 2),
        "Low": np.round(low, 2),
        "Close": np.round(close, 2),
        "Volume": volume,
        "Symbol": symbol,
        "Simulated": True,
    })


def _mock_stock_price_artifact(stock_code, market="US", period="1mo", reason="fallback"):
    symbol = str(stock_code or "AAPL").upper()
    if symbol == "AAPL":
        data = _mock_ohlcv(symbol, period=period, base=190.0, total_return=0.062, volatility=0.010, volume_base=56_000_000)
    else:
        data = _mock_ohlcv(symbol, period=period, base=100.0, total_return=0.025, volatility=0.012, volume_base=18_000_000)
    return _evidence_mgr.register(
        "mock_stock_price",
        {"stock_code": stock_code, "market": market, "period": period, "reason": reason},
        data,
        source=f"Simulated financial API fallback for {symbol}; generated locally for VeriGraph demo because live financial data was unavailable.",
        name=f"Mock stock price history ({symbol}, {period})",
        description="Deterministic simulated OHLCV table for demo verification; not real market data.",
        evidence_type="simulated_api",
    )


def _mock_market_index_artifact(index_symbol="^GSPC", period="1mo", reason="fallback"):
    symbol = str(index_symbol or "^GSPC").upper()
    data = _mock_ohlcv(symbol, period=period, base=5200.0, total_return=0.021, volatility=0.006, volume_base=3_900_000_000)
    return _evidence_mgr.register(
        "mock_market_index",
        {"index_symbol": index_symbol, "period": period, "reason": reason},
        data,
        source=f"Simulated financial API fallback for {symbol}; generated locally for VeriGraph demo because live index data was unavailable.",
        name=f"Mock market index history ({symbol}, {period})",
        description="Deterministic simulated OHLCV index table for demo verification; not real market data.",
        evidence_type="simulated_api",
    )


def _use_mock_finance():
    return str(os.getenv("VERIGRAPH_MOCK_FINANCE", "")).strip().lower() in {"1", "true", "yes", "on", "force"}


_FINSIGHT_TOOL_REGISTRY = {
    "stock_price": ("tools.financial.stock", "StockPrice"),
    "stock_profile": ("tools.financial.stock", "StockBasicInfo"),
    "stock_metrics": ("tools.financial.stock", "StockBaseInfo"),
    "shareholding": ("tools.financial.stock", "ShareHoldingStructure"),
    "balance_sheet": ("tools.financial.company_statements", "BalanceSheet"),
    "income_statement": ("tools.financial.company_statements", "IncomeStatement"),
    "cash_flow": ("tools.financial.company_statements", "CashFlowStatement"),
    "us_market_index": ("tools.macro.us_macro", "USMarketIndex"),
    "us_cpi": ("tools.macro.us_macro", "USCPI"),
    "us_gdp": ("tools.macro.us_macro", "USGDP"),
    "us_unemployment": ("tools.macro.us_macro", "USUnemployment"),
    "us_rates": ("tools.macro.us_macro", "USInterestRates"),
}


def list_external_tools():
    return sorted(list(_FINSIGHT_TOOL_REGISTRY.keys()) + ["web_search", "open_page"])


_WEB_TOOL_NAMES = {"web_search", "open_page", "browse", "open_search_result"}


def classify_source_type(tool_name: Any) -> str:
    name = str(tool_name or "").strip().lower()
    if not name:
        return "external"
    if name in _WEB_TOOL_NAMES:
        return "web"
    if name in _FINSIGHT_TOOL_REGISTRY:
        return "finance_api"
    if name.startswith("fetch_") or name.startswith("us_") or name.endswith("_statement") or name in {"stock_price", "balance_sheet", "income_statement", "cash_flow"}:
        return "finance_api"
    return "web"


def fetch_finsight(tool_name, **kwargs):
    tool_key = str(tool_name).strip().lower()
    if tool_key not in _FINSIGHT_TOOL_REGISTRY:
        raise ValueError(f"Unknown FinSight tool: {tool_name}. Available: {list_external_tools()}")
    module_name, class_name = _FINSIGHT_TOOL_REGISTRY[tool_key]
    ToolClass = _ev_load_class(module_name, class_name)
    tool = ToolClass()
    results = _ev_run_async(tool.api_function(**kwargs))
    artifacts = _ev_register_tool_results(tool_key, kwargs, results, evidence_type="api")
    return artifacts[0] if len(artifacts) == 1 else artifacts


def _yfinance_price_artifact(stock_code, market="US", period="1y"):
    try:
        import yfinance as _yf
    except Exception:
        return None
    symbol = str(stock_code or "").strip()
    if (market or "").upper() == "HK" and "." not in symbol:
        symbol = f"{int(symbol):04d}.HK" if symbol.isdigit() else f"{symbol}.HK"
    try:
        hist = _yf.Ticker(symbol).history(period=period or "1y")
    except Exception as exc:
        print(f"[Evidence][WARN] yfinance HK fallback for {stock_code}: {exc}")
        return None
    if hist is None or hist.empty:
        return None
    hist = hist.reset_index()
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
    df = hist[keep].copy()
    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str).str.slice(0, 10)
    src = f"Yahoo Finance, {symbol} historical prices. https://finance.yahoo.com/quote/{symbol}/history"
    return _evidence_mgr.register(
        "stock_price",
        {"stock_code": stock_code, "market": market, "period": period, "via": "yfinance_fallback"},
        df,
        source=src,
        name=f"Stock price history ({symbol}, {period})",
        description=f"Yahoo Finance OHLCV history for {symbol}",
        evidence_type="api",
    )


def fetch_stock_price(stock_code, market="US", period="1y"):
    if _use_mock_finance():
        return _mock_stock_price_artifact(stock_code, market=market, period=period, reason="forced_by_VERIGRAPH_MOCK_FINANCE")
    try:
        artifact = fetch_finsight("stock_price", stock_code=stock_code, market=market, period=period)
        if not _ev_is_missing_artifact(artifact):
            return artifact
        print("[Evidence][WARN] live stock_price returned empty data; trying yfinance fallback.")
    except Exception as exc:
        print(f"[Evidence][WARN] live stock_price failed; trying yfinance fallback: {exc}")
    yf_artifact = _yfinance_price_artifact(stock_code, market=market, period=period)
    if yf_artifact is not None:
        return yf_artifact
    print("[Evidence][WARN] yfinance fallback also empty; using mock financial data.")
    return _mock_stock_price_artifact(stock_code, market=market, period=period)


def fetch_stock_profile(stock_code, market="US"):
    return fetch_finsight("stock_profile", stock_code=stock_code, market=market)


def fetch_stock_metrics(stock_code, market="US"):
    return fetch_finsight("stock_metrics", stock_code=stock_code, market=market)


def fetch_balance_sheet(stock_code, market="US", period="annual"):
    return fetch_finsight("balance_sheet", stock_code=stock_code, market=market, period=period)


def fetch_income_statement(stock_code, market="US"):
    return fetch_finsight("income_statement", stock_code=stock_code, market=market)


def fetch_cash_flow(stock_code, market="US"):
    return fetch_finsight("cash_flow", stock_code=stock_code, market=market)


def fetch_us_market_index(index_symbol="^GSPC", period="1y"):
    if _use_mock_finance():
        return _mock_market_index_artifact(index_symbol=index_symbol, period=period, reason="forced_by_VERIGRAPH_MOCK_FINANCE")
    try:
        artifact = fetch_finsight("us_market_index", index_symbol=index_symbol, period=period)
        if not _ev_is_missing_artifact(artifact):
            return artifact
        print("[Evidence][WARN] live us_market_index returned empty data; using mock financial data.")
    except Exception as exc:
        print(f"[Evidence][WARN] live us_market_index failed; using mock financial data: {exc}")
    return _mock_market_index_artifact(index_symbol=index_symbol, period=period)


def fetch_us_macro(indicator, start=None):
    key = str(indicator).strip().lower()
    aliases = {"cpi": "us_cpi", "gdp": "us_gdp", "unemployment": "us_unemployment", "rates": "us_rates", "interest_rates": "us_rates"}
    if key not in aliases:
        raise ValueError("indicator must be one of: cpi, gdp, unemployment, rates")
    kwargs = {}
    if start is not None:
        kwargs["start"] = start
    return fetch_finsight(aliases[key], **kwargs)


def _ev_fallback_open_page(url):
    try:
        import requests as _ev_requests
        from bs4 import BeautifulSoup as _ev_BeautifulSoup
        response = _ev_requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 VeriGraphDemo/0.1"})
        response.raise_for_status()
        soup = _ev_BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "meta", "noscript", "head"]):
            element.extract()
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        return text[:8000]
    except Exception as exc:
        return f"Error fetching page: {exc}"


def _bc_clean_text(text):
    text = str(text or "")
    text = text.replace("Your browser can't play this video.", " ")
    text = _ev_re.sub(r"\s+", " ", text)
    text = _ev_re.sub(r"\s*Show results with:.*$", "", text)
    text = _ev_re.sub(r"\s*Missing:.*$", "", text)
    return text.strip(" -\n\t")


def _bc_canonicalize_url(url):
    raw = str(url or "").strip()
    if not raw:
        return ""
    parts = _ev_urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return raw
    path = parts.path.rstrip("/") or "/"
    return _ev_urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def _bc_domain(url):
    try:
        return _ev_urlsplit(url).netloc.lower()
    except Exception:
        return ""


def _bc_post_json(endpoint, payload, timeout=(8, 45)):
    import requests as _ev_requests
    response = _ev_requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if response.status_code == 422:
        raise ValueError(f"API rejected request: HTTP 422 | {str(payload)[:240]}")
    if response.status_code >= 400:
        raise RuntimeError(f"API request failed: HTTP {response.status_code} | {response.text[:400]}")
    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"API returned invalid JSON: {exc}") from exc
    if not data.get("overall_success", False):
        raise RuntimeError(data.get("error_message") or "API returned overall_success=false")
    return data


def _bc_search_candidates(query, top_k=5):
    queries = list(query) if isinstance(query, (list, tuple)) else [query]
    candidates = []
    seen = set()
    for q_idx, q in enumerate(queries):
        q = str(q or "").strip()
        if not q:
            continue
        payload = {"query": q, "max_num_results": max(1, min(int(top_k or 5), 10)), "provider": _BC_SEARCH_PROVIDER or "google"}
        data = _bc_post_json(_BC_SEARCH_ENDPOINT, payload, timeout=(5, 30))
        for rank, page in enumerate(data.get("items", []) or [], 1):
            url = str(page.get("url") or "").strip()
            canonical = _bc_canonicalize_url(url)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            snippet = _bc_clean_text(page.get("snippets", ""))
            if len(snippet) > 260:
                snippet = snippet[:257].rstrip() + "..."
            candidates.append({
                "query": q,
                "query_index": q_idx,
                "rank": rank,
                "title": _bc_clean_text(page.get("title", "No Title")) or "No Title",
                "url": url,
                "canonical_url": canonical,
                "domain": _bc_domain(url),
                "snippet": snippet,
                "provider": _BC_SEARCH_PROVIDER or "google",
            })
            if len(candidates) >= int(top_k or 5):
                return candidates
    return candidates


def _bc_browse_page(url):
    data = _bc_post_json(_BC_BROWSE_ENDPOINT, {"url": str(url), "max_tokens": 8192}, timeout=(10, 60))
    text = data.get("semanticDocument", "")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Browse API returned empty semanticDocument")
    text = _ev_re.sub(r"\(https?:.*?\)|\[https?:.*?\]", "", text)
    text = text.replace("---", "-").replace("===", "=")
    while "   " in text:
        text = text.replace("   ", " ")
    return text.strip()


def _extract_url_for_open_page(value):
    if isinstance(value, EvidenceArtifact):
        if value.url:
            return value.url
        raise ValueError(f"Evidence artifact {value.evidence_id} does not contain a URL")
    if isinstance(value, dict):
        for key in ("url", "link", "canonical_url", "page_url"):
            if value.get(key):
                return str(value.get(key))
    if isinstance(value, (list, tuple)) and value:
        return _extract_url_for_open_page(value[0])
    text = str(value or "").strip()
    match = _ev_re.search(r"https?://\S+", text)
    return match.group(0).rstrip(".,);]") if match else text


def web_search(query, top_k=5):
    top_k = max(1, min(int(top_k or 5), 10))
    try:
        candidates = _bc_search_candidates(query, top_k=top_k)
        if candidates:
            artifacts = []
            for item in candidates[:top_k]:
                artifacts.append(_evidence_mgr.register(
                    "web_search",
                    {"query": item.get("query"), "provider": item.get("provider"), "endpoint": _BC_SEARCH_ENDPOINT},
                    item,
                    source=f"{item.get('title', '')}\n{item.get('url', '')}",
                    name=item.get("title") or item.get("url") or "Search result",
                    description=item.get("snippet") or "Search API result",
                    evidence_type="search_result",
                ))
            return artifacts
    except Exception as exc:
        print(f"[Evidence][WARN] search API failed, using fallback search tools: {exc}")

    specs = []
    if os.getenv("SERPER_API_KEY"):
        specs.append(("tools.web.search_engine_requests", "SerperSearch"))
    specs.extend([
        ("tools.web.search_engine_requests", "DuckDuckGoSearch"),
        ("tools.web.search_engine_requests", "ExaSearch"),
    ])
    last_error = None
    for module_name, class_name in specs:
        try:
            ToolClass = _ev_load_class(module_name, class_name)
            tool = ToolClass()
            results = _ev_run_async(tool.api_function(query))
            if results:
                artifacts = _ev_register_tool_results("web_search", {"query": query, "backend": class_name}, results[:top_k], evidence_type="search_result")
                return artifacts
        except Exception as exc:
            last_error = exc
            continue
    print(f"[Evidence][WARN] web_search returned no results for {query!r}. Last error: {last_error}")
    return []


def open_page(url, task=""):
    url = _extract_url_for_open_page(url)
    if not url:
        raise ValueError("open_page requires a URL string, search-result dict, or EvidenceArtifact with a URL")
    try:
        text = _bc_browse_page(url)
        return _evidence_mgr.register(
            "open_page",
            {"url": str(url), "task": task or "", "endpoint": _BC_BROWSE_ENDPOINT},
            text[:12000],
            source=f"URL: {url}",
            name=str(url),
            description=f"Browse API page snapshot for task: {task or 'general review'}",
            evidence_type="web_page",
        )
    except Exception as exc:
        print(f"[Evidence][WARN] browse API failed, using fallback page fetchers: {exc}")

    try:
        ToolClass = _ev_load_class("tools.web.web_crawler", "Click")
        tool = ToolClass()
        results = _ev_run_async(tool.api_function([url], task or ""))
        artifacts = _ev_register_tool_results("open_page", {"url": url, "task": task or ""}, results, evidence_type="web_page")
        if artifacts:
            return artifacts[0]
    except Exception as exc:
        print(f"[Evidence][WARN] FinSight open_page failed, using fallback: {exc}")
    text = _ev_fallback_open_page(url)
    return _evidence_mgr.register("open_page", {"url": url, "task": task or "", "backend": "fallback"}, text, source=f"URL: {url}", name=str(url), description="Fetched web page text", evidence_type="web_page")


def open_search_result(search_result, task=""):
    return open_page(search_result, task=task)


browse = open_page


def get_evidence_catalog():
    return _evidence_mgr.export()


def _ev_normalize_refs(evidence, locator="", row=None, columns=None, snippet=""):
    if evidence is None:
        return []
    if isinstance(evidence, (list, tuple, set)):
        refs = []
        for item in evidence:
            refs.extend(_ev_normalize_refs(item, locator=locator, row=row, columns=columns, snippet=snippet))
        return refs
    if isinstance(evidence, EvidenceRef):
        return [evidence.to_dict()]
    if isinstance(evidence, EvidenceArtifact):
        return [evidence.ref(locator=locator, row=row, columns=columns, snippet=snippet).to_dict()]
    artifact = _evidence_mgr.get(str(evidence))
    if artifact is not None:
        return [artifact.ref(locator=locator, row=row, columns=columns, snippet=snippet).to_dict()]
    raise ValueError(f"Unsupported evidence reference: {evidence!r}")


def attach_evidence(claim, evidence, _locator="", _row=None, _columns=None, _snippet=""):
    if not isinstance(claim, Claim):
        raise TypeError("attach_evidence expects a Claim object")
    refs = _ev_normalize_refs(evidence, locator=_locator, row=_row, columns=_columns, snippet=_snippet)
    existing = list(getattr(claim, "evidence_refs", []) or [])
    claim.evidence_refs = existing + refs
    print(f"[Evidence][OK] attached {len(refs)} external reference(s) to {claim.id}")
    return claim


def bind_from(evidence, template_str, **kwargs):
    locator = kwargs.pop("_locator", "")
    row = kwargs.pop("_row", None)
    columns = kwargs.pop("_columns", None)
    snippet = kwargs.pop("_snippet", "")
    claim = bind(template_str, **kwargs)
    return attach_evidence(claim, evidence, _locator=locator, _row=row, _columns=columns, _snippet=snippet)
'''


_WEB_TOOL_NAMES_SERVER = {"web_search", "open_page", "browse", "open_search_result"}
_FINANCE_TOOL_NAMES_SERVER = {
    "stock_price", "stock_profile", "stock_metrics", "shareholding",
    "balance_sheet", "income_statement", "cash_flow",
    "us_market_index", "us_cpi", "us_gdp", "us_unemployment", "us_rates",
    "mock_stock_price", "mock_market_index",
}


def classify_source_type(tool_name: Any) -> str:
    name = str(tool_name or "").strip().lower()
    if not name:
        return "external"
    if name in _WEB_TOOL_NAMES_SERVER:
        return "web"
    if name in _FINANCE_TOOL_NAMES_SERVER:
        return "finance_api"
    if name.startswith("fetch_") or name.startswith("us_") or name.endswith("_statement"):
        return "finance_api"
    return "web"


def build_demo_init_code() -> str:
    finsight_src = str(FINSIGHT_SRC_DIR.resolve()).replace("\\", "/") if FINSIGHT_SRC_DIR else ""
    search_endpoint = SEARCH_ENDPOINT
    browse_endpoint = BROWSE_ENDPOINT
    search_provider = SEARCH_PROVIDER
    external_code = (
        EXTERNAL_EVIDENCE_INIT_CODE
        .replace("<<FINSIGHT_SRC_PLACEHOLDER>>", finsight_src)
        .replace("<<SEARCH_ENDPOINT_PLACEHOLDER>>", search_endpoint)
        .replace("<<BROWSE_ENDPOINT_PLACEHOLDER>>", browse_endpoint)
        .replace("<<SEARCH_PROVIDER_PLACEHOLDER>>", search_provider)
    )
    return VERIGRAPH_INIT_CODE + "\n\n" + external_code


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


if ENV_FILE is not None:
    load_env_file(ENV_FILE)


MODEL_NAME = os.getenv("VERIGRAPH_MODEL", "gpt-5")
BASE_URL = os.getenv("VERIGRAPH_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("VERIGRAPH_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY")
REASONING_EFFORT = os.getenv("VERIGRAPH_REASONING_EFFORT", "medium")
REASONING_SUMMARY = os.getenv("VERIGRAPH_REASONING_SUMMARY", "auto")
TEXT_VERBOSITY = os.getenv("VERIGRAPH_TEXT_VERBOSITY", "low")
MAX_OUTPUT_TOKENS = int(os.getenv("VERIGRAPH_MAX_OUTPUT_TOKENS", "8192"))
MAX_TURNS = int(os.getenv("VERIGRAPH_MAX_TURNS", "30"))
EXECUTION_TIMEOUT_SECONDS = float(os.getenv("VERIGRAPH_EXEC_TIMEOUT", "120"))
MAX_GRAPH_VARIABLES = int(os.getenv("VERIGRAPH_MAX_GRAPH_VARIABLES", "18"))
MAX_ASSISTANT_STREAM_CHARS = int(os.getenv("VERIGRAPH_MAX_ASSISTANT_STREAM_CHARS", "14000"))
MAX_CODE_LINES = int(os.getenv("VERIGRAPH_MAX_CODE_LINES", "90"))
MODEL_TOTAL_TIMEOUT_SECONDS = float(os.getenv("VERIGRAPH_MODEL_TIMEOUT", "240"))
MODEL_CONNECT_TIMEOUT_SECONDS = float(os.getenv("VERIGRAPH_MODEL_CONNECT_TIMEOUT", "20"))
MODEL_STREAM_IDLE_TIMEOUT_SECONDS = float(os.getenv("VERIGRAPH_MODEL_STREAM_IDLE_TIMEOUT", "90"))
MODEL_PROGRESS_INTERVAL_SECONDS = float(os.getenv("VERIGRAPH_MODEL_PROGRESS_INTERVAL", "8"))
MODEL_STREAM_RETRIES = int(os.getenv("VERIGRAPH_MODEL_STREAM_RETRIES", "1"))
MOCK_MODE = env_bool("VERIGRAPH_MOCK", False)
MODEL_REPORT_MODE = env_bool("VERIGRAPH_MODEL_REPORT", True)
RESPONSES_STORE_MODE = os.getenv("VERIGRAPH_RESPONSES_STORE", "stateless").strip().lower()
STORE_SUPPORTED_CACHE: Optional[bool] = None


@dataclass
class RunSession:
    run_id: str
    query: str
    workspace: Path
    files: List[str]
    data_mode: str = "local"
    events: "queue.Queue[Tuple[str, Dict[str, Any]]]" = field(default_factory=queue.Queue)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    done: bool = False
    created_at: float = field(default_factory=time.time)


RUNS: Dict[str, RunSession] = {}
RUNS_LOCK = threading.Lock()


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [json_safe(v) for v in value]
        return str(value)


def emit(session: RunSession, event: str, data: Dict[str, Any]) -> None:
    session.events.put((event, json_safe(data)))


def finish(session: RunSession) -> None:
    session.done = True
    session.events.put(("done", {"run_id": session.run_id}))


def sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "data.csv").strip()
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base)
    base = base.strip(" .") or "data.csv"
    return base[:120]


def truncate_text(text: Any, limit: int = 5000) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def to_python(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [to_python(item) for item in value]
    if isinstance(value, tuple):
        return [to_python(item) for item in value]
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method(exclude_none=True)
            except TypeError:
                dumped = method()
            return to_python(dumped)
    if hasattr(value, "__dict__"):
        return {k: to_python(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def content_part_text(part: Any) -> str:
    if part is None:
        return ""
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return str(part)

    part_type = str(part.get("type") or "")
    if part_type in {"input_text", "output_text", "text", "summary_text"}:
        return content_part_text(part.get("text"))
    if part_type == "refusal":
        return content_part_text(part.get("refusal") or part.get("text"))
    if "text" in part:
        return content_part_text(part.get("text"))
    return ""


def normalize_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()
    return "".join(content_part_text(item).strip() for item in content).strip()


def parse_response(response: Any) -> Dict[str, Any]:
    raw = to_python(response)
    output_items = [item for item in raw.get("output", []) or [] if isinstance(item, dict)]
    reasoning_blocks: List[str] = []
    message_blocks: List[str] = []

    for item in output_items:
        item_type = item.get("type")
        if item_type == "reasoning":
            text = normalize_message_text(item.get("summary"))
            if not text:
                text = normalize_message_text(item.get("content"))
            if not text:
                text = normalize_message_text(item.get("text"))
            if text:
                reasoning_blocks.append(text)
        elif item_type == "message" and item.get("role") == "assistant":
            text = normalize_message_text(item.get("content"))
            if text:
                message_blocks.append(text)

    output_text = normalize_message_text(raw.get("output_text"))
    content = "\n\n".join(message_blocks).strip() or output_text
    return {
        "content": content.strip(),
        "reasoning": "\n\n".join(reasoning_blocks).strip(),
        "response_id": raw.get("id"),
        "status": raw.get("status"),
        "output_items": copy.deepcopy(output_items),
        "raw": raw,
    }


def extract_between(text: str, start: str, end: str) -> Optional[str]:
    matches = re.findall(re.escape(start) + r"(.*?)" + re.escape(end), text or "", flags=re.DOTALL)
    if matches:
        content = matches[-1].strip()
    else:
        idx = (text or "").find(start)
        if idx == -1:
            return None
        content = (text or "")[idx + len(start):].strip()
    fenced = re.fullmatch(r"```(?:python)?\s*(.*?)```", content, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else content


def extract_all_between(text: str, start: str, end: str) -> List[str]:
    return [item.strip() for item in re.findall(re.escape(start) + r"(.*?)" + re.escape(end), text or "", flags=re.DOTALL)]


def build_reasoning_config() -> Optional[Dict[str, Any]]:
    reasoning: Dict[str, Any] = {}
    if REASONING_EFFORT:
        reasoning["effort"] = REASONING_EFFORT
    if REASONING_SUMMARY and REASONING_SUMMARY.lower() not in {"none", "off", "false"}:
        reasoning["summary"] = REASONING_SUMMARY
    return reasoning or None


def build_text_config() -> Optional[Dict[str, Any]]:
    if not TEXT_VERBOSITY:
        return None
    return {"verbosity": TEXT_VERBOSITY}


def build_openai_timeout() -> Any:
    if httpx is None:
        return MODEL_TOTAL_TIMEOUT_SECONDS
    return httpx.Timeout(
        timeout=MODEL_TOTAL_TIMEOUT_SECONDS,
        connect=MODEL_CONNECT_TIMEOUT_SECONDS,
        read=MODEL_STREAM_IDLE_TIMEOUT_SECONDS,
        write=min(60.0, MODEL_TOTAL_TIMEOUT_SECONDS),
        pool=MODEL_CONNECT_TIMEOUT_SECONDS,
    )


def build_response_kwargs(
    *,
    input_items: List[Dict[str, Any]],
    instructions: Optional[str] = None,
    previous_response_id: Optional[str] = None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    store: bool = True,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": MODEL_NAME,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
    }
    if store:
        kwargs["store"] = True
    if instructions is not None:
        kwargs["instructions"] = instructions
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
    reasoning = build_reasoning_config()
    if reasoning:
        kwargs["reasoning"] = reasoning
    text = build_text_config()
    if text:
        kwargs["text"] = text
    return kwargs


def responses_create(
    client: OpenAI,
    *,
    input_items: List[Dict[str, Any]],
    instructions: Optional[str] = None,
    previous_response_id: Optional[str] = None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    store: bool = True,
) -> Any:
    return client.responses.create(
        **build_response_kwargs(
            input_items=input_items,
            instructions=instructions,
            previous_response_id=previous_response_id,
            max_output_tokens=max_output_tokens,
            store=store,
        )
    )


def get_event_field(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def is_transient_model_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(token in text for token in ("fetch failed", "internalservererror", "500", "timeout", "connection", "temporar"))


def format_model_stream_error(exc: BaseException, phase: str, turn: int, stream_state: Dict[str, Any]) -> str:
    if stream_state.get("last_delta_at"):
        last_seen = f"Last token arrived {int(time.time() - stream_state['last_delta_at'])} seconds ago"
    else:
        last_seen = "The upstream endpoint failed before any token was received"
    upstream = truncate_text(f"{type(exc).__name__}: {exc}", 700)
    return f"Model stream failed during {phase} turn {turn}. {last_seen}. Upstream error: {upstream}"


def responses_stream(
    client: OpenAI,
    *,
    session: RunSession,
    phase: str,
    turn: int,
    input_items: List[Dict[str, Any]],
    instructions: Optional[str] = None,
    previous_response_id: Optional[str] = None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    store: bool = True,
) -> Dict[str, Any]:
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    stream_state = {"chars": 0, "reasoning_chars": 0, "last_delta_at": 0.0}
    kwargs = build_response_kwargs(
        input_items=input_items,
        instructions=instructions,
        previous_response_id=previous_response_id,
        max_output_tokens=max_output_tokens,
        store=store,
    )

    final_response = None
    stream_truncated = False
    stop_progress = threading.Event()

    def progress_loop() -> None:
        started = time.time()
        while not stop_progress.wait(MODEL_PROGRESS_INTERVAL_SECONDS):
            elapsed = int(time.time() - started)
            last_delta_at = stream_state.get("last_delta_at") or 0.0
            idle = int(time.time() - last_delta_at) if last_delta_at else elapsed
            emit(
                session,
                "status",
                {
                    "message": "waiting for model",
                    "phase": phase,
                    "turn": turn,
                    "elapsed_seconds": elapsed,
                    "idle_seconds": idle,
                    "text_chars": stream_state["chars"],
                    "reasoning_chars": stream_state["reasoning_chars"],
                    "detail": "The model stream is still open; the demo will continue as soon as new tokens arrive.",
                },
            )

    threading.Thread(target=progress_loop, daemon=True, name=f"stream-watch-{session.run_id}-{phase}-{turn}").start()
    stream_started_at = time.time()
    stream_state["last_delta_at"] = stream_started_at
    try:
        last_error: Optional[BaseException] = None
        for attempt in range(MODEL_STREAM_RETRIES + 1):
            try:
                with client.responses.stream(**kwargs) as stream:
                    for event in stream:
                        now = time.time()
                        if now - stream_started_at > MODEL_TOTAL_TIMEOUT_SECONDS:
                            emit(session, "status", {
                                "message": "model stream wall-clock timeout",
                                "phase": phase, "turn": turn,
                                "elapsed_seconds": int(now - stream_started_at),
                                "detail": f"No response in {int(MODEL_TOTAL_TIMEOUT_SECONDS)}s; cutting the stream and continuing with whatever was produced.",
                            })
                            stream_truncated = True
                            break
                        last_delta_at = stream_state.get("last_delta_at") or stream_started_at
                        if now - last_delta_at > MODEL_STREAM_IDLE_TIMEOUT_SECONDS:
                            emit(session, "status", {
                                "message": "model stream idle timeout",
                                "phase": phase, "turn": turn,
                                "idle_seconds": int(now - last_delta_at),
                                "detail": f"No tokens for {int(MODEL_STREAM_IDLE_TIMEOUT_SECONDS)}s; cutting the stream.",
                            })
                            stream_truncated = True
                            break
                        event_type = str(get_event_field(event, "type", ""))
                        if event_type == "response.output_text.delta":
                            delta = str(get_event_field(event, "delta", "") or "")
                            if delta:
                                text_parts.append(delta)
                                stream_state["chars"] += len(delta)
                                stream_state["last_delta_at"] = time.time()
                                emit(session, f"{phase}_delta", {"turn": turn, "delta": delta})
                                if phase == "assistant" and stream_state["chars"] >= MAX_ASSISTANT_STREAM_CHARS:
                                    stream_truncated = True
                                    emit(
                                        session,
                                        "status",
                                        {
                                            "message": "output budget reached",
                                            "turn": turn,
                                            "text_chars": stream_state["chars"],
                                            "detail": "The assistant action was too long for an interactive demo; asking the model to retry with a compact code block.",
                                        },
                                    )
                                    break
                        elif event_type in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
                            delta = str(get_event_field(event, "delta", "") or "")
                            if delta:
                                reasoning_parts.append(delta)
                                stream_state["reasoning_chars"] += len(delta)
                                stream_state["last_delta_at"] = time.time()
                                emit(session, f"{phase}_reasoning_delta", {"turn": turn, "delta": delta})
                        elif event_type == "response.completed":
                            final_response = get_event_field(event, "response")

                    if final_response is None and not stream_truncated:
                        try:
                            final_response = stream.get_final_response()
                        except RuntimeError:
                            final_response = None
                last_error = None
                break
            except BadRequestError:
                raise
            except Exception as exc:
                last_error = exc
                can_retry = not text_parts and not reasoning_parts and attempt < MODEL_STREAM_RETRIES and is_transient_model_error(exc)
                if not can_retry:
                    break
                emit(
                    session,
                    "status",
                    {
                        "message": "retrying model stream",
                        "phase": phase,
                        "turn": turn,
                        "attempt": attempt + 2,
                        "detail": truncate_text(f"{type(exc).__name__}: {exc}", 360),
                    },
                )
                time.sleep(min(2 + attempt, 5))
        if last_error is not None:
            raise RuntimeError(format_model_stream_error(last_error, phase, turn, stream_state)) from last_error
    finally:
        stop_progress.set()

    content = "".join(text_parts).strip()
    reasoning = "".join(reasoning_parts).strip()
    if stream_truncated:
        return {
            "content": content,
            "reasoning": reasoning,
            "response_id": None,
            "status": "stream_truncated",
            "output_items": [],
            "raw": {},
        }
    if final_response is None:
        if not content:
            raise RuntimeError("Streaming response ended without response.completed and without text deltas.")
        return {
            "content": content,
            "reasoning": reasoning,
            "response_id": None,
            "status": "stream_incomplete",
            "output_items": [{"type": "message", "role": "assistant", "content": content}],
            "raw": {},
        }

    parsed = parse_response(final_response)
    if not parsed.get("content") and content:
        parsed["content"] = content
    if not parsed.get("reasoning") and reasoning:
        parsed["reasoning"] = reasoning
    return parsed


def wants_stateful_responses() -> bool:
    if RESPONSES_STORE_MODE == "auto" and STORE_SUPPORTED_CACHE is False:
        return False
    if RESPONSES_STORE_MODE in {"0", "false", "off", "no", "stateless"}:
        return False
    return True


def mark_store_unsupported() -> None:
    global STORE_SUPPORTED_CACHE
    if RESPONSES_STORE_MODE == "auto":
        STORE_SUPPORTED_CACHE = False


def is_store_unsupported(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "store is not supported" in text or "unsupported_value" in text and "store" in text


def format_code_result(executor: CodeExecutor, call_result: Dict[str, Any]) -> str:
    stdout = str(call_result.get("stdout") or "").strip() or "No output"
    stderr = str(call_result.get("stderr") or "").strip()
    environment = executor.get_variable_info()[:4000]
    claims = executor.get_graph_context_with_vars()[:5000]
    lines = [
        "Execution Result:",
        f'Execution Status: {"Failed" if call_result.get("error") else "Success"}',
        f"Output:\n```{truncate_text(stdout, 4000)}```",
    ]
    if call_result.get("error"):
        lines.append(f"Traceback/Error:\n```{truncate_text(stderr, 4000)}```")
    lines.extend(
        [
            "",
            "Current Environment Space:",
            environment or "No variables defined yet.",
            "",
            "Current claims:",
            claims or "No claims established yet.",
        ]
    )
    return "\n".join(lines)


def template_fields(template: str) -> List[str]:
    fields: List[str] = []
    for _, field_name, _, _ in __import__("string").Formatter().parse(template or ""):
        if not field_name:
            continue
        root = field_name.split(".")[0].split("[")[0]
        if root and root not in fields:
            fields.append(root)
    return fields


def parse_variable_info(variable_info: str) -> List[Dict[str, str]]:
    variables: List[Dict[str, str]] = []
    for line in (variable_info or "").splitlines():
        match = re.match(r"^-\s+([A-Za-z_]\w*):\s*(.*)$", line.strip())
        if not match:
            continue
        name, detail = match.groups()
        if re.fullmatch(r"c\d+", name):
            continue
        variables.append({"id": f"data:{name}", "name": name, "label": name, "detail": detail})
    return variables


def select_data_file(files: List[str]) -> Optional[str]:
    data_exts = (".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".db", ".sqlite")
    for filename in files:
        if filename == SAMPLE_DATA.name:
            return filename
    for filename in files:
        if str(filename).lower().endswith(data_exts):
            return filename
    return files[0] if files else None


def summarize_fields(fields: List[str], limit: int = 10) -> str:
    if not fields:
        return "computed values"
    head = fields[:limit]
    suffix = "" if len(fields) <= limit else f", +{len(fields) - limit} more"
    return ", ".join(head) + suffix


def shortTextForGraph(value: Any, limit: int = 72) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "..."


OPERATION_LABELS = {
    "model": "fee model",
    "merchant": "merchant ranking",
    "segment": "segment comparison",
    "aci": "ACI optimization",
    "risk": "risk/security check",
    "volume": "volume sensitivity",
    "recommendation": "recommendation synthesis",
    "quality": "data validation",
}


def classify_operation(text: str, fields: List[str]) -> str:
    haystack = " ".join([text or "", " ".join(fields or [])]).lower()
    if any(token in haystack for token in ("recommend", "priority", "action", "strategy")):
        return "recommendation"
    if any(token in haystack for token in ("aci", "authorization", "optimal_aci")):
        return "aci"
    if any(token in haystack for token in ("fraud", "security", "approval", "risk")):
        return "risk"
    if any(token in haystack for token in ("merchant", "rank", "top", "median")):
        return "merchant"
    if any(token in haystack for token in ("domestic", "international", "credit", "debit", "scheme", "segment")):
        return "segment"
    if any(token in haystack for token in ("volume", "tier", "sensitivity")):
        return "volume"
    if any(token in haystack for token in ("row", "column", "missing", "schema", "dataset")):
        return "quality"
    return "model"


def select_visible_variables(variables: List[Dict[str, str]], limit: int = MAX_GRAPH_VARIABLES) -> List[Dict[str, str]]:
    if limit <= 0:
        return []

    def score(var: Dict[str, str]) -> tuple:
        name = var.get("name", "")
        detail = var.get("detail", "")
        priority = 4
        if "DataFrame" in detail:
            priority = 0
        elif any(token in name.lower() for token in ("summary", "by_", "total", "saving", "fee", "fraud", "merchant")):
            priority = 1
        elif any(token in detail for token in ("dict", "list", "float", "int")):
            priority = 2
        return (priority, len(name), name)

    selected = sorted(variables, key=score)[:limit]
    return selected


def enrich_graph(exported_graph: Dict[str, Any], variable_info: str, files: List[str]) -> Dict[str, Any]:
    graph = copy.deepcopy(exported_graph or {"graph": {"claims": [], "edges": []}, "final_claim_ids": []})
    claims = graph.get("graph", {}).get("claims", []) or []
    claim_edges = graph.get("graph", {}).get("edges", []) or []
    external_artifacts = graph.get("external_artifacts") or graph.get("graph", {}).get("artifacts", []) or []
    cited_artifact_ids = set()
    for claim in claims:
        for ref in claim.get("evidence_refs") or []:
            if isinstance(ref, dict):
                aid = str(ref.get("artifact_id") or "")
                if aid:
                    cited_artifact_ids.add(aid)
    if cited_artifact_ids:
        external_artifacts = [a for a in external_artifacts if isinstance(a, dict) and str(a.get("id") or a.get("evidence_id") or "") in cited_artifact_ids]
    final_ids = set(str(item) for item in graph.get("final_claim_ids", []) or [])

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}
    data_file = select_data_file(files)
    data_file_id = f"file:{data_file}" if data_file else None

    def add_node(node_id: str, **attrs: Any) -> None:
        existing = nodes.get(node_id, {})
        existing.update({"id": node_id, **attrs})
        nodes[node_id] = existing

    def add_edge(source: str, target: str, edge_type: str, **attrs: Any) -> None:
        edge_id = f"{source}->{target}:{edge_type}"
        edges[edge_id] = {"id": edge_id, "source": source, "target": target, "type": edge_type, **attrs}

    def add_operation(op_key: str) -> str:
        label = OPERATION_LABELS.get(op_key, op_key.replace("_", " "))
        op_id = f"op:{op_key}"
        add_node(op_id, label=label, kind="operation", content=f"Computed stage: {label}")
        if data_file_id:
            add_edge(data_file_id, op_id, "compute", label=f"load -> {label}")
        return op_id

    for filename in files:
        add_node(
            f"file:{filename}",
            label=filename,
            kind="data",
            data_role="raw",
            source_type="local_file",
            content=f"Raw data file: {filename}",
        )

    artifact_by_id: Dict[str, Dict[str, Any]] = {}
    for artifact in external_artifacts:
        if not isinstance(artifact, dict):
            continue
        aid = str(artifact.get("id") or artifact.get("evidence_id") or "")
        if not aid:
            continue
        artifact_by_id[aid] = artifact
        source_text = str(artifact.get("source") or "").strip()
        source_url = ""
        for token in re.split(r"\s+", source_text):
            if token.startswith("http://") or token.startswith("https://"):
                source_url = token.strip()
                break
        source_label = urlparse(source_url).netloc if source_url else ""
        tool_name = str(artifact.get("tool") or "external")
        source_type = classify_source_type(tool_name)
        artifact_id = f"artifact:{aid}"
        display_label = str(artifact.get("name") or aid)[:80]
        if source_label and source_label not in display_label:
            display_label = f"{display_label} · {source_label}"
        content_lines = [
            f"Tool: {tool_name}",
            f"Fetched at: {artifact.get('fetched_at') or 'unknown'}",
        ]
        if source_url:
            content_lines.append(f"URL: {source_url}")
        elif source_text:
            content_lines.append(f"Source: {source_text}")
        if artifact.get("content_hash"):
            content_lines.append(f"Hash: {artifact.get('content_hash')}")
        if artifact.get("snapshot_path"):
            content_lines.append(f"Snapshot: {artifact.get('snapshot_path')}")
        if artifact.get("description"):
            content_lines.append(str(artifact.get("description")))
        add_node(
            artifact_id,
            label=display_label,
            kind="data",
            data_role="external",
            source_type=source_type,
            content="\n".join(content_lines),
            evidence_type=artifact.get("evidence_type", "external"),
            fetched_at=artifact.get("fetched_at"),
            source=source_text,
            source_url=source_url,
            source_label=source_label,
            tool=tool_name,
            snapshot_path=artifact.get("snapshot_path"),
            content_hash=artifact.get("content_hash"),
        )

    all_variables = parse_variable_info(variable_info)
    for var in select_visible_variables(all_variables):
        op_key = classify_operation(f"{var.get('name', '')} {var.get('detail', '')}", [var.get("name", "")])
        op_label = OPERATION_LABELS.get(op_key, op_key.replace("_", " "))
        add_node(
            var["id"],
            label=var["label"],
            kind="data",
            data_role="computed",
            source_type="computed",
            operation=op_label,
            content=var.get("detail", ""),
        )
        if data_file_id:
            add_edge(data_file_id, var["id"], "compute", label=op_label)

    for claim in claims:
        cid = str(claim.get("id") or "")
        if not cid:
            continue
        add_node(
            cid,
            label=cid,
            kind="claim",
            claim_type=claim.get("type", "atomic"),
            content=claim.get("content", ""),
            reasoning=claim.get("reasoning", ""),
            final=bool(claim.get("final_node") or cid in final_ids),
            template=claim.get("template", ""),
            premise_ids=claim.get("premise_ids", []),
        )
        if claim.get("type") == "atomic":
            fields = template_fields(str(claim.get("template") or ""))
            op_key = classify_operation(f"{claim.get('content', '')} {claim.get('template', '')}", fields)
            op_label = OPERATION_LABELS.get(op_key, op_key.replace("_", " "))
            source_id = f"data:bound:{cid}"
            add_node(
                source_id,
                label=summarize_fields(fields, 3) or f"data for {cid}",
                kind="data",
                data_role="bound",
                source_type="computed",
                operation=op_label,
                content=f"Bind interpretation: {summarize_fields(fields)}",
                fields=fields,
            )
            if data_file_id:
                add_edge(data_file_id, source_id, "compute", label=op_label)
            add_edge(source_id, cid, "bind", label=f"bind {summarize_fields(fields, 3)}")
        for idx, ref in enumerate(claim.get("evidence_refs") or []):
            if not isinstance(ref, dict):
                continue
            aid = str(ref.get("artifact_id") or "")
            if not aid:
                continue
            artifact_id = f"artifact:{aid}"
            if artifact_id not in nodes:
                artifact = artifact_by_id.get(aid, {})
                tool_name = str(artifact.get("tool") or "external")
                add_node(
                    artifact_id,
                    label=str(artifact.get("name") or aid),
                    kind="data",
                    data_role="external",
                    source_type=classify_source_type(tool_name),
                    tool=tool_name,
                    content=str(artifact.get("source") or f"External artifact {aid}"),
                )
            locator_bits = []
            if ref.get("locator"):
                locator_bits.append(str(ref.get("locator")))
            if ref.get("row") is not None:
                locator_bits.append(f"row={ref.get('row')}")
            if ref.get("columns"):
                locator_bits.append("cols=" + ", ".join(str(c) for c in ref.get("columns") or []))
            locator = "; ".join(locator_bits) or "external evidence reference"
            evidence_id = f"evidence:{cid}:{idx + 1}"
            add_node(
                evidence_id,
                label=locator[:80],
                kind="data",
                data_role="evidence",
                source_type="evidence",
                content=str(ref.get("snippet") or locator or f"Reference into {aid}"),
                artifact_id=aid,
                locator=locator,
            )
            add_edge(artifact_id, evidence_id, "cite", label=locator_bits[0] if locator_bits else "span")
            add_edge(evidence_id, cid, "bind", label="bind evidence")

    for edge in claim_edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source and target:
            target_claim = next((claim for claim in claims if str(claim.get("id")) == target), {})
            reasoning = str(target_claim.get("reasoning") or "")
            short_reason = truncate_text(reasoning, 70) if reasoning else "infer"
            add_edge(source, target, "infer", label=short_reason, reasoning=truncate_text(reasoning, 240))

    return {
        "claims": claims,
        "edges": claim_edges,
        "final_claim_ids": list(final_ids),
        "nodes": list(nodes.values()),
        "links": list(edges.values()),
        "stats": {
            "claims": len(claims),
            "atomic": len([c for c in claims if c.get("type") == "atomic"]),
            "composite": len([c for c in claims if c.get("type") == "composite"]),
            "final": len(final_ids),
            "variables": len([n for n in nodes.values() if n.get("kind") == "data" and n.get("data_role") != "raw"]),
            "data_nodes": len([n for n in nodes.values() if n.get("kind") == "data"]),
            "operations": len([n for n in nodes.values() if n.get("kind") == "operation"]),
            "artifacts": len(external_artifacts),
            "hidden_variables": max(0, len(all_variables) - len(select_visible_variables(all_variables))),
        },
    }


def extract_final_claims(exported_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = exported_graph.get("graph", {}).get("claims", []) or []
    final_ids = [str(item) for item in exported_graph.get("final_claim_ids", []) or []]
    if not final_ids:
        final_ids = [str(c.get("id")) for c in claims if c.get("final_node")]
    claim_map = {str(c.get("id")): c for c in claims if c.get("id")}
    return [claim_map[cid] for cid in final_ids if cid in claim_map]


def auto_select_final_claims(exported_graph: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    claims = exported_graph.get("graph", {}).get("claims", []) or []
    composite = [c for c in claims if c.get("type") == "composite" and c.get("id")]
    atomic = [c for c in claims if c.get("type") != "composite" and c.get("id")]
    selected = (composite[-limit:] if composite else atomic[-limit:])[:limit]
    final_ids = [str(c.get("id")) for c in selected]
    exported_graph["final_claim_ids"] = final_ids
    for claim in claims:
        if str(claim.get("id")) in final_ids:
            claim["final_node"] = True
    return selected


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def fallback_report(question: str, final_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    sentences = [
        {"text": str(claim.get("content") or ""), "claim_ids": [str(claim.get("id"))]}
        for claim in final_claims
        if claim.get("content") and claim.get("id")
    ]
    recommendation_tokens = ("recommend", "optimization", "strategy", "savings", "priority", "reconfigure")
    recommendations = [
        sentence for sentence in sentences if any(token in sentence["text"].lower() for token in recommendation_tokens)
    ]
    findings = [sentence for sentence in sentences if sentence not in recommendations]
    sections = []
    if findings:
        sections.append({"heading": "Evidence-grounded findings", "sentences": findings})
    if recommendations:
        sections.append({"heading": "Recommendations", "sentences": recommendations})
    if not sections:
        sections = [{"heading": "Evidence-grounded findings", "sentences": sentences}]
    return {
        "title": "Verifiable Analysis Report",
        "summary": sentences[:3],
        "sections": sections,
        "question": question,
    }


def text_tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", text or "") if len(token) > 2 or token.isdigit()}


def normalize_report(
    data: Optional[Dict[str, Any]],
    question: str,
    final_claims: List[Dict[str, Any]],
    all_claims: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    candidate_claims = [claim for claim in (all_claims or final_claims) if claim.get("id")]
    if not candidate_claims:
        candidate_claims = [claim for claim in final_claims if claim.get("id")]
    valid_ids_ordered = [str(claim.get("id")) for claim in candidate_claims]
    valid_ids = set(valid_ids_ordered)
    claim_by_id = {str(claim.get("id")): claim for claim in candidate_claims}
    if not data:
        return fallback_report(question, final_claims)
    report = {
        "title": str(data.get("title") or "Verifiable Analysis Report"),
        "summary": [],
        "sections": [],
        "question": question,
    }

    def choose_primary_claim(text: str, ids: List[str]) -> Optional[str]:
        pool = [cid for cid in ids if cid in valid_ids] or valid_ids_ordered
        if not pool:
            return None
        sentence_tokens = text_tokens(text)
        sentence_numbers = {token for token in sentence_tokens if token.isdigit()}

        def score(cid: str) -> tuple:
            claim = claim_by_id.get(cid, {})
            claim_text = " ".join(
                str(claim.get(key) or "") for key in ("content", "reasoning", "template")
            )
            claim_tokens = text_tokens(claim_text)
            overlap = len(sentence_tokens & claim_tokens)
            number_overlap = len(sentence_numbers & {token for token in claim_tokens if token.isdigit()})
            exact_bonus = 3 if str(claim.get("content") or "").lower() in text.lower() else 0
            provided_bonus = 2 if cid in ids else 0
            return (overlap * 4 + number_overlap * 3 + exact_bonus + provided_bonus, -len(claim_tokens), cid)

        return max(pool, key=score)

    def clean_sentence(item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, str):
            text = item.strip()
            ids: List[str] = []
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("sentence") or "").strip()
            ids = [str(cid) for cid in item.get("claim_ids", item.get("claims", [])) or []]
        else:
            return None
        valid = [cid for cid in ids if cid in valid_ids]
        # Dedupe while preserving order
        seen = set()
        ordered: List[str] = []
        for cid in valid:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        if not text:
            return None
        if not ordered:
            inferred = choose_primary_claim(text, ids)
            if inferred:
                ordered = [inferred]
            else:
                return None
        return {"text": text, "claim_ids": ordered}

    for item in data.get("summary", []) or []:
        cleaned = clean_sentence(item)
        if cleaned:
            report["summary"].append(cleaned)

    for section in data.get("sections", []) or []:
        if not isinstance(section, dict):
            continue
        sentences = []
        for item in section.get("sentences", []) or []:
            cleaned = clean_sentence(item)
            if cleaned:
                sentences.append(cleaned)
        if sentences:
            report["sections"].append({"heading": str(section.get("heading") or "Findings"), "sentences": sentences})

    if not report["sections"] and not report["summary"]:
        return fallback_report(question, final_claims)
    if not report["sections"]:
        report["sections"].append({"heading": "Findings", "sentences": report["summary"]})
    return report


def generate_cited_report(
    client: OpenAI,
    question: str,
    exported_graph: Dict[str, Any],
    final_claims: List[Dict[str, Any]],
    session: Optional[RunSession] = None,
) -> Dict[str, Any]:
    if not final_claims:
        return {"title": "No submitted claims", "summary": [], "sections": [], "question": question}

    if session is not None:
        emit(session, "report_start", {"turn": 0, "mode": "deterministic" if not MODEL_REPORT_MODE else "model"})

    if not MODEL_REPORT_MODE:
        return fallback_report(question, final_claims)

    # Build a slim claim catalog: final claims + their direct ancestors only,
    # so the report prompt stays compact even on long runs.
    all_graph_claims = exported_graph.get("graph", {}).get("claims", []) or []
    by_id = {str(c.get("id")): c for c in all_graph_claims if c.get("id")}
    relevant: Dict[str, Dict[str, Any]] = {}
    queue = [str(c.get("id")) for c in final_claims if c.get("id")]
    while queue:
        cid = queue.pop()
        if cid in relevant or cid not in by_id:
            continue
        c = by_id[cid]
        relevant[cid] = c
        for pid in c.get("premise_ids") or []:
            queue.append(str(pid))
    if not relevant:
        relevant = {str(c.get("id")): c for c in final_claims if c.get("id")}

    def _slim(c: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": c.get("id"),
            "type": c.get("type"),
            "content": truncate_text(str(c.get("content") or ""), 280),
            "reasoning": truncate_text(str(c.get("reasoning") or ""), 200),
            "premise_ids": c.get("premise_ids") or [],
        }

    slim_final = [_slim(c) for c in final_claims]
    slim_supporting = [_slim(c) for cid, c in relevant.items() if cid not in {str(fc.get("id")) for fc in final_claims}]

    prompt = f"""
You are writing the final analyst-style report for a VeriGraph demo.

User question:
{question}

Final claims (the headline conclusions you must build the report around):
{json.dumps(slim_final, ensure_ascii=False, indent=2)}

Supporting claims (direct evidence for the final claims; cite by id when used):
{json.dumps(slim_supporting, ensure_ascii=False, indent=2)}

Write a polished, readable report — not a list of disconnected facts.
- Match the language of the user question (if Chinese, write Chinese; if English, write English). Do NOT mix languages.
- 3-5 short sections with descriptive headings tailored to the question (e.g. "价格表现 / 基本面 / 宏观背景 / 催化与风险 / 综合判断" or the English equivalents).
- Each section is 1-2 short paragraphs (2-4 sentences each), written in flowing analyst prose with connective phrasing ("which", "however", "as a result", "by contrast" — or the equivalent in the target language). Quantitative facts come from the claims; the connective phrasing comes from you.
- A sentence MAY cite multiple claim ids when it synthesizes several facts. List every claim that supports the sentence in `claim_ids`.
- Do NOT pad with content that isn't backed by a claim. Do NOT invent numbers.
- Open with a short `summary` (2-3 sentences) acting as a TL;DR — this can be the most claim-dense part.

Return JSON only in this exact shape:
{{
  "title": "concise informative title in the target language",
  "summary": [
    {{"text": "one TL;DR sentence", "claim_ids": ["c1", "c2"]}}
  ],
  "sections": [
    {{
      "heading": "section heading in the target language",
      "sentences": [
        {{"text": "sentence text", "claim_ids": ["c3"]}},
        {{"text": "another sentence in the same paragraph", "claim_ids": ["c4", "c5"]}}
      ]
    }}
  ]
}}
""".strip()

    try:
        if session is not None:
            parsed = responses_stream(
                client,
                session=session,
                phase="report",
                turn=0,
                instructions="You produce strict JSON for evidence-cited reports.",
                input_items=[{"type": "message", "role": "user", "content": prompt}],
                max_output_tokens=4096,
                store=False,
            )
        else:
            response = responses_create(
                client,
                instructions="You produce strict JSON for evidence-cited reports.",
                input_items=[{"type": "message", "role": "user", "content": prompt}],
                max_output_tokens=4096,
                store=False if RESPONSES_STORE_MODE == "auto" else wants_stateful_responses(),
            )
            parsed = parse_response(response)
        all_claims = exported_graph.get("graph", {}).get("claims", []) or []
        return normalize_report(extract_json_object(parsed.get("content", "")), question, final_claims, all_claims)
    except Exception:
        return fallback_report(question, final_claims)


def build_demo_prompt() -> str:
    return DEMO_VERIGRAPH_PROMPT + EXTERNAL_EVIDENCE_PROMPT


def make_user_prompt(query: str, files: List[str], data_mode: str = "local") -> str:
    attached = ", ".join(files)
    file_hint = f"\n\n(Attach files: {attached})" if attached else ""
    mode = (data_mode or "local").strip().lower()
    mode_hint = {
        "local": "Use the attached local data files as the primary evidence source.",
        "hybrid": "Use attached local data when relevant and fetch external web/API evidence only when it materially helps answer the query.",
        "web": "No local data file is required. Use web_search/open_page to gather external evidence and ground claims with bind_from.",
        "finance": "No local data file is required. Use FinSight financial/API tools for structured market/company data, and use web_search/open_page when the query asks for public context, news, or other unstructured evidence. Ground external claims with bind_from.",
    }.get(mode, "Use available local and external evidence tools as appropriate.")
    return (query or DEFAULT_QUERY).strip() + file_hint + f"\n\n(Data source mode: {mode}. {mode_hint})"


def report_to_markdown(report: Dict[str, Any]) -> str:
    lines = [f"# {report.get('title') or 'Verifiable Analysis Report'}", ""]

    def sentence_text(sentence: Dict[str, Any]) -> str:
        cites = " ".join(f"[{cid}]" for cid in sentence.get("claim_ids", []) or [])
        text = str(sentence.get("text", "")).strip()
        return f"{text} {cites}".strip()

    if report.get("summary"):
        lines.append(" ".join(sentence_text(sentence) for sentence in report.get("summary", []) or []))
        lines.append("")
    for section in report.get("sections", []) or []:
        lines.extend([f"## {section.get('heading') or 'Findings'}", ""])
        paragraph = " ".join(sentence_text(sentence) for sentence in section.get("sentences", []) or [])
        if paragraph:
            lines.append(paragraph)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_run_outputs(
    session: RunSession,
    *,
    report: Dict[str, Any],
    final_claims: List[Dict[str, Any]],
    raw_graph: Dict[str, Any],
    graph: Dict[str, Any],
) -> Dict[str, str]:
    output_dir = OUTPUT_DIR / session.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "trace": output_dir / "trace.json",
        "graph": output_dir / "graph.json",
        "report_json": output_dir / "report.json",
        "report_md": output_dir / "report.md",
    }
    files["trace"].write_text(json.dumps(session.trace, ensure_ascii=False, indent=2), encoding="utf-8")
    files["graph"].write_text(json.dumps({"raw_graph": raw_graph, "view_graph": graph}, ensure_ascii=False, indent=2), encoding="utf-8")
    files["report_json"].write_text(
        json.dumps({"report": report, "final_claims": final_claims}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    files["report_md"].write_text(report_to_markdown(report), encoding="utf-8")
    latest = OUTPUT_DIR / "latest.json"
    latest.write_text(
        json.dumps({"run_id": session.run_id, "output_dir": str(output_dir), "files": {k: str(v) for k, v in files.items()}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir), **{k: str(v) for k, v in files.items()}}


def run_real_agent(session: RunSession) -> None:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=build_openai_timeout())
    executor = CodeExecutor(
        str(session.workspace),
        build_demo_init_code(),
        execution_timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
    )
    previous_response_id: Optional[str] = None
    stateful_responses = wants_stateful_responses()
    conversation_items: List[Dict[str, Any]] = []
    last_exported = {"graph": {"claims": [], "edges": []}, "final_claim_ids": []}
    last_claim_ids: set[str] = set()

    emit(
        session,
        "status",
        {
            "message": "started",
            "model": MODEL_NAME,
            "base_url": BASE_URL,
            "reasoning": build_reasoning_config(),
            "responses_state": "previous_response_id" if stateful_responses else "stateless_reasoning_items",
            "files": session.files,
            "data_mode": session.data_mode,
        },
    )

    try:
        first_user_item = {"type": "message", "role": "user", "content": make_user_prompt(session.query, session.files, session.data_mode)}
        pending_input = [first_user_item]
        conversation_items = [copy.deepcopy(first_user_item)]
        for turn in range(1, MAX_TURNS + 1):
            input_items = pending_input if stateful_responses else conversation_items
            emit(session, "assistant_start", {"turn": turn})
            emit(
                session,
                "status",
                {
                    "message": "calling model",
                    "turn": turn,
                    "detail": "Waiting for the next ReAct action from the model.",
                },
            )
            try:
                parsed = responses_stream(
                    client,
                    session=session,
                    phase="assistant",
                    turn=turn,
                    instructions=build_demo_prompt() if (previous_response_id is None or not stateful_responses) else None,
                    previous_response_id=previous_response_id if stateful_responses else None,
                    input_items=input_items,
                    store=stateful_responses,
                )
            except BadRequestError as exc:
                if not stateful_responses or not is_store_unsupported(exc):
                    raise
                mark_store_unsupported()
                stateful_responses = False
                previous_response_id = None
                emit(
                    session,
                    "status",
                    {
                        "message": "stateless responses",
                        "responses_state": "stateless_reasoning_items",
                        "detail": "store=true is not supported by this endpoint; replaying reasoning output items in input.",
                    },
                )
                parsed = responses_stream(
                    client,
                    session=session,
                    phase="assistant",
                    turn=turn,
                    instructions=build_demo_prompt(),
                    input_items=conversation_items,
                    store=False,
                )
            assistant_text = parsed.get("content", "")
            thought_blocks = extract_all_between(assistant_text, "<think>", "</think>")
            code = extract_between(assistant_text, "<code_interpreter>", "</code_interpreter>")
            code_lines = len([line for line in (code or "").splitlines() if line.strip()])
            emit(
                session,
                "assistant_done",
                {
                    "turn": turn,
                    "reasoning_summary": parsed.get("reasoning", ""),
                    "thought": "\n\n".join(thought_blocks),
                    "assistant_text": truncate_text(assistant_text, 12000),
                    "code": code or "",
                    "response_status": parsed.get("status"),
                },
            )

            if parsed.get("status") == "stream_truncated" or (code and code_lines > MAX_CODE_LINES):
                code_result = (
                    "[Verifier runtime guard] The previous assistant action was too long for the interactive demo. "
                    f"Limit the next Python block to <= {MAX_CODE_LINES} non-empty lines, create only 3-5 claims, "
                    "print only compact diagnostics, and continue from existing variables instead of producing one huge script."
                )
                exported = executor.export_graph()
                last_exported = exported
                graph = enrich_graph(exported, executor.get_variable_info(), session.files)
                step_payload = {
                    "turn": turn,
                    "thought": "\n\n".join(thought_blocks),
                    "reasoning_summary": parsed.get("reasoning", ""),
                    "code": truncate_text(code or assistant_text, 12000),
                    "result": code_result,
                    "stdout": "",
                    "stderr": code_result,
                    "error": True,
                    "graph": graph,
                    "variables": executor.get_variable_info(),
                    "added_claim_ids": [],
                }
                session.trace.append(copy.deepcopy(step_payload))
                emit(session, "step", step_payload)
                tool_item = {"type": "message", "role": "user", "content": f"<tool_response>\n{code_result}\n</tool_response>"}
                pending_input = [tool_item]
                if not stateful_responses:
                    conversation_items.append(copy.deepcopy(tool_item))
                continue

            if stateful_responses:
                previous_response_id = parsed.get("response_id") or previous_response_id
            else:
                conversation_items.extend(parsed.get("output_items", []))

            if code:
                emit(
                    session,
                    "status",
                    {
                        "message": "executing code",
                        "turn": turn,
                        "detail": "Running the model-generated Python block and updating the evidence graph.",
                    },
                )
                call_result = executor.execute(code.strip())
                code_result = format_code_result(executor, call_result)
                exported = executor.export_graph()
                current_ids = {str(c.get("id")) for c in exported.get("graph", {}).get("claims", []) if c.get("id")}
                added_claim_ids = sorted(current_ids - last_claim_ids, key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
                last_claim_ids = current_ids
                last_exported = exported
                graph = enrich_graph(exported, executor.get_variable_info(), session.files)
                graph["added_claim_ids"] = added_claim_ids

                step_payload = {
                        "turn": turn,
                        "thought": "\n\n".join(thought_blocks),
                        "reasoning_summary": parsed.get("reasoning", ""),
                        "code": code,
                        "result": code_result,
                        "stdout": str(call_result.get("stdout") or ""),
                        "stderr": str(call_result.get("stderr") or ""),
                        "error": bool(call_result.get("error")),
                        "graph": graph,
                        "variables": executor.get_variable_info(),
                        "added_claim_ids": added_claim_ids,
                    }
                session.trace.append(copy.deepcopy(step_payload))
                emit(session, "step", step_payload)

                tool_item = {
                    "type": "message",
                    "role": "user",
                    "content": f"<tool_response>\n{code_result}\n</tool_response>",
                }
                pending_input = [tool_item]
                if not stateful_responses:
                    conversation_items.append(copy.deepcopy(tool_item))
                if executor.is_submission_finished():
                    break
                continue

            break

        final_claims = extract_final_claims(last_exported)
        if not final_claims:
            final_claims = auto_select_final_claims(last_exported)
            if final_claims:
                emit(
                    session,
                    "status",
                    {
                        "message": "auto-selected final claims",
                        "detail": "The model reached the demo turn budget before submit_answer; using the strongest available claims for the cited report.",
                    },
                )
        report = generate_cited_report(client, session.query, last_exported, final_claims, session=session)
        final_graph = enrich_graph(last_exported, executor.get_variable_info(), session.files)
        output_files = write_run_outputs(
            session,
            report=report,
            final_claims=final_claims,
            raw_graph=last_exported,
            graph=final_graph,
        )
        emit(
            session,
            "report",
            {
                "report": report,
                "final_claims": final_claims,
                "graph": final_graph,
                "raw_graph": last_exported,
                "short_answer": "\n".join(str(c.get("content") or "") for c in final_claims),
                "output": output_files,
            },
        )
    finally:
        executor.close()


def run_mock_agent(session: RunSession) -> None:
    emit(session, "status", {"message": "mock_started", "model": "mock", "files": session.files})
    claims: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []

    def snapshot(final_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        exported = {"graph": {"claims": copy.deepcopy(claims), "edges": copy.deepcopy(edges)}, "final_claim_ids": final_ids or []}
        return enrich_graph(exported, "- df: DataFrame (12, 6)\n- q4_revenue: float = 162000\n- growth: float = 0.184", session.files)

    scripted = [
        {
            "thought": "Load the table, inspect monthly revenue and margin, then ground the main computed facts as claims.",
            "code": "df = pd.read_csv('retail_metrics.csv')\nq4_revenue = df[df['quarter'] == 'Q4']['revenue'].sum()\nc1 = bind('Q4 revenue totals {v:.0f}.', v=q4_revenue)",
            "claim": {"id": "c1", "content": "Q4 revenue totals 162000.", "type": "atomic", "template": "Q4 revenue totals {v:.0f}.", "premise_ids": [], "reasoning": "", "final_node": False},
        },
        {
            "thought": "Compare Q4 with Q3 and bind the quarter-over-quarter growth rate.",
            "code": "q3_revenue = df[df['quarter'] == 'Q3']['revenue'].sum()\ngrowth = (q4_revenue - q3_revenue) / q3_revenue\nc2 = bind('Q4 revenue is {g:.1%} higher than Q3.', g=growth)",
            "claim": {"id": "c2", "content": "Q4 revenue is 18.4% higher than Q3.", "type": "atomic", "template": "Q4 revenue is {g:.1%} higher than Q3.", "premise_ids": [], "reasoning": "", "final_node": False},
        },
        {
            "thought": "Synthesize the revenue and margin facts into the final business interpretation.",
            "code": "c3 = infer([c1, c2], 'The strongest visible opportunity is the Q4 acceleration in revenue.', 'Q4 has the highest revenue and a clear sequential increase over Q3.')\nsubmit_answer([c1, c2, c3])",
            "claim": {"id": "c3", "content": "The strongest visible opportunity is the Q4 acceleration in revenue.", "type": "composite", "template": "", "premise_ids": ["c1", "c2"], "reasoning": "Q4 has the highest revenue and a clear sequential increase over Q3.", "final_node": True},
            "final": True,
        },
    ]

    for idx, item in enumerate(scripted, start=1):
        time.sleep(0.45)
        claims.append(item["claim"])
        if item["claim"].get("type") == "composite":
            for pid in item["claim"].get("premise_ids", []):
                edges.append({"source": pid, "target": item["claim"]["id"]})
        if item.get("final"):
            for claim in claims:
                if claim["id"] in {"c1", "c2", "c3"}:
                    claim["final_node"] = True
        step_payload = {
                "turn": idx,
                "thought": item["thought"],
                "reasoning_summary": "Mock reasoning summary for offline presentation mode.",
                "code": item["code"],
                "result": "Execution Result:\nExecution Status: Success\nOutput:\n```Successfully updated the evidence graph.```",
                "stdout": "Successfully updated the evidence graph.",
                "stderr": "",
                "error": False,
                "graph": snapshot(["c1", "c2", "c3"] if item.get("final") else []),
                "variables": "- df: DataFrame (12, 6)\n- q4_revenue: float = 162000\n- growth: float = 0.184",
                "added_claim_ids": [item["claim"]["id"]],
            }
        session.trace.append(copy.deepcopy(step_payload))
        emit(session, "step", step_payload)

    report = {
        "title": "Retail Metrics Evidence Report",
        "summary": [
            {"text": "Q4 is the strongest revenue period in the dataset.", "claim_ids": ["c1"]},
            {"text": "The sequential Q4 lift indicates a clear growth opportunity.", "claim_ids": ["c2", "c3"]},
        ],
        "sections": [
            {
                "heading": "Evidence-grounded findings",
                "sentences": [
                    {"text": "Q4 revenue totals 162000.", "claim_ids": ["c1"]},
                    {"text": "Q4 revenue is 18.4% higher than Q3.", "claim_ids": ["c2"]},
                    {"text": "The strongest visible opportunity is the Q4 acceleration in revenue.", "claim_ids": ["c3"]},
                ],
            }
        ],
    }
    final_exported = {"graph": {"claims": claims, "edges": edges}, "final_claim_ids": ["c1", "c2", "c3"]}
    final_graph = enrich_graph(final_exported, "- df: DataFrame (12, 6)", session.files)
    output_files = write_run_outputs(session, report=report, final_claims=claims, raw_graph=final_exported, graph=final_graph)
    emit(session, "report", {"report": report, "final_claims": claims, "graph": final_graph, "raw_graph": final_exported, "output": output_files})


def run_session(session: RunSession) -> None:
    try:
        if MOCK_MODE:
            run_mock_agent(session)
        else:
            run_real_agent(session)
    except Exception as exc:
        emit(
            session,
            "error",
            {
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "hint": "Check that the GPT-5.5-compatible Responses API is running at VERIGRAPH_BASE_URL.",
            },
        )
    finally:
        finish(session)


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "VeriGraphDemo/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self.serve_file(STATIC_DIR / "index.html")
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            return self.serve_file((STATIC_DIR / rel).resolve())
        if path == "/api/config":
            return self.send_json(
                {
                    "default_query": DEFAULT_QUERY,
                    "model": MODEL_NAME,
                    "base_url": BASE_URL,
                    "mock": MOCK_MODE,
                    "sample_file": SAMPLE_DATA.name,
                    "finsight_src": str(FINSIGHT_SRC_DIR) if FINSIGHT_SRC_DIR else "",
                    "search_endpoint": SEARCH_ENDPOINT,
                    "browse_endpoint": BROWSE_ENDPOINT,
                    "data_modes": ["local", "hybrid", "web", "finance"],
                    "sample_cases": load_sample_cases(),
                }
            )
        match = re.fullmatch(r"/api/runs/([A-Za-z0-9_-]+)/events", path)
        if match:
            return self.serve_events(match.group(1))
        match = re.fullmatch(r"/api/runs/([A-Za-z0-9_-]+)/result", path)
        if match:
            return self.serve_result(match.group(1))
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            session = self.create_run()
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
            return
        threading.Thread(target=run_session, args=(session,), daemon=True, name=f"run-{session.run_id}").start()
        self.send_json({"run_id": session.run_id})

    def serve_file(self, path: Path) -> None:
        try:
            path = path.resolve()
            if not str(path).startswith(str(STATIC_DIR.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            return

    def create_run(self) -> RunSession:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        query = (form.getfirst("query") or DEFAULT_QUERY).strip()
        use_sample = (form.getfirst("use_sample") or "").lower() in {"1", "true", "yes", "on"}
        data_mode = (form.getfirst("data_mode") or "local").strip().lower()
        if data_mode not in {"local", "hybrid", "web", "finance"}:
            data_mode = "local"
        run_id = uuid.uuid4().hex[:12]
        workspace = RUNS_DIR / run_id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []
        file_items = form["files"] if "files" in form else []
        if not isinstance(file_items, list):
            file_items = [file_items]
        for item in file_items:
            filename = sanitize_filename(getattr(item, "filename", "") or "")
            if not filename:
                continue
            target = workspace / filename
            with target.open("wb") as out:
                shutil.copyfileobj(item.file, out)
            if target.stat().st_size > 0:
                saved_files.append(filename)

        file_optional = data_mode in {"web", "finance"}

        if not saved_files and use_sample and data_mode in {"local", "hybrid"}:
            target = workspace / SAMPLE_DATA.name
            shutil.copyfile(SAMPLE_DATA, target)
            saved_files.append(SAMPLE_DATA.name)

        if saved_files and use_sample and data_mode in {"local", "hybrid"} and SAMPLE_DATA.exists() and SAMPLE_DATA.name not in saved_files:
            target = workspace / SAMPLE_DATA.name
            shutil.copyfile(SAMPLE_DATA, target)
            saved_files.append(SAMPLE_DATA.name)

        if not saved_files and not file_optional:
            raise ValueError("Please upload at least one data file, or enable the sample dataset.")

        session = RunSession(run_id=run_id, query=query, workspace=workspace, files=saved_files, data_mode=data_mode)
        with RUNS_LOCK:
            RUNS[run_id] = session
        return session

    def serve_result(self, run_id: str) -> None:
        # Return the persisted final result so the browser can recover when
        # its SSE stream dropped before the report event arrived.
        output_dir = OUTPUT_DIR / run_id
        report_path = output_dir / "report.json"
        graph_path = output_dir / "graph.json"
        if not report_path.exists():
            with RUNS_LOCK:
                session = RUNS.get(run_id)
            self.send_json({
                "ready": False,
                "running": session is not None and not session.done,
                "run_id": run_id,
            })
            return
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.send_json({"ready": False, "error": str(exc), "run_id": run_id}, status=500)
            return
        graph_view = None
        try:
            if graph_path.exists():
                graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
                graph_view = graph_data.get("view_graph") or graph_data.get("graph")
        except Exception:
            graph_view = None
        payload = {
            "ready": True,
            "run_id": run_id,
            "report": report_data.get("report") if isinstance(report_data, dict) else None,
            "final_claims": report_data.get("final_claims") if isinstance(report_data, dict) else [],
            "graph": graph_view,
            "output": {"output_dir": str(output_dir), "report_md": str(output_dir / "report.md"), "report_json": str(report_path)},
        }
        self.send_json(payload)

    def serve_events(self, run_id: str) -> None:
        with RUNS_LOCK:
            session = RUNS.get(run_id)
        if session is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown run")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                try:
                    event, data = session.events.get(timeout=12)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    if session.done:
                        break
                    continue
                payload = json.dumps(data, ensure_ascii=False)
                self.wfile.write(f"event: {event}\n".encode("utf-8"))
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                if event == "done":
                    self.close_connection = True
                    break
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    host = os.getenv("VERIGRAPH_DEMO_HOST", "127.0.0.1")
    port = int(os.getenv("VERIGRAPH_DEMO_PORT", "7867"))
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"VeriGraph demo running at http://{host}:{port}")
    print(f"Model endpoint: {BASE_URL} | model: {MODEL_NAME} | mock: {MOCK_MODE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
