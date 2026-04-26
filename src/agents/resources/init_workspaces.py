VERIGRAPH_INIT_CODE = r"""
import pandas as pd
import numpy as np
import os
import json
import string
import re
import warnings
import hashlib
import inspect
import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

# --- Workspace path ---
data_dir = "<<SYSTEM_DIR_PLACEHOLDER>>"
if os.path.exists(data_dir):
    os.chdir(data_dir)
else:
    print(f"[Warning] System dir does not exist.")

# --- Core class definitions ---

@dataclass
class Claim:
    id: str
    content: str               # The conclusion/fact content.
    type: str                  # 'atomic' (bound) or 'composite' (inferred)
    template: str = ""         # Original template (only used for atomic claims).
    premise_ids: List[str] = field(default_factory=list) # IDs of dependency claims.
    bound_vars: dict = field(default_factory=dict)       # Bound variables (atomic only).
    reasoning: str = ""        # Reasoning trace (composite only).
    final_node: bool = False   # Whether this claim is part of the final answer.

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        # Compact repr for tool output readability.
        return f'Claim(id={self.id}, content="{self.content}")'

class ClaimGraphManager:
    def __init__(self):
        self.claims: Dict[str, Claim] = {}
        self.counter = 0
        self.finished = False
        # Cap how many claims are displayed in tool context to avoid context explosion.
        self.max_summary_claims = 30
        
    def _generate_id(self):
        self.counter += 1
        return f"c{self.counter}"

    def register_atomic(self, template: str, kwargs: Dict[str, Any]) -> Claim:
        cid = self._generate_id()
        try:
            # Try to format the template.
            content = template.format(**kwargs)
        except Exception as e:
            content = f"[Error formatting: {e}] {{template}}"
            
        claim = Claim(
            id=cid,
            content=content,
            template=template,
            type="atomic",
            bound_vars=kwargs
        )
        self.claims[cid] = claim
        return claim

    def register_composite(self, premises: List[Claim], conclusion: str, reasoning: str) -> Claim:
        cid = self._generate_id()
        premise_ids = [p.id for p in premises if isinstance(p, Claim)]
            
        claim = Claim(
            id=cid,
            content=conclusion,
            type="composite",
            premise_ids=premise_ids,
            reasoning=reasoning
        )
        self.claims[cid] = claim
        return claim

    def export_graph(self) -> str:
        return json.dumps({
            "claims": [c.to_dict() for c in self.claims.values()],
            "edges": [
                {"source": pid, "target": cid} 
                for cid, c in self.claims.items() 
                for pid in c.premise_ids
            ]
        }, ensure_ascii=False, default=str)
    
    def get_final_answer(self) -> List[Dict]:
        return [c.to_dict() for c in self.claims.values() if c.final_node]

    def get_state_summary(self, env_globals: Dict[str, Any] = None) -> str:
        if not self.claims:
            return "No claims established yet."

        # 1. Build a reverse map: Claim instance -> variable name.
        # Only top-level user variables are considered.
        claim_to_var = {}
        if env_globals:
            for var_name, var_val in env_globals.items():
                if var_name.startswith('_'): continue # Skip private/internal names.

                # Check whether the value is a Claim instance.
                if isinstance(var_val, Claim):
                    # Record the shortest variable name pointing at this claim.
                    if var_val.id not in claim_to_var:
                        claim_to_var[var_val.id] = var_name
                    elif len(var_name) < len(claim_to_var[var_val.id]):
                        claim_to_var[var_val.id] = var_name

                # Lists of Claim objects are intentionally not unpacked here.

        # 2. Render textual summary.
        # Limit output to keep tool context small.
        final_ids = [cid for cid, c in self.claims.items() if c.final_node]
        non_final_ids = [cid for cid, c in self.claims.items() if not c.final_node]
        k = int(self.max_summary_claims) if self.max_summary_claims else 0
        tail_non_final_ids = non_final_ids if k <= 0 else non_final_ids[-k:]

        display_ids = []
        display_ids.extend(final_ids)
        for cid in tail_non_final_ids:
            if cid not in display_ids:
                display_ids.append(cid)

        summary_lines = []
        for cid in display_ids:
            claim = self.claims.get(cid)
            if claim is None:
                continue
            # Variable name shown to the agent, "N/A" if not bound at top level.
            var_name = claim_to_var.get(cid, "N/A")

            # Format inspired by source-code comments to be friendly to the agent.
            # [ID] var_name: "Content" (Type)

            prefix = "[Final]" if claim.final_node else f"[{cid}]"

            # Show the variable name when present, otherwise fall back to the id.
            if var_name != "N/A":
                ref_str = f"{var_name} ({cid})"
            else:
                ref_str = f"{cid}"

            type_tag = "Context" if claim.type == 'atomic' else "Inferred"

            # Truncate content to keep tool context bounded.
            content_preview = claim.content
            if len(content_preview) > 100:
                content_preview = content_preview[:100] + "..."

            line = f"{prefix} {ref_str} [{type_tag}]: {content_preview}"
            summary_lines.append(line)
        
        return "\n".join(summary_lines)


_graph_mgr = ClaimGraphManager()

def _truncate_text(s: Any, max_len: int = 200) -> str:
    try:
        txt = str(s)
    except Exception:
        txt = repr(s)
    if max_len is None or max_len <= 0:
        return txt
    return txt if len(txt) <= max_len else (txt[:max_len] + "...[truncated]")

def _preview_value(v: Any) -> Any:
    # Return JSON-serializable preview only.
    try:
        if isinstance(v, Claim):
            return {"type": "Claim", "id": v.id}
        if pd is not None and isinstance(v, pd.DataFrame):
            return {"type": "DataFrame", "shape": list(v.shape), "columns": list(v.columns)[:20]}
        if np is not None and isinstance(v, np.ndarray):
            return {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype)}
        if isinstance(v, (list, tuple, set)):
            vv = list(v)
            return {"type": type(v).__name__, "len": len(vv), "head": [_truncate_text(x, 80) for x in vv[:5]]}
        if isinstance(v, dict):
            keys = list(v.keys())
            return {"type": "dict", "len": len(keys), "keys_head": [_truncate_text(k, 80) for k in keys[:10]]}
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        if isinstance(v, str):
            return _truncate_text(v, 200)
        return _truncate_text(repr(v), 200)
    except Exception:
        return _truncate_text(repr(v), 200)

def _template_fields(template_str: str) -> List[str]:
    fields = []
    for _, field_name, _, _ in string.Formatter().parse(template_str):
        if not field_name:
            continue
        root = field_name.split(".")[0].split("[")[0]
        fields.append(root)
    seen = set()
    out = []
    for f in fields:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

def _has_hardcoded_digits(template_str: str) -> bool:
    # Heuristic: forbid digits outside placeholders to enforce "no hardcoding numbers".
    stripped = []
    in_brace = 0
    for ch in template_str:
        if ch == "{":
            in_brace += 1
        elif ch == "}" and in_brace > 0:
            in_brace -= 1
        elif in_brace == 0:
            stripped.append(ch)
    stripped_text = "".join(stripped)
    # Flag numeric "tokens" like " 25% " or " 2023 " but allow labels like "Q3" (alphanumeric).
    return re.search(r"(^|[^A-Za-z0-9])\d+(\.\d+)?([^A-Za-z0-9]|$)", stripped_text) is not None

def _call_validator(event: Dict[str, Any]) -> Dict[str, Any]:
    # Optional hook for extra validation (e.g., LLM-based). Local (non-LLM) rules for
    # bind()/infer() live here to keep a single source of truth and avoid duplication.
    fn = globals().get("__verigraph_validator__")
    if fn is None:
        return {"ok": True, "reason": "no_validator"}
    # Any exception here should propagate to stop execution (e.g., validator API failure).
    res = fn(event)
    if isinstance(res, dict):
        return res
    return {"ok": False, "reason": f"validator_return_type={type(res)}"}

def _validate_bind_local(template_str: str, kwargs: Dict[str, Any]) -> (bool, str):
    if not isinstance(template_str, str) or template_str.strip() == "":
        return False, "bind: template_str must be a non-empty string"
    fields = _template_fields(template_str)
    if not fields:
        return False, "bind: template_str must contain at least one placeholder like {x}"
    else:
        return True, "ok"
    # missing = [f for f in fields if f not in kwargs]
    # if missing:
    #     return False, f"bind: missing kwargs for placeholders: {missing}"
    # if _has_hardcoded_digits(template_str):
    #     return False, "bind: template_str contains hardcoded digits; compute values and pass via kwargs placeholders"
    # for k, v in kwargs.items():
    #     if isinstance(v, Claim):
    #         return False, f"bind: kwargs[{k}] is a Claim; bind should ground raw data, not claims"
    #     if pd is not None and isinstance(v, pd.DataFrame):
    #         return False, f"bind: kwargs[{k}] is a DataFrame; summarize it to scalars/short text before binding"
    #     if np is not None and isinstance(v, np.ndarray) and getattr(v, "size", 0) > 1000:
    #         return False, f"bind: kwargs[{k}] is a large ndarray (size={v.size}); summarize before binding"
    # return True, "ok"

def _validate_infer_local(premises: Any, conclusion: str, reasoning: str) -> (bool, str, List[Claim]):
    if premises is None:
        premises_list: List[Claim] = []
    elif isinstance(premises, Claim):
        premises_list = [premises]
    elif isinstance(premises, list):
        premises_list = premises
    else:
        return False, f"infer: premises must be Claim or List[Claim], got {type(premises)}", []

    for i, p in enumerate(premises_list):
        if not isinstance(p, Claim):
            return False, f"infer: premises[{i}] is not a Claim (got {type(p)})", []

    if not isinstance(conclusion, str) or conclusion.strip() == "":
        return False, "infer: conclusion must be a non-empty string", []
    if not isinstance(reasoning, str) or reasoning.strip() == "":
        return False, "infer: reasoning must be a non-empty string", []
    if len(reasoning) > 2000:
        return False, "infer: reasoning is too long; keep it concise and rely on premises", []
    return True, "ok", premises_list

def bind(template_str: str, **kwargs) -> Claim:
    ok, msg = _validate_bind_local(template_str, kwargs)
    if not ok:
        # print(f"[VeriGraph][bind][REJECT][LOCAL] {msg}")
        raise ValueError(msg)

    try:
        rendered = template_str.format(**kwargs)
    except Exception as e:
        msg = f"bind: template format failed: {e}"
        # print(f"[VeriGraph][bind][REJECT][LOCAL] {msg}")
        raise ValueError(msg)

    event = {
        "op": "bind",
        "template": _truncate_text(template_str, 500),
        "rendered": _truncate_text(rendered, 500),
        "kwargs_preview": {k: _preview_value(v) for k, v in kwargs.items()},
    }
    verdict = _call_validator(event)
    if not verdict.get("ok", False):
        reason = verdict.get("reason", "validator_reject")
        fix = verdict.get("fix_suggestion", "")
        full_msg = f"bind rejected: {reason}" + (f" | fix: {fix}" if fix else "")
        # print(f"[VeriGraph][bind][REJECT] {full_msg}")
        raise ValueError(full_msg)

    claim = _graph_mgr.register_atomic(template_str, kwargs)
    # print(f"[VeriGraph][bind][OK] {claim.id}: {_truncate_text(claim.content, 200)}")
    print("Successfully bound a new claim.")
    return claim

def infer(premises: List[Claim], conclusion: str, reasoning: str) -> Claim:
    ok, msg, premises_list = _validate_infer_local(premises, conclusion, reasoning)
    if not ok:
        # print(f"[VeriGraph][infer][REJECT][LOCAL] {msg}")
        raise ValueError(msg)

    event = {
        "op": "infer",
        "premises": [{"id": p.id, "content": _truncate_text(p.content, 300), "type": p.type} for p in premises_list],
        "conclusion": _truncate_text(conclusion, 500),
        "reasoning": _truncate_text(reasoning, 800),
    }
    verdict = _call_validator(event)
    if not verdict.get("ok", False):
        reason = verdict.get("reason", "validator_reject")
        fix = verdict.get("fix_suggestion", "")
        full_msg = f"infer rejected: {reason}" + (f" | fix: {fix}" if fix else "")
        # print(f"[VeriGraph][infer][REJECT] {full_msg}")
        raise ValueError(full_msg)

    claim = _graph_mgr.register_composite(premises_list, conclusion, reasoning)
    print("Successfully inferred a new claim.")
    return claim

def submit_answer(final_claims: Union[List[Claim], Claim]) -> None:
    if _graph_mgr.finished:
        print("[System] Answer already submitted. Ignore.")
        return

    # Allow callers to pass a single Claim instead of a list.
    if isinstance(final_claims, Claim):
        final_claims = [final_claims]
    
    if not isinstance(final_claims, list):
        print(f"[Error] submit_answer expects a List[Claim], got {type(final_claims)}")
        return

    print("=== FINAL ANSWER SUBMISSION ===")
    for i, claim in enumerate(final_claims):
        if isinstance(claim, Claim):
            claim.final_node = True
            print(f"{i+1}. {claim.content}")
        else:
            print(f"{i+1}. [Invalid Object: {type(claim)}]")
    
    _graph_mgr.finished = True
    print("===============================")

def get_flow_report():
    return _graph_mgr.export_graph()
"""


BASE_INIT_CODE = r"""
import pandas as pd
import numpy as np
import os
import json
import warnings
import hashlib
import inspect
import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

data_dir = "<<SYSTEM_DIR_PLACEHOLDER>>"
if os.path.exists(data_dir):
    os.chdir(data_dir)
else:
    print(f"[Warning] System dir {{data_dir}} does not exist.")
"""



BASE_INIT_CODE = r"""
import pandas as pd
import numpy as np
import os
import json
import warnings
import hashlib
import inspect
import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

data_dir = "<<SYSTEM_DIR_PLACEHOLDER>>"
if os.path.exists(data_dir):
    os.chdir(data_dir)
else:
    print(f"[Warning] System dir {{data_dir}} does not exist.")
"""
