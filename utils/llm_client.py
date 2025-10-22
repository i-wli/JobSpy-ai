from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# Built-in pricing table (per 1M tokens) for common models
PRICING_PER_1M: Dict[str, Tuple[float, float]] = {
    # key -> (input_per_1M_usd, output_per_1M_usd)
    "gpt-5": (1.25, 10.00),
    "gpt5": (1.25, 10.00),
    "gpt-5-chat-latest": (1.25, 10.00),
    "gpt5chatlatest": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt5-nano": (0.05, 0.40),
    "gpt-nano": (0.05, 0.40),
    "o3": (2.00, 8.00),
    # Gemini (Vertex)
    "gemini-2-5-pro": (1.25, 10.00),
    "gemini-2-5-flash-lite": (0.10, 0.40),
    "gemini-2-5-flash": (0.30, 2.50),
}


def normalize_model_key(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s


def infer_per_1k_prices(model_name: str) -> Tuple[float, float, Optional[str]]:
    key = normalize_model_key(model_name)
    if key in PRICING_PER_1M:
        pin_1m, pout_1m = PRICING_PER_1M[key]
        return pin_1m / 1000.0, pout_1m / 1000.0, key
    if key.startswith("gpt5-"):
        maybe = key.replace("gpt5-", "gpt-5-")
        if maybe in PRICING_PER_1M:
            pin_1m, pout_1m = PRICING_PER_1M[maybe]
            return pin_1m / 1000.0, pout_1m / 1000.0, maybe
    return 0.0, 0.0, None


def estimate_cost(usage: Dict[str, int], model: str) -> Tuple[float, str, float, float]:
    """Return (estimated_cost_usd, label, price_in_per_1k, price_out_per_1k).

    For Gemini models, charges output rate on (output + thinking) tokens.
    """
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    think_tok = int(usage.get("thinking_tokens", 0) or 0)
    pin_per_1k, pout_per_1k, inferred_key = infer_per_1k_prices(model)
    model_key = normalize_model_key(model)
    cost = 0.0
    if model_key.startswith("gemini") and (pin_per_1k or pout_per_1k):
        cost = (in_tok / 1000.0) * pin_per_1k + ((out_tok + think_tok) / 1000.0) * pout_per_1k
    elif pin_per_1k or pout_per_1k:
        cost = (in_tok / 1000.0) * pin_per_1k + (out_tok / 1000.0) * pout_per_1k
    return cost, (inferred_key or model_key), pin_per_1k, pout_per_1k


def call_gemini(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    thinking_budget: int = -1,
    preview: bool = False,
) -> Tuple[str, Dict[str, int]]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set (Vertex AI API key)")

    system_msgs = [m.get("content", "") for m in messages if m.get("role") in ("developer", "system")]
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    system_instruction = "\n\n".join([s for s in system_msgs if s]).strip() or None
    user_text = "\n\n".join([u for u in user_msgs if u]).strip()

    if preview:
        try:
            print(json.dumps({"system_instruction": system_instruction, "user": user_text[:10000]}, indent=2))
        except Exception:
            print(user_text[:2000])

    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    client = genai.Client(vertexai=True, api_key=api_key)

    cfg_kwargs: Dict[str, Any] = {
        "temperature": float(temperature),
        "system_instruction": system_instruction,
        "response_mime_type": "application/json",
    }
    if max_tokens is not None:
        cfg_kwargs["max_output_tokens"] = int(max_tokens)
    try:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(thinking_budget))
    except Exception:
        pass
    try:
        config = types.GenerateContentConfig(**cfg_kwargs)
    except TypeError:
        _cfg = dict(cfg_kwargs)
        _cfg.pop("response_mime_type", None)
        try:
            config = types.GenerateContentConfig(**_cfg)
        except TypeError:
            _cfg.pop("thinking_config", None)
            config = types.GenerateContentConfig(**_cfg)

    resp = client.models.generate_content(
        model=model,
        contents=user_text,
        config=config,
    )

    text = getattr(resp, "text", "") or ""
    text = text.strip()
    if not text:
        raise RuntimeError("Empty response from Gemini (Vertex AI)")

    usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
    try:
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            pt = getattr(um, "prompt_token_count", None)
            ct = getattr(um, "candidates_token_count", None)
            tt = getattr(um, "total_token_count", None)
            th = None
            for k in ("thoughts_token_count", "thinking_token_count", "reasoning_token_count"):
                v = getattr(um, k, None)
                if v is not None:
                    th = v
                    break
            if pt is not None:
                usage["input_tokens"] = int(pt)
            if ct is not None:
                usage["output_tokens"] = int(ct)
            if th is not None:
                usage["thinking_tokens"] = int(th)
            if tt is not None:
                usage["total_tokens"] = int(tt)
    except Exception:
        pass

    return text, usage


def call_openai(
    input_obj: Any,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    effort: str = "medium",
    verbosity: str = "low",
    preview: bool = False,
    response_schema: Optional[Dict[str, Any]] = None,
    response_format_type: str = "json_schema",
) -> Tuple[str, Dict[str, int]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)

    if preview:
        try:
            print(json.dumps(input_obj, indent=2))
        except TypeError:
            print(input_obj)

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": input_obj,
        "temperature": float(temperature),
        "reasoning": {"effort": str(effort)},
    }
    if response_schema is not None:
        kwargs["response_format"] = {
            "type": response_format_type,
            "json_schema": {"name": "schema", "schema": copy.deepcopy(response_schema)},
        }
    if verbosity:
        kwargs["verbosity"] = str(verbosity)
    if max_tokens is not None:
        kwargs["max_output_tokens"] = int(max_tokens)

    def _extract_text_from_responses(resp: Any) -> str:
        try:
            txt = (getattr(resp, "output_text", None) or "").strip()
            if txt:
                return txt
        except Exception:
            pass
        try:
            output = getattr(resp, "output", [])
            if output and hasattr(output[0], "content"):
                contents = output[0].content  # type: ignore[attr-defined]
                if contents and hasattr(contents[0], "text"):
                    return contents[0].text  # type: ignore[attr-defined]
        except Exception:
            pass
        return ""

    def _extract_usage(resp: Any) -> Dict[str, int]:
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            u = getattr(resp, "usage", None)
            if u is None:
                return usage
            it = getattr(u, "input_tokens", None)
            ot = getattr(u, "output_tokens", None)
            tt = getattr(u, "total_tokens", None)
            if it is None:
                it = getattr(u, "prompt_tokens", None)
            if ot is None:
                ot = getattr(u, "completion_tokens", None)
            if tt is None and (it is not None or ot is not None):
                try:
                    tt = int((it or 0)) + int((ot or 0))
                except Exception:
                    tt = None
            if it is not None:
                usage["input_tokens"] = int(it)
            if ot is not None:
                usage["output_tokens"] = int(ot)
            if tt is not None:
                usage["total_tokens"] = int(tt)
        except Exception:
            pass
        return usage

    try:
        resp = client.responses.create(**kwargs)
        text = _extract_text_from_responses(resp)
        usage = _extract_usage(resp)
        return text, usage
    except TypeError:
        simplified = {k: v for k, v in kwargs.items() if k not in {"reasoning", "verbosity", "max_output_tokens", "temperature", "response_format"}}
        resp = client.responses.create(**simplified)
        text = _extract_text_from_responses(resp)
        usage = _extract_usage(resp)
        return text, usage


__all__ = [
    "PRICING_PER_1M",
    "normalize_model_key",
    "infer_per_1k_prices",
    "estimate_cost",
    "call_gemini",
    "call_openai",
]

