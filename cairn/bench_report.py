#!/usr/bin/env python3
"""
cairn-bench-report: Parse Claude Code JSONL transcripts for token and time metrics.

Compares A/B sessions (with vs without Cairn) to measure overhead.

Usage:
  cairn-bench-report <session.jsonl>              # single session summary
  cairn-bench-report <arm-a.jsonl> <arm-b.jsonl>  # side-by-side comparison
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SessionMetrics:
    path: str
    turns: int = 0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    total_duration_ms: int = 0
    hook_duration_ms: int = 0  # approximate: sum of turn_duration minus api latency (not available separately)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens + self.output_tokens

    @property
    def effective_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens

    @property
    def cost_usd(self) -> float:
        # claude-sonnet-4-6 pricing (per million tokens)
        input_rate = 3.00
        cache_write_rate = 3.75
        cache_read_rate = 0.30
        output_rate = 15.00
        return (
            self.input_tokens * input_rate
            + self.cache_creation_tokens * cache_write_rate
            + self.cache_read_tokens * cache_read_rate
            + self.output_tokens * output_rate
        ) / 1_000_000


def parse_jsonl(path: str) -> SessionMetrics:
    metrics = SessionMetrics(path=path)
    seen_messages = set()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")

            # Turn duration system messages
            if entry_type == "system" and entry.get("subtype") == "turn_duration":
                metrics.total_duration_ms += entry.get("durationMs", 0)
                continue

            # Assistant messages with usage data
            if entry_type == "assistant":
                msg = entry.get("message", {})
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("id", "")
                if msg_id and msg_id in seen_messages:
                    continue
                if msg_id:
                    seen_messages.add(msg_id)

                usage = msg.get("usage", {})
                if not usage:
                    continue

                metrics.turns += 1
                metrics.input_tokens += usage.get("input_tokens", 0)
                metrics.cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
                metrics.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
                metrics.output_tokens += usage.get("output_tokens", 0)

    return metrics


def _pct(value: float, base: float) -> str:
    if base == 0:
        return "  n/a"
    delta = (value - base) / base * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:+.0f}%".replace("+-", "-")


def _fmt_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms/1000:.1f}s"


def _fmt_k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def print_single(m: SessionMetrics) -> None:
    print(f"\nSession: {Path(m.path).name}")
    print(f"  Turns:              {m.turns}")
    print(f"  Input tokens:       {_fmt_k(m.input_tokens)}")
    print(f"  Cache write tokens: {_fmt_k(m.cache_creation_tokens)}")
    print(f"  Cache read tokens:  {_fmt_k(m.cache_read_tokens)}")
    print(f"  Output tokens:      {_fmt_k(m.output_tokens)}")
    print(f"  Effective input:    {_fmt_k(m.effective_input_tokens)}")
    print(f"  Total duration:     {_fmt_ms(m.total_duration_ms)}")
    print(f"  Est. cost:          ${m.cost_usd:.4f}")


def print_comparison(a: SessionMetrics, b: SessionMetrics, label_a: str = "A (with Cairn)", label_b: str = "B (without)") -> None:
    col_w = 22
    a_w = 14
    b_w = 14
    d_w = 10

    header = f"{'Metric':<{col_w}} {label_a:>{a_w}} {label_b:>{b_w}} {'Delta':>{d_w}}"
    print()
    print(header)
    print("-" * len(header))

    rows = [
        ("Turns", a.turns, b.turns, False),
        ("Input tokens", a.input_tokens, b.input_tokens, True),
        ("Cache write tokens", a.cache_creation_tokens, b.cache_creation_tokens, True),
        ("Cache read tokens", a.cache_read_tokens, b.cache_read_tokens, True),
        ("Output tokens", a.output_tokens, b.output_tokens, True),
        ("Effective input", a.effective_input_tokens, b.effective_input_tokens, True),
        ("Total tokens", a.total_tokens, b.total_tokens, True),
        ("Duration", a.total_duration_ms, b.total_duration_ms, False),
        ("Est. cost USD", None, None, False),
    ]

    for label, av, bv, is_tok in rows:
        if label == "Est. cost USD":
            av_str = f"${a.cost_usd:.4f}"
            bv_str = f"${b.cost_usd:.4f}"
            d_str = _pct(b.cost_usd, a.cost_usd)
        elif label == "Duration":
            av_str = _fmt_ms(av)
            bv_str = _fmt_ms(bv)
            d_str = _pct(bv, av)
        elif label == "Turns":
            av_str = str(av)
            bv_str = str(bv)
            d_str = _pct(bv, av)
        else:
            av_str = _fmt_k(av)
            bv_str = _fmt_k(bv)
            d_str = _pct(bv, av)

        print(f"{label:<{col_w}} {av_str:>{a_w}} {bv_str:>{b_w}} {d_str:>{d_w}}")

    print()
    print(f"  A session: {Path(a.path).name}")
    print(f"  B session: {Path(b.path).name}")
    print()


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if len(args) == 1:
        m = parse_jsonl(args[0])
        print_single(m)
    elif len(args) == 2:
        label_a = "A (with Cairn)"
        label_b = "B (without)"
        # Allow --label-a / --label-b overrides via simple positional convention
        a = parse_jsonl(args[0])
        b = parse_jsonl(args[1])
        print_comparison(a, b, label_a, label_b)
    else:
        print("Usage: cairn-bench-report <a.jsonl> [<b.jsonl>]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
