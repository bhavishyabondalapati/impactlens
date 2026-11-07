"""LLM access with token logging and a hard per-run cost cap.

Every call goes through ``CostTracker.charge``. BEFORE a call we check the worst case (estimated
input + the full max_tokens of output); if that could take the run past the cap we raise
``BudgetExceeded`` and never send the request. So a run can never spend more than the cap.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-haiku-4-5"

# US dollars per million tokens (Anthropic first-party API). Cache writes cost 1.25x input,
# cache reads 0.1x input.
PRICES = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "fake": {"input": 1.00, "output": 5.00},  # tests: priced like Haiku 4.5
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class CostTracker:
    model: str
    budget_usd: float = 0.10
    log_path: Path | None = None
    spent_usd: float = 0.0
    calls: list[dict] = field(default_factory=list)

    @property
    def price(self) -> dict:
        return PRICES[self.model]

    def cost(self, u: Usage) -> float:
        p = self.price
        return (u.input_tokens * p["input"] + u.output_tokens * p["output"]
                + u.cache_creation_input_tokens * p["input"] * 1.25
                + u.cache_read_input_tokens * p["input"] * 0.10) / 1e6  # fmt: skip

    def check(self, prompt_chars: int, max_tokens: int) -> None:
        """Raise if the worst case of the next call could exceed the budget."""
        est_input = prompt_chars / 3.0  # generous: English is ~4 chars per token
        worst = (est_input * self.price["input"] * 1.25 + max_tokens * self.price["output"]) / 1e6
        if self.spent_usd + worst > self.budget_usd:
            raise BudgetExceeded(
                f"next call could cost up to ${worst:.4f}; already spent ${self.spent_usd:.4f} "
                f"of the ${self.budget_usd:.2f} cap. Stopping the run."
            )

    def charge(self, step: str, u: Usage) -> float:
        c = self.cost(u)
        self.spent_usd += c
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step, "model": self.model,
               **u.__dict__, "cost_usd": round(c, 6),
               "run_total_usd": round(self.spent_usd, 6)}  # fmt: skip
        self.calls.append(row)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return c

    def totals(self) -> dict:
        keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                "cache_read_input_tokens")  # fmt: skip
        return {"model": self.model, "calls": len(self.calls),
                **{k: sum(c[k] for c in self.calls) for k in keys},
                "cost_usd": round(self.spent_usd, 6), "budget_usd": self.budget_usd}  # fmt: skip


class AnthropicLLM:
    """``complete(shared, task, step)``: ``shared`` is the cacheable prefix common to every
    report (rules, facts, SOP chunks); ``task`` is the per-report instruction."""

    def __init__(self, tracker: CostTracker, max_tokens: int = 2500):
        import anthropic

        from impactlens.geo.postgis import load_dotenv

        load_dotenv()  # ANTHROPIC_API_KEY lives only in .env
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY missing: add it to .env (see .env.example)")
        self.client = anthropic.Anthropic()
        self.tracker, self.max_tokens = tracker, max_tokens

    def complete(self, system: str, shared: str, task: str, step: str) -> str:
        self.tracker.check(len(system) + len(shared) + len(task), self.max_tokens)
        response = self.client.messages.create(
            model=self.tracker.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": [
                # identical for all three reports -> cached after the first call (>= 4096 tokens)
                {"type": "text", "text": shared, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": task},
            ]}],
        )  # fmt: skip
        u = response.usage
        self.tracker.charge(step, Usage(u.input_tokens, u.output_tokens,
                                        u.cache_creation_input_tokens or 0,
                                        u.cache_read_input_tokens or 0))  # fmt: skip
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{step}: report cut off at max_tokens={self.max_tokens}")
        return "".join(b.text for b in response.content if b.type == "text")


class FakeLLM:
    """Test double: returns scripted texts in order and charges a fixed, made-up usage."""

    def __init__(self, tracker: CostTracker, responses: list[str] | None = None, respond=None,
                 usage: Usage | None = None):  # fmt: skip
        self.tracker, self.responses, self.respond = tracker, list(responses or []), respond
        self.usage = usage or Usage(input_tokens=3000, output_tokens=800)
        self.prompts: list[dict] = []

    def complete(self, system: str, shared: str, task: str, step: str) -> str:
        self.tracker.check(len(system) + len(shared) + len(task), 2500)
        self.prompts.append({"system": system, "shared": shared, "task": task, "step": step})
        self.tracker.charge(step, self.usage)
        if self.respond:
            return self.respond(system, shared, task, step)
        return self.responses.pop(0)
