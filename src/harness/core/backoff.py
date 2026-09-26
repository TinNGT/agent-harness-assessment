from __future__ import annotations

import random


def compute_delay(attempt: int, *, base: float = 0.05, cap: float = 2.0) -> float:
    """Exponential backoff with full jitter (AWS-style): a random value in
    [0, min(cap, base * 2**(attempt-1))]. `attempt` is the 1-indexed attempt
    that just failed; the delay is applied before the next one.

    Full jitter (rather than a fixed exponential delay) avoids every retrying
    caller waking up at the same instant and re-hammering a struggling
    dependency in lockstep. `base` defaults small (50ms) so a handful of
    retries in a test suite stay fast without needing a fake clock — unlike
    step/deadline budgets, backoff delay is real wall-clock time by design,
    not something tests need to assert on.
    """
    exp = min(cap, base * (2 ** (attempt - 1)))
    return random.uniform(0, exp)
