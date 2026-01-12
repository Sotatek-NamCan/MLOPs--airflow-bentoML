from __future__ import annotations

import difflib
from typing import Any, Iterable, Mapping


def ensure_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{context} must be a JSON object (mapping).")


def validate_config_keys(
    raw: Mapping[str, Any] | None,
    allowed_keys: Iterable[str],
    *,
    context: str,
) -> None:
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a JSON object (mapping).")
    if not raw:
        return
    allowed = sorted(set(allowed_keys))
    unknown = sorted(key for key in raw.keys() if key not in allowed)
    if not unknown:
        return

    suggestions = []
    for key in unknown:
        matches = difflib.get_close_matches(key, allowed, n=3, cutoff=0.6)
        if matches:
            suggestions.append(f"{key} -> {', '.join(matches)}")
    suggestion_text = f" Did you mean: {'; '.join(suggestions)}?" if suggestions else ""

    preview_limit = 12
    preview = ", ".join(allowed[:preview_limit])
    remainder = len(allowed) - preview_limit
    more_text = f", ... (+{remainder} more)" if remainder > 0 else ""
    raise ValueError(
        f"Unsupported parameter name(s) in {context}: {', '.join(unknown)}."
        f"{suggestion_text} Valid keys include: {preview}{more_text}"
    )
