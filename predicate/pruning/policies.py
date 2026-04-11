"""
Deterministic pruning policies for browser-agent snapshots.

Supports relaxation levels for over-pruning recovery:
- Level 0: Strict category-specific pruning (default)
- Level 1: Relaxed - allow more interactive roles
- Level 2: Loose - include nearly all interactive elements
- Level 3+: Fallback to generic (minimal pruning)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .types import PruningTaskCategory

NodePredicate = Callable[[Any, str], bool]


@dataclass(frozen=True)
class PruningPolicy:
    """Simple allow/block policy for an initial pruning category."""

    category: PruningTaskCategory
    max_nodes: int
    allow: NodePredicate
    block: NodePredicate
    relaxation_level: int = 0

    def with_relaxation(self, level: int) -> "PruningPolicy":
        """Return a new policy with the specified relaxation level."""
        return PruningPolicy(
            category=self.category,
            max_nodes=self.max_nodes + (level * 15),  # Increase budget per level
            allow=self.allow,
            block=self.block,
            relaxation_level=level,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _text(el: Any) -> str:
    return str(getattr(el, "text", "") or "").lower()


def _href(el: Any) -> str:
    return str(getattr(el, "href", "") or "").lower()


def _nearby_text(el: Any) -> str:
    return str(getattr(el, "nearby_text", "") or "").lower()


def _role(el: Any) -> str:
    return str(getattr(el, "role", "") or "").lower()


def _is_interactive(el: Any) -> bool:
    """Check if element has an interactive role."""
    role = _role(el)
    return role in {
        "button", "link", "textbox", "searchbox", "combobox",
        "checkbox", "radio", "slider", "tab", "menuitem",
        "option", "switch", "cell", "input", "select", "textarea",
    }


# ---------------------------------------------------------------------------
# Category-specific allow predicates
# ---------------------------------------------------------------------------

def _allow_shopping(el: Any, goal: str) -> bool:
    text = _text(el)
    nearby = _nearby_text(el)
    href = _href(el)
    role = _role(el)

    if role in {"button", "link", "textbox", "searchbox", "combobox"} and (
        "add to cart" in text
        or "add to bag" in text
        or "buy now" in text
        or "checkout" in text
        or "cart" in text
    ):
        return True
    if role == "link" and href and getattr(el, "in_dominant_group", False):
        return True
    if "$" in text or "price" in nearby:
        return True
    if role in {"textbox", "searchbox", "combobox"} and "search" in text:
        return True
    if getattr(el, "in_dominant_group", False) and len(text.strip()) >= 3:
        return True
    return False


def _allow_shopping_relaxed(el: Any, goal: str) -> bool:
    """Level 1 relaxation: include more interactive elements."""
    if _allow_shopping(el, goal):
        return True
    role = _role(el)
    # Add more buttons and links even outside dominant group
    if role in {"button", "link"} and len(_text(el).strip()) >= 2:
        return True
    # Include quantity selectors, size selectors
    if role in {"select", "combobox", "listbox"}:
        return True
    return False


def _allow_shopping_loose(el: Any, goal: str) -> bool:
    """Level 2 relaxation: include nearly all interactive elements."""
    if _allow_shopping_relaxed(el, goal):
        return True
    return _is_interactive(el)


def _allow_form_filling(el: Any, goal: str) -> bool:
    text = _text(el)
    role = _role(el)
    if role in {"textbox", "searchbox", "combobox", "checkbox", "radio", "textarea"}:
        return True
    if role == "button" and any(token in text for token in ("submit", "send", "continue", "sign up")):
        return True
    return False


def _allow_form_filling_relaxed(el: Any, goal: str) -> bool:
    """Level 1 relaxation for form filling."""
    if _allow_form_filling(el, goal):
        return True
    role = _role(el)
    # Include all buttons and selects
    if role in {"button", "select", "listbox", "option"}:
        return True
    return False


def _allow_search(el: Any, goal: str) -> bool:
    text = _text(el)
    role = _role(el)
    href = _href(el)
    if role in {"searchbox", "textbox", "combobox"}:
        return True
    if role == "button" and "search" in text:
        return True
    if role == "link" and href:
        return True
    return False


def _allow_search_relaxed(el: Any, goal: str) -> bool:
    """Level 1 relaxation for search."""
    if _allow_search(el, goal):
        return True
    role = _role(el)
    if role in {"button", "tab", "menuitem"}:
        return True
    return False


def _allow_generic(el: Any, goal: str) -> bool:
    role = _role(el)
    return role in {"button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio"}


def _allow_generic_relaxed(el: Any, goal: str) -> bool:
    """Relaxed generic - include all interactive elements."""
    return _is_interactive(el)


def _block_common(el: Any, goal: str) -> bool:
    text = _text(el)
    href = _href(el)
    return any(token in text for token in ("privacy policy", "terms", "cookie policy")) or any(
        token in href for token in ("/privacy", "/terms", "/cookies")
    )


def _block_nothing(el: Any, goal: str) -> bool:
    """At high relaxation levels, don't block anything."""
    return False


# ---------------------------------------------------------------------------
# Policy factory with relaxation support
# ---------------------------------------------------------------------------

def get_pruning_policy(
    category: PruningTaskCategory,
    relaxation_level: int = 0,
) -> PruningPolicy:
    """
    Return the deterministic policy for a category with optional relaxation.

    Args:
        category: The task category
        relaxation_level: 0=strict, 1=relaxed, 2=loose, 3+=fallback

    Returns:
        PruningPolicy configured for the category and relaxation level
    """
    # At level 3+, fall back to generic with no blocking
    if relaxation_level >= 3:
        return PruningPolicy(
            category=category,
            max_nodes=80,
            allow=_allow_generic_relaxed,
            block=_block_nothing,
            relaxation_level=relaxation_level,
        )

    if category in {PruningTaskCategory.SHOPPING, PruningTaskCategory.CHECKOUT}:
        if relaxation_level == 0:
            allow_fn = _allow_shopping
            max_nodes = 25
        elif relaxation_level == 1:
            allow_fn = _allow_shopping_relaxed
            max_nodes = 40
        else:  # level 2
            allow_fn = _allow_shopping_loose
            max_nodes = 60
        return PruningPolicy(
            category=category,
            max_nodes=max_nodes,
            allow=allow_fn,
            block=_block_common if relaxation_level < 2 else _block_nothing,
            relaxation_level=relaxation_level,
        )

    if category == PruningTaskCategory.FORM_FILLING:
        if relaxation_level == 0:
            allow_fn = _allow_form_filling
            max_nodes = 20
        else:
            allow_fn = _allow_form_filling_relaxed
            max_nodes = 35 + (relaxation_level * 10)
        return PruningPolicy(
            category=category,
            max_nodes=max_nodes,
            allow=allow_fn,
            block=_block_common if relaxation_level == 0 else _block_nothing,
            relaxation_level=relaxation_level,
        )

    if category == PruningTaskCategory.SEARCH:
        if relaxation_level == 0:
            allow_fn = _allow_search
            max_nodes = 20
        else:
            allow_fn = _allow_search_relaxed
            max_nodes = 35 + (relaxation_level * 10)
        return PruningPolicy(
            category=category,
            max_nodes=max_nodes,
            allow=allow_fn,
            block=_block_common if relaxation_level == 0 else _block_nothing,
            relaxation_level=relaxation_level,
        )

    # Generic or other categories
    if relaxation_level == 0:
        allow_fn = _allow_generic
        max_nodes = 20
    else:
        allow_fn = _allow_generic_relaxed
        max_nodes = 40 + (relaxation_level * 15)
    return PruningPolicy(
        category=category,
        max_nodes=max_nodes,
        allow=allow_fn,
        block=_block_common if relaxation_level == 0 else _block_nothing,
        relaxation_level=relaxation_level,
    )
