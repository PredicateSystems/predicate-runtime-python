"""
SentienceContext: Token-Slasher Context Middleware for browser-use.

This module provides a compact, ranked DOM context block for browser-use agents,
reducing tokens and improving reliability by using Sentience snapshots.

Example usage:
    from browser_use import Agent
    from predicate.backends import PredicateContext

    ctx = PredicateContext(show_overlay=True)
    state = await ctx.build(agent.browser_session, goal="Click the first Show HN post")
    if state:
        agent.add_context(state.prompt_block)  # or however browser-use injects state
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ..constants import PREDICATE_API_URL

if TYPE_CHECKING:
    from ..models import Element, Snapshot

logger = logging.getLogger(__name__)


@dataclass
class SentienceContextState:
    """Sentience context state with snapshot and formatted prompt block."""

    url: str
    snapshot: Snapshot
    prompt_block: str


@dataclass
class TopElementSelector:
    """
    Configuration for element selection strategy.

    The selector uses a 3-way merge to pick elements for the LLM context:
    1. Top N by importance score (most actionable elements)
    2. Top N from dominant group (for ordinal tasks like "click 3rd item")
    3. Top N by position (elements at top of page, lowest doc_y)

    Elements are deduplicated across all three sources.
    """

    by_importance: int = 60
    """Number of top elements to select by importance score (descending)."""

    from_dominant_group: int = 15
    """Number of top elements to select from the dominant group (for ordinal tasks)."""

    by_position: int = 10
    """Number of top elements to select by position (lowest doc_y = top of page)."""


class SentienceContext:
    """
    Token-Slasher Context Middleware for browser-use.

    Creates a compact, ranked DOM context block using Sentience snapshots,
    reducing tokens and improving reliability for LLM-based browser agents.

    Example:
        from browser_use import Agent
        from predicate.backends import SentienceContext

        ctx = SentienceContext(show_overlay=True)
        state = await ctx.build(agent.browser_session, goal="Click the first Show HN post")
        if state:
            agent.add_context(state.prompt_block)
    """

    # Sentience API endpoint
    API_URL = PREDICATE_API_URL

    def __init__(
        self,
        *,
        predicate_api_key: str | None = None,
        sentience_api_key: str | None = None,
        use_api: bool | None = None,
        max_elements: int = 60,
        show_overlay: bool = False,
        top_element_selector: TopElementSelector | None = None,
    ) -> None:
        """
        Initialize SentienceContext.

        Args:
            predicate_api_key: Canonical API key parameter for gateway mode.
            sentience_api_key: Backward-compatible API key alias (legacy name).
            use_api: Force API vs extension mode (auto-detected if None)
            max_elements: Maximum elements to fetch from snapshot
            show_overlay: Show visual overlay highlighting elements in browser
            top_element_selector: Configuration for element selection strategy
        """
        self._api_key = predicate_api_key or sentience_api_key
        self._use_api = use_api
        self._max_elements = max_elements
        self._show_overlay = show_overlay
        self._selector = top_element_selector or TopElementSelector()

    async def build(
        self,
        browser_session: Any,
        *,
        goal: str | None = None,
        wait_for_extension_ms: int = 5000,
        retries: int = 2,
        retry_delay_s: float = 1.0,
    ) -> SentienceContextState | None:
        """
        Build context state from browser session.

        Takes a snapshot using the Sentience extension and formats it for LLM consumption.
        Returns None if snapshot fails (extension not loaded, timeout, etc.).

        Args:
            browser_session: Browser-use BrowserSession instance
            goal: Optional goal/task description (passed to gateway for reranking)
            wait_for_extension_ms: Maximum time to wait for extension injection
            retries: Number of retry attempts on snapshot failure
            retry_delay_s: Delay between retries in seconds

        Returns:
            SentienceContextState with snapshot and formatted prompt, or None if failed
        """
        try:
            # Import here to avoid requiring sentience as a hard dependency
            from ..models import SnapshotOptions
            from .browser_use_adapter import BrowserUseAdapter
            from .snapshot import snapshot

            # Create adapter and backend
            adapter = BrowserUseAdapter(browser_session)
            backend = await adapter.create_backend()

            # Wait for extension to inject (poll until ready or timeout)
            await self._wait_for_extension(backend, timeout_ms=wait_for_extension_ms)

            # Build snapshot options
            options = SnapshotOptions(
                limit=self._max_elements,
                show_overlay=self._show_overlay,
                goal=goal,
            )

            # Set API options
            if self._api_key:
                options.predicate_api_key = self._api_key
            if self._use_api is not None:
                options.use_api = self._use_api
            elif self._api_key:
                options.use_api = True

            # Take snapshot with retry logic
            snap = None
            last_error: Exception | None = None

            for attempt in range(retries):
                try:
                    snap = await snapshot(backend, options=options)
                    break  # Success
                except Exception as e:
                    last_error = e
                    if attempt < retries - 1:
                        logger.debug(
                            "Sentience snapshot attempt %d failed: %s, retrying...",
                            attempt + 1,
                            e,
                        )
                        await asyncio.sleep(retry_delay_s)
                    else:
                        logger.warning(
                            "Sentience snapshot failed after %d attempts: %s",
                            retries,
                            last_error,
                        )
                        return None

            if snap is None:
                logger.warning("Sentience snapshot returned None")
                return None

            # Get URL from snapshot
            url = snap.url or ""

            # Format for LLM
            formatted = self._format_snapshot_for_llm(snap)

            # Build prompt block
            prompt = (
                "Elements: ID|role|text|imp|is_primary|docYq|ord|DG|href\n"
                "Rules: ordinal→DG=1 then ord asc; otherwise imp desc. "
                "Use click(ID)/input_text(ID,...).\n"
                f"{formatted}"
            )

            logger.info(
                "SentienceContext snapshot: %d elements URL=%s",
                len(snap.elements),
                url,
            )

            return SentienceContextState(url=url, snapshot=snap, prompt_block=prompt)

        except ImportError as e:
            logger.warning("Sentience SDK not available: %s", e)
            return None
        except Exception as e:
            logger.warning("Sentience snapshot skipped: %s", e)
            return None

    def _format_snapshot_for_llm(self, snapshot: Snapshot) -> str:
        """
        Format Sentience snapshot for LLM consumption.

        Creates an ultra-compact inventory of interactive elements optimized
        for minimal token usage. Uses 3-way selection: by importance,
        from dominant group, and by position.

        Args:
            snapshot: Sentience Snapshot object

        Returns:
            Formatted string with format: ID|role|text|imp|is_primary|docYq|ord|DG|href
        """
        # Filter to interactive elements only
        interactive_roles = {
            "button",
            "link",
            "textbox",
            "searchbox",
            "combobox",
            "checkbox",
            "radio",
            "slider",
            "tab",
            "menuitem",
            "option",
            "switch",
            "cell",
            "a",
            "input",
            "select",
            "textarea",
        }

        interactive_elements: list[Element] = []
        for el in snapshot.elements:
            role = (el.role or "").lower()
            if role in interactive_roles:
                interactive_elements.append(el)

        # Sort by importance (descending) for importance-based selection
        interactive_elements.sort(key=lambda el: el.importance or 0, reverse=True)

        # Get top N by importance (track by ID for deduplication)
        selected_ids: set[int] = set()
        selected_elements: list[Element] = []

        for el in interactive_elements[: self._selector.by_importance]:
            if el.id not in selected_ids:
                selected_ids.add(el.id)
                selected_elements.append(el)

        # Get top elements from dominant group (for ordinal tasks)
        # Prefer in_dominant_group field (uses fuzzy matching from gateway)
        dominant_group_elements = [
            el for el in interactive_elements if el.in_dominant_group is True
        ]

        # Fallback to exact group_key match if in_dominant_group not populated
        if not dominant_group_elements and snapshot.dominant_group_key:
            dominant_group_elements = [
                el for el in interactive_elements if el.group_key == snapshot.dominant_group_key
            ]

        # Sort by group_index for ordinal ordering
        dominant_group_elements.sort(key=lambda el: el.group_index or 999)

        for el in dominant_group_elements[: self._selector.from_dominant_group]:
            if el.id not in selected_ids:
                selected_ids.add(el.id)
                selected_elements.append(el)

        # Get top elements by position (lowest doc_y = top of page)
        def get_y_position(el: Element) -> float:
            if el.doc_y is not None:
                return el.doc_y
            if el.bbox is not None:
                return el.bbox.y
            return float("inf")

        elements_by_position = sorted(
            interactive_elements, key=lambda el: (get_y_position(el), -(el.importance or 0))
        )

        for el in elements_by_position[: self._selector.by_position]:
            if el.id not in selected_ids:
                selected_ids.add(el.id)
                selected_elements.append(el)

        # Compute local rank_in_group for dominant group elements
        rank_in_group_map: dict[int, int] = {}
        if True:  # Always compute rank_in_group
            # Sort dominant group elements by position for rank computation
            dg_elements_for_rank = [
                el for el in interactive_elements if el.in_dominant_group is True
            ]
            if not dg_elements_for_rank and snapshot.dominant_group_key:
                dg_elements_for_rank = [
                    el for el in interactive_elements if el.group_key == snapshot.dominant_group_key
                ]

            # Sort by (doc_y, bbox.y, bbox.x, -importance)
            def rank_sort_key(el: Element) -> tuple[float, float, float, float]:
                doc_y = el.doc_y if el.doc_y is not None else float("inf")
                bbox_y = el.bbox.y if el.bbox else float("inf")
                bbox_x = el.bbox.x if el.bbox else float("inf")
                neg_importance = -(el.importance or 0)
                return (doc_y, bbox_y, bbox_x, neg_importance)

            dg_elements_for_rank.sort(key=rank_sort_key)
            for rank, el in enumerate(dg_elements_for_rank):
                rank_in_group_map[el.id] = rank

        # Format lines
        lines: list[str] = []
        for el in selected_elements:
            # Get role (override to "link" if element has href)
            role = el.role or ""
            if el.href:
                role = "link"
            elif not role:
                # Generic fallback for interactive elements without explicit role
                role = "element"

            # Get name/text (truncate aggressively, normalize whitespace)
            name = el.text or ""
            # Remove newlines and normalize whitespace
            name = re.sub(r"\s+", " ", name.strip())
            if len(name) > 30:
                name = name[:27] + "..."

            # Extract fields
            importance = el.importance or 0
            doc_y = el.doc_y or 0

            # is_primary: from visual_cues.is_primary (boolean)
            is_primary = False
            if el.visual_cues:
                is_primary = el.visual_cues.is_primary or False
            is_primary_flag = "1" if is_primary else "0"

            # Pre-encode fields for compactness
            # docYq: bucketed doc_y (round to nearest 200 for smaller numbers)
            doc_yq = int(round(doc_y / 200)) if doc_y else 0

            # Determine if in dominant group
            in_dg = el.in_dominant_group
            if in_dg is None and snapshot.dominant_group_key:
                # Fallback for older gateway versions
                in_dg = el.group_key == snapshot.dominant_group_key

            # ord_val: rank_in_group if in dominant group
            if in_dg and el.id in rank_in_group_map:
                ord_val: str | int = rank_in_group_map[el.id]
            else:
                ord_val = "-"

            # DG: 1 if dominant group, else 0
            dg_flag = "1" if in_dg else "0"

            # href: meaningful path portion (e.g., "/dp/B0FC5SJNQX" for products)
            href = self._compress_href(el.href, snapshot.url)

            # Ultra-compact format: ID|role|text|imp|is_primary|docYq|ord|DG|href
            line = f"{el.id}|{role}|{name}|{importance}|{is_primary_flag}|{doc_yq}|{ord_val}|{dg_flag}|{href}"
            lines.append(line)

        logger.debug(
            "Formatted %d elements (top %d by importance + top %d from dominant group + top %d by position)",
            len(lines),
            self._selector.by_importance,
            self._selector.from_dominant_group,
            self._selector.by_position,
        )

        return "\n".join(lines)

    async def _wait_for_extension(
        self,
        backend: Any,
        *,
        timeout_ms: int = 5000,
        poll_interval_ms: int = 100,
    ) -> bool:
        """
        Wait for Sentience extension to be ready in the browser.

        Polls window.sentience until it's defined or timeout is reached.

        Args:
            backend: Browser backend with evaluate() method
            timeout_ms: Maximum time to wait in milliseconds
            poll_interval_ms: Interval between polls in milliseconds

        Returns:
            True if extension is ready, False if timeout
        """
        elapsed_ms = 0
        poll_interval_s = poll_interval_ms / 1000

        while elapsed_ms < timeout_ms:
            try:
                result = await backend.evaluate("typeof window.sentience !== 'undefined'")
                if result is True:
                    logger.debug("Sentience extension ready after %dms", elapsed_ms)
                    return True
            except Exception:
                # Extension not ready yet, continue polling
                pass

            await asyncio.sleep(poll_interval_s)
            elapsed_ms += poll_interval_ms

        logger.warning("Sentience extension not ready after %dms timeout", timeout_ms)
        return False

    def _compress_href(self, href: str | None, page_url: str | None = None) -> str:
        """
        Compress href into a meaningful short token for minimal tokens.

        Extracts the path portion that's useful for navigation decisions,
        stripping the base domain if it matches the current page.

        Args:
            href: Full URL or None
            page_url: Current page URL (to detect same-domain links)

        Returns:
            Compressed href with meaningful path info (e.g., "/dp/B0FC5SJNQX" or "/s?k=mouse")
        """
        if not href:
            return ""

        try:
            parsed = urlparse(href)
            page_parsed = urlparse(page_url) if page_url else None

            # Check if same domain as current page
            is_same_domain = (
                page_parsed
                and parsed.netloc
                and (parsed.netloc == page_parsed.netloc
                     or parsed.netloc.endswith("." + page_parsed.netloc.split(".")[-2] + "." + page_parsed.netloc.split(".")[-1])
                     if len(page_parsed.netloc.split(".")) >= 2 else False)
            )

            if parsed.netloc and not is_same_domain:
                # External link - show domain
                parts = parsed.netloc.split(".")
                if len(parts) >= 2:
                    return parts[-2][:15]
                return parsed.netloc[:15]

            # Same domain or relative link - extract meaningful path
            path = parsed.path or ""

            # For product pages, extract key identifiers
            # Amazon: /dp/XXXXX, /gp/product/XXXXX
            # Generic: /product/XXX, /item/XXX, /p/XXX
            import re
            product_patterns = [
                r"(/dp/[A-Z0-9]+)",  # Amazon product
                r"(/gp/product/[A-Z0-9]+)",  # Amazon alt
                r"(/product/[^/]+)",  # Generic product
                r"(/item/[^/]+)",  # Generic item
                r"(/p/[^/]+)",  # Short product
            ]
            for pattern in product_patterns:
                match = re.search(pattern, path, re.IGNORECASE)
                if match:
                    return match.group(1)[:30]

            # For search pages, extract search indicator
            if "/s?" in href or "/search?" in href or "?q=" in href or "?k=" in href:
                query = parsed.query or ""
                # Extract search term if present
                for param in ["k", "q", "query", "search"]:
                    match = re.search(rf"{param}=([^&]+)", query)
                    if match:
                        term = match.group(1)[:15]
                        return f"/s?{param}={term}"
                return "/search"

            # For cart/checkout pages
            if "/cart" in path.lower():
                return "/cart"
            if "/checkout" in path.lower():
                return "/checkout"

            # Fallback: use last meaningful path segment
            segments = [s for s in path.split("/") if s and len(s) > 1]
            if segments:
                # Return last 2 segments for context (max 30 chars)
                result = "/" + "/".join(segments[-2:])
                return result[:30]

            return path[:30] if path else ""

        except Exception:
            pass

        return ""


# ---------------------------------------------------------------------------
# Predicate-named counterparts (canonical moving forward).
# Keep Sentience* names for backward compatibility.
# ---------------------------------------------------------------------------

PredicateContext = SentienceContext
PredicateContextState = SentienceContextState
