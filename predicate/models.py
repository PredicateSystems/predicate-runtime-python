"""
Pydantic models for Sentience SDK - matches spec/snapshot.schema.json
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BBox(BaseModel):
    """Bounding box coordinates"""

    x: float
    y: float
    width: float
    height: float


class Viewport(BaseModel):
    """Viewport dimensions"""

    width: float
    height: float


class VisualCues(BaseModel):
    """Visual analysis cues"""

    is_primary: bool
    background_color_name: str | None = None
    fallback_background_color_name: str | None = None
    is_clickable: bool


class Element(BaseModel):
    """Element from snapshot"""

    id: int
    role: str
    text: str | None = None
    importance: int
    bbox: BBox
    visual_cues: VisualCues
    in_viewport: bool = True
    is_occluded: bool = False
    z_index: int = 0

    # ML reranking metadata (optional - can be absent or null)
    fused_rank_index: int | None = None  # 0-based, The rank index after ML reranking
    heuristic_index: int | None = None  # 0-based, Where it would have been without ML
    ml_probability: float | None = None  # Confidence score from ONNX model (0.0 - 1.0)
    ml_score: float | None = None  # Raw logit score (optional, for debugging)

    # Diff status for frontend Diff Overlay feature
    diff_status: Literal["ADDED", "REMOVED", "MODIFIED", "MOVED"] | None = None

    # Phase 1: Ordinal support fields for position-based selection
    center_x: float | None = None  # X coordinate of element center (viewport coords)
    center_y: float | None = None  # Y coordinate of element center (viewport coords)
    doc_y: float | None = None  # Y coordinate in document (center_y + scroll_y)
    group_key: str | None = None  # Geometric bucket key for ordinal grouping
    group_index: int | None = None  # Position within group (0-indexed, sorted by doc_y)

    # Hyperlink URL (for link elements)
    href: str | None = None

    # Nearby static text (best-effort, usually only for top-ranked elements)
    nearby_text: str | None = None

    # ===== v1 state-aware assertion fields (optional) =====
    # Best-effort accessible name/label for controls (distinct from visible text)
    name: str | None = None
    # Current value for inputs/textarea/select (PII-aware: may be omitted/redacted)
    value: str | None = None
    # Input type (e.g., "text", "email", "password")
    input_type: str | None = None
    # Whether value was redacted for privacy
    value_redacted: bool | None = None
    # Normalized boolean states (best-effort)
    checked: bool | None = None
    disabled: bool | None = None
    expanded: bool | None = None
    # Raw ARIA state strings (tri-state / debugging)
    aria_checked: str | None = None
    aria_disabled: str | None = None
    aria_expanded: str | None = None

    # Phase 3.2: Pre-computed dominant group membership (uses fuzzy matching)
    # This field is computed by the gateway so downstream consumers don't need to
    # implement fuzzy matching logic themselves.
    in_dominant_group: bool | None = None

    # Layout-derived metadata (internal-only in v0, not exposed in API responses)
    # Per ChatGPT feedback: explicitly optional to prevent users assuming layout is always present
    # Note: This field is marked with skip_serializing_if in Rust, so it won't appear in API responses
    layout: LayoutHints | None = None


class GridPosition(BaseModel):
    """Grid position within a detected grid/list"""

    row_index: int  # 0-based row index
    col_index: int  # 0-based column index
    cluster_id: int  # ID of the row cluster (for distinguishing separate grids)


class LayoutHints(BaseModel):
    """Layout-derived metadata for an element (internal-only in v0)"""

    # Grid ID (maps to GridInfo.grid_id) - distinguishes multiple grids on same page
    # Per feedback: Add grid_id to distinguish main feed + sidebar lists + nav links
    grid_id: int | None = None
    # Grid position within the grid (row_index, col_index)
    grid_pos: GridPosition | None = None
    # Inferred parent index in elements array
    parent_index: int | None = None
    # Indices of child elements (optional to avoid payload bloat - container elements can have hundreds)
    # Per feedback: Make optional/capped to prevent serializing large arrays
    children_indices: list[int] | None = None
    # Confidence score for grid position assignment (0.0-1.0)
    grid_confidence: float = 0.0
    # Confidence score for parent-child containment (0.0-1.0)
    parent_confidence: float = 0.0
    # Optional: Page region (header/nav/main/aside/footer) - killer signal for ordinality + dominant group
    # Per feedback: Optional but very useful for region detection
    region: Literal["header", "nav", "main", "aside", "footer"] | None = None
    region_confidence: float = 0.0  # Confidence score for region assignment (0.0-1.0)


class GridInfo(BaseModel):
    """Grid bounding box and metadata for a detected grid"""

    grid_id: int  # The grid ID (matches grid_id in LayoutHints)
    bbox: BBox  # Bounding box: x, y, width, height (document coordinates)
    row_count: int  # Number of rows in the grid
    col_count: int  # Number of columns in the grid
    item_count: int  # Total number of items in the grid
    confidence: float = 1.0  # Confidence score (currently 1.0)
    label: str | None = (
        None  # Optional inferred label (e.g., "product_grid", "search_results", "navigation")
    )
    is_dominant: bool = False  # Whether this grid is the dominant group (main content area)

    # Z-index and modal detection fields (from gateway/sentience-core)
    z_index: int = 0  # Z-index of this grid (max among elements in this grid)
    z_index_max: int = 0  # Global max z-index across ALL grids (for comparison)
    blocks_interaction: bool = False  # Whether this grid blocks interaction with content behind it
    viewport_coverage: float = 0.0  # Ratio of grid area to viewport area (0.0-1.0)


class MlRerankTags(BaseModel):
    """ML rerank tag configuration used for candidate text"""

    repeated: bool
    sponsored_ish: bool
    non_sponsored: bool
    pos: bool
    occ: bool
    vocc: bool
    short: bool
    action_ish: bool
    nav_ish: bool


class MlRerankInfo(BaseModel):
    """ML rerank metadata for a snapshot response"""

    enabled: bool
    applied: bool
    reason: str | None = None
    candidate_count: int | None = None
    top_probability: float | None = None
    min_confidence: float | None = None
    is_high_confidence: bool | None = None
    tags: MlRerankTags | None = None
    error: str | None = None


class Snapshot(BaseModel):
    """Snapshot response from extension"""

    status: Literal["success", "error"]
    timestamp: str | None = None
    url: str
    viewport: Viewport | None = None
    elements: list[Element]
    screenshot: str | None = None
    screenshot_format: Literal["png", "jpeg"] | None = None
    error: str | None = None
    requires_license: bool | None = None
    # Phase 2: Dominant group key for ordinal selection
    dominant_group_key: str | None = None  # The most common group_key (main content group)
    # Phase 2: Runtime stability/debug info (confidence/reasons/metrics)
    diagnostics: SnapshotDiagnostics | None = None
    # Modal detection fields (from gateway)
    modal_detected: bool | None = None  # True if a modal/overlay grid was detected
    modal_grids: list[GridInfo] | None = None  # Array of GridInfo for detected modal grids
    # ML rerank metadata (optional)
    ml_rerank: MlRerankInfo | None = None

    def save(self, filepath: str) -> None:
        """Save snapshot as JSON file"""
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    def get_grid_bounds(self, grid_id: int | None = None) -> list[GridInfo]:
        """
        Get grid coordinates (bounding boxes) for detected grids.

        Groups elements by grid_id and computes the overall bounding box,
        row/column counts, and item count for each grid.

        Args:
            grid_id: Optional grid ID to filter by. If None, returns all grids.

        Returns:
            List of GridInfo objects, one per detected grid, sorted by grid_id.
        """
        from collections import defaultdict

        # Group elements by grid_id
        grid_elements: dict[int, list[Element]] = defaultdict(list)

        for elem in self.elements:
            if elem.layout and elem.layout.grid_id is not None:
                grid_elements[elem.layout.grid_id].append(elem)

        # Filter by grid_id if specified
        if grid_id is not None:
            if grid_id not in grid_elements:
                return []
            grid_elements = {grid_id: grid_elements[grid_id]}

        grid_infos: list[GridInfo] = []

        # First pass: compute all grid infos and count dominant group elements
        grid_dominant_counts: dict[int, tuple[int, int]] = {}
        for gid, elements_in_grid in sorted(grid_elements.items()):
            if not elements_in_grid:
                continue

            # Count dominant group elements in this grid
            dominant_count = sum(1 for elem in elements_in_grid if elem.in_dominant_group is True)
            grid_dominant_counts[gid] = (dominant_count, len(elements_in_grid))

            # Compute bounding box
            min_x = min(elem.bbox.x for elem in elements_in_grid)
            min_y = min(elem.bbox.y for elem in elements_in_grid)
            max_x = max(elem.bbox.x + elem.bbox.width for elem in elements_in_grid)
            max_y = max(elem.bbox.y + elem.bbox.height for elem in elements_in_grid)

            # Count rows and columns
            row_indices = set()
            col_indices = set()

            for elem in elements_in_grid:
                if elem.layout and elem.layout.grid_pos:
                    row_indices.add(elem.layout.grid_pos.row_index)
                    col_indices.add(elem.layout.grid_pos.col_index)

            # Infer grid label from element patterns (best-effort heuristic)
            # Keep the heuristic implementation in one place.
            label = SnapshotDiagnostics._infer_grid_label(elements_in_grid)

            grid_infos.append(
                GridInfo(
                    grid_id=gid,
                    bbox=BBox(
                        x=min_x,
                        y=min_y,
                        width=max_x - min_x,
                        height=max_y - min_y,
                    ),
                    row_count=len(row_indices) if row_indices else 0,
                    col_count=len(col_indices) if col_indices else 0,
                    item_count=len(elements_in_grid),
                    confidence=1.0,
                    label=label,
                    is_dominant=False,  # Will be set below
                )
            )

        # Second pass: identify dominant grid
        # The grid with the highest count (or highest percentage >= 50%) of dominant group elements
        if grid_dominant_counts:
            # Find grid with highest absolute count
            max_dominant_count = max(count for count, _ in grid_dominant_counts.values())
            if max_dominant_count > 0:
                # Find grid(s) with highest count
                dominant_grids = [
                    gid
                    for gid, (count, _total) in grid_dominant_counts.items()
                    if count == max_dominant_count
                ]
                # If multiple grids tie, prefer the one with highest percentage
                if len(dominant_grids) > 1:
                    dominant_grids.sort(
                        key=lambda gid: (
                            grid_dominant_counts[gid][0] / grid_dominant_counts[gid][1]
                            if grid_dominant_counts[gid][1] > 0
                            else 0
                        ),
                        reverse=True,
                    )

                # Mark the dominant grid
                dominant_gid = dominant_grids[0]
                # Only mark as dominant if it has >= 50% dominant group elements or >= 3 elements
                dominant_count, total_count = grid_dominant_counts[dominant_gid]
                if dominant_count >= 3 or (total_count > 0 and dominant_count / total_count >= 0.5):
                    for grid_info in grid_infos:
                        if grid_info.grid_id == dominant_gid:
                            grid_info.is_dominant = True
                            break

        return grid_infos


class SnapshotDiagnosticsMetrics(BaseModel):
    ready_state: str | None = None
    quiet_ms: float | None = None
    node_count: int | None = None
    interactive_count: int | None = None
    raw_elements_count: int | None = None


class CaptchaEvidence(BaseModel):
    text_hits: list[str] = Field(default_factory=list)
    selector_hits: list[str] = Field(default_factory=list)
    iframe_src_hits: list[str] = Field(default_factory=list)
    url_hits: list[str] = Field(default_factory=list)


class CaptchaDiagnostics(BaseModel):
    """Detection-only CAPTCHA signal (no solving/bypass)."""

    detected: bool = False
    provider_hint: str | None = None
    confidence: float = 0.0
    evidence: CaptchaEvidence = Field(default_factory=CaptchaEvidence)


class SnapshotDiagnostics(BaseModel):
    """Runtime stability/debug information (reserved for diagnostics, not ML metadata)."""

    confidence: float | None = None
    reasons: list[str] = Field(default_factory=list)
    metrics: SnapshotDiagnosticsMetrics | None = None
    captcha: CaptchaDiagnostics | None = None
    # P1-01: forward-compatible vision recommendation signal (optional)
    requires_vision: bool | None = None
    requires_vision_reason: str | None = None

    def get_grid_bounds(self, grid_id: int | None = None) -> list[GridInfo]:
        """
        Get grid coordinates (bounding boxes) for detected grids.

        Groups elements by grid_id and computes the overall bounding box,
        row/column counts, and item count for each grid.

        Args:
            grid_id: Optional grid ID to filter by. If None, returns all grids.

        Returns:
            List of GridInfo objects, one per detected grid, sorted by grid_id.
            Each GridInfo contains:
            - grid_id: The grid identifier
            - bbox: Bounding box (x, y, width, height) in document coordinates
            - row_count: Number of rows in the grid
            - col_count: Number of columns in the grid
            - item_count: Total number of items in the grid
            - confidence: Confidence score (currently 1.0)
            - label: Optional inferred label (e.g., "product_grid", "search_results", "navigation")
              Note: Label inference is best-effort and may not always be accurate

        Example:
            >>> snapshot = browser.snapshot()
            >>> # Get all grids
            >>> all_grids = snapshot.get_grid_bounds()
            >>> # Get specific grid
            >>> main_grid = snapshot.get_grid_bounds(grid_id=0)
            >>>             if main_grid:
            ...     print(f"Grid 0: {main_grid[0].item_count} items at ({main_grid[0].bbox.x}, {main_grid[0].bbox.y})")
        """
        from collections import defaultdict

        # Group elements by grid_id
        grid_elements: dict[int, list[Element]] = defaultdict(list)

        for elem in self.elements:
            if elem.layout and elem.layout.grid_id is not None:
                grid_elements[elem.layout.grid_id].append(elem)

        # Filter by grid_id if specified
        if grid_id is not None:
            if grid_id not in grid_elements:
                return []
            grid_elements = {grid_id: grid_elements[grid_id]}

        grid_infos = []

        # First pass: compute all grid infos and count dominant group elements
        grid_dominant_counts = {}
        for gid, elements_in_grid in sorted(grid_elements.items()):
            if not elements_in_grid:
                continue

            # Count dominant group elements in this grid
            dominant_count = sum(1 for elem in elements_in_grid if elem.in_dominant_group is True)
            grid_dominant_counts[gid] = (dominant_count, len(elements_in_grid))

            # Compute bounding box
            min_x = min(elem.bbox.x for elem in elements_in_grid)
            min_y = min(elem.bbox.y for elem in elements_in_grid)
            max_x = max(elem.bbox.x + elem.bbox.width for elem in elements_in_grid)
            max_y = max(elem.bbox.y + elem.bbox.height for elem in elements_in_grid)

            # Count rows and columns
            row_indices = set()
            col_indices = set()

            for elem in elements_in_grid:
                if elem.layout and elem.layout.grid_pos:
                    row_indices.add(elem.layout.grid_pos.row_index)
                    col_indices.add(elem.layout.grid_pos.col_index)

            # Infer grid label from element patterns (best-effort heuristic)
            label = Snapshot._infer_grid_label(elements_in_grid)

            grid_infos.append(
                GridInfo(
                    grid_id=gid,
                    bbox=BBox(
                        x=min_x,
                        y=min_y,
                        width=max_x - min_x,
                        height=max_y - min_y,
                    ),
                    row_count=len(row_indices) if row_indices else 0,
                    col_count=len(col_indices) if col_indices else 0,
                    item_count=len(elements_in_grid),
                    confidence=1.0,
                    label=label,
                    is_dominant=False,  # Will be set below
                )
            )

        # Second pass: identify dominant grid
        # The grid with the highest count (or highest percentage >= 50%) of dominant group elements
        if grid_dominant_counts:
            # Find grid with highest absolute count
            max_dominant_count = max(count for count, _ in grid_dominant_counts.values())
            if max_dominant_count > 0:
                # Find grid(s) with highest count
                dominant_grids = [
                    gid
                    for gid, (count, total) in grid_dominant_counts.items()
                    if count == max_dominant_count
                ]
                # If multiple grids tie, prefer the one with highest percentage
                if len(dominant_grids) > 1:
                    dominant_grids.sort(
                        key=lambda gid: (
                            grid_dominant_counts[gid][0] / grid_dominant_counts[gid][1]
                            if grid_dominant_counts[gid][1] > 0
                            else 0
                        ),
                        reverse=True,
                    )
                # Mark the dominant grid
                dominant_gid = dominant_grids[0]
                # Only mark as dominant if it has >= 50% dominant group elements or >= 3 elements
                dominant_count, total_count = grid_dominant_counts[dominant_gid]
                if dominant_count >= 3 or (total_count > 0 and dominant_count / total_count >= 0.5):
                    for grid_info in grid_infos:
                        if grid_info.grid_id == dominant_gid:
                            grid_info.is_dominant = True
                            break

        return grid_infos

    @staticmethod
    def _infer_grid_label(elements: list[Element]) -> str | None:
        """
        Infer grid label from element patterns using text fingerprinting (best-effort heuristic).

        Uses patterns similar to dominant_group.rs content filtering logic, inverted to detect
        semantic grid types. Analyzes first 5 items as a "bag of features".

        Returns None if label cannot be reliably determined.
        This is a simple heuristic and may not always be accurate.
        """
        import re

        if not elements:
            return None

        # Sample first 5 items for fingerprinting (as suggested in feedback)
        sample_elements = elements[:5]
        element_texts = [(elem.text or "").strip() for elem in sample_elements if elem.text]

        if not element_texts:
            return None

        # Collect text patterns
        all_text = " ".join(text.lower() for text in element_texts)
        hrefs = [elem.href or "" for elem in sample_elements if elem.href]

        # =========================================================================
        # 1. PRODUCT GRID: Currency symbols, action verbs, ratings
        # =========================================================================
        # Currency patterns: $, €, £, or price patterns like "19.99", "$50", "€30"
        currency_pattern = re.search(r"[\$€£¥]\s*\d+|\d+\.\d{2}", all_text)
        product_action_verbs = [
            "add to cart",
            "buy now",
            "shop now",
            "purchase",
            "out of stock",
            "in stock",
        ]
        has_product_actions = any(verb in all_text for verb in product_action_verbs)

        # Ratings pattern: "4.5 stars", "(120 reviews)", "4.5/5"
        rating_pattern = re.search(r"\d+\.?\d*\s*(stars?|reviews?|/5|/10)", all_text, re.IGNORECASE)

        # Product URL patterns
        product_url_patterns = ["/product/", "/item/", "/dp/", "/p/", "/products/"]
        has_product_urls = any(
            pattern in href.lower() for href in hrefs for pattern in product_url_patterns
        )

        if (currency_pattern or has_product_actions or rating_pattern) and (
            has_product_urls
            or len(
                [
                    t
                    for t in element_texts
                    if currency_pattern and currency_pattern.group() in t.lower()
                ]
            )
            >= 2
        ):
            return "product_grid"

        # =========================================================================
        # 2. ARTICLE/NEWS FEED: Timestamps, bylines, reading time
        # =========================================================================
        # Timestamp patterns (reusing logic from dominant_group.rs)
        # "2 hours ago", "3 days ago", "5 minutes ago", "1 second ago", "2 ago"
        timestamp_patterns = [
            r"\d+\s+(hour|day|minute|second)s?\s+ago",
            r"\d+\s+ago",  # Short form: "2 ago"
            r"\d{1,2}\s+(hour|day|minute|second)\s+ago",  # Singular
        ]
        has_timestamps = any(
            re.search(pattern, all_text, re.IGNORECASE) for pattern in timestamp_patterns
        )

        # Date patterns: "Aug 21, 2024", "2024-01-13", "Jan 15"
        date_patterns = [
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}",
            r"\d{4}-\d{2}-\d{2}",
            r"\d{1,2}/\d{1,2}/\d{4}",
        ]
        has_dates = any(re.search(pattern, all_text, re.IGNORECASE) for pattern in date_patterns)

        # Bylines: "By [Name]", "Author:", "Written by"
        byline_patterns = ["by ", "author:", "written by", "posted by"]
        has_bylines = any(pattern in all_text for pattern in byline_patterns)

        # Reading time: "5 min read", "10 min", "read more"
        reading_time_pattern = re.search(r"\d+\s*(min|minute)s?\s*(read)?", all_text, re.IGNORECASE)

        if has_timestamps or (has_dates and has_bylines) or reading_time_pattern:
            return "article_feed"

        # =========================================================================
        # 3. SEARCH RESULTS: Snippets, metadata, ellipses
        # =========================================================================
        search_keywords = ["result", "search", "found", "showing", "results 1-", "sponsored"]
        has_search_metadata = any(keyword in all_text for keyword in search_keywords)

        # Snippet indicators: ellipses, "match found", truncated text
        has_ellipses = "..." in all_text or any(
            len(text) > 100 and "..." in text for text in element_texts
        )

        # Check if many elements are links (typical for search results)
        link_count = sum(1 for elem in sample_elements if elem.role == "link" or elem.href)
        is_mostly_links = link_count >= len(sample_elements) * 0.7  # 70%+ are links

        if (has_search_metadata or has_ellipses) and is_mostly_links:
            return "search_results"

        # =========================================================================
        # 4. NAVIGATION: Short length, homogeneity, common nav terms
        # =========================================================================
        # Calculate average text length and variance
        text_lengths = [len(text) for text in element_texts]
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            # Low variance = homogeneous (typical of navigation)
            variance = (
                sum((l - avg_length) ** 2 for l in text_lengths) / len(text_lengths)
                if len(text_lengths) > 1
                else 0
            )

            nav_keywords = [
                "home",
                "about",
                "contact",
                "menu",
                "login",
                "sign in",
                "profile",
                "settings",
            ]
            has_nav_keywords = any(keyword in all_text for keyword in nav_keywords)

            # Navigation: short average length (< 15 chars) AND low variance OR nav keywords
            if avg_length < 15 and (variance < 20 or has_nav_keywords):
                # Also check if all are links
                if all(elem.role == "link" or elem.href for elem in sample_elements):
                    return "navigation"

        # =========================================================================
        # 5. BUTTON GRID: All buttons
        # =========================================================================
        if all(elem.role == "button" for elem in sample_elements):
            return "button_grid"

        # =========================================================================
        # 6. LINK LIST: Mostly links but not navigation
        # =========================================================================
        link_count = sum(1 for elem in sample_elements if elem.role == "link" or elem.href)
        if link_count >= len(sample_elements) * 0.8:  # 80%+ are links
            return "link_list"

        # Unknown/unclear
        return None


class ActionResult(BaseModel):
    """Result of an action (click, type, press)"""

    success: bool
    duration_ms: int
    outcome: Literal["navigated", "dom_updated", "no_change", "error"] | None = None
    url_changed: bool | None = None
    snapshot_after: Snapshot | None = None
    error: dict | None = None
    # Optional action metadata (e.g., human-like cursor movement path)
    cursor: dict[str, Any] | None = None


class TabInfo(BaseModel):
    """Metadata about an open browser tab/page."""

    tab_id: str
    url: str | None = None
    title: str | None = None
    is_active: bool = False


class TabListResult(BaseModel):
    """Result of listing tabs."""

    ok: bool
    tabs: list[TabInfo] = Field(default_factory=list)
    error: str | None = None


class TabOperationResult(BaseModel):
    """Result of tab operations (open/switch/close)."""

    ok: bool
    tab: TabInfo | None = None
    error: str | None = None


class StepHookContext(BaseModel):
    """Context passed to lifecycle hooks."""

    step_id: str
    step_index: int
    goal: str
    attempt: int = 0
    url: str | None = None
    success: bool | None = None
    outcome: str | None = None
    error: str | None = None


class EvaluateJsRequest(BaseModel):
    """Request for evaluate_js helper."""

    code: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="JavaScript source code to evaluate in the page context.",
    )
    max_output_chars: int = Field(
        4000,
        ge=1,
        le=20000,
        description="Maximum number of characters to return in the text field.",
    )
    truncate: bool = Field(
        True,
        description="Whether to truncate text output when it exceeds max_output_chars.",
    )


class EvaluateJsResult(BaseModel):
    """Result of evaluate_js helper."""

    ok: bool = Field(..., description="Whether evaluation succeeded.")
    value: Any | None = Field(None, description="Raw value returned by the page evaluation.")
    text: str | None = Field(None, description="Best-effort string representation of the value.")
    truncated: bool = Field(False, description="True if text output was truncated.")
    error: str | None = Field(None, description="Error string when ok=False.")


class WaitResult(BaseModel):
    """Result of wait_for operation"""

    found: bool
    element: Element | None = None
    duration_ms: int
    timeout: bool


# ========== Agent Layer Models ==========


class ScreenshotConfig(BaseModel):
    """Screenshot format configuration"""

    format: Literal["png", "jpeg"] = "png"
    quality: int | None = Field(None, ge=1, le=100)  # Only for JPEG (1-100)


class SnapshotFilter(BaseModel):
    """Filter options for snapshot elements"""

    min_area: int | None = Field(None, ge=0)
    allowed_roles: list[str] | None = None
    min_z_index: int | None = None


class SnapshotOptions(BaseModel):
    """
    Configuration for snapshot calls.
    Matches TypeScript SnapshotOptions interface from sdk-ts/src/snapshot.ts

    For browser-use integration (where you don't have a SentienceBrowser),
    you can pass sentience_api_key directly in options:

        from predicate.models import SnapshotOptions
        options = SnapshotOptions(
            sentience_api_key="sk_pro_xxxxx",
            use_api=True,
            goal="Find the login button"
        )
    """

    screenshot: bool | ScreenshotConfig = False  # Union type: boolean or config
    limit: int = Field(50, ge=1, le=500)
    filter: SnapshotFilter | None = None
    use_api: bool | None = None  # Force API vs extension
    gateway_timeout_s: float | None = None  # Gateway snapshot timeout (seconds)
    save_trace: bool = False  # Save raw_elements to JSON for benchmarking/training
    trace_path: str | None = None  # Path to save trace (default: "trace_{timestamp}.json")
    goal: str | None = None  # Optional goal/task description for the snapshot
    show_overlay: bool = False  # Show visual overlay highlighting elements in browser
    show_grid: bool = False  # Show visual overlay highlighting detected grids
    grid_id: int | None = (
        None  # Optional grid ID to show specific grid (only used if show_grid=True)
    )

    # API credentials (for browser-use integration without SentienceBrowser)
    # Keep both names during migration; Predicate name is canonical.
    predicate_api_key: str | None = None
    sentience_api_key: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _sync_api_key_aliases(self) -> "SnapshotOptions":
        """
        Keep predicate_api_key and sentience_api_key in sync during migration.
        Predicate naming wins when both are set.
        """
        if self.predicate_api_key:
            self.sentience_api_key = self.predicate_api_key
        elif self.sentience_api_key:
            self.predicate_api_key = self.sentience_api_key
        return self


class AgentActionResult(BaseModel):
    """Result of a single agent action (from agent.act())"""

    success: bool
    action: Literal["click", "type", "press", "finish", "error"]
    goal: str
    duration_ms: int
    attempt: int

    # Optional fields based on action type
    element_id: int | None = None
    text: str | None = None
    key: str | None = None
    outcome: Literal["navigated", "dom_updated", "no_change", "error"] | None = None
    url_changed: bool | None = None
    error: str | None = None
    message: str | None = None  # For FINISH action
    # Optional: action metadata (e.g., human-like cursor movement path)
    cursor: dict[str, Any] | None = None

    def __getitem__(self, key):
        """
        Support dict-style access for backward compatibility.
        This allows existing code using result["success"] to continue working.
        """
        import warnings

        warnings.warn(
            f"Dict-style access result['{key}'] is deprecated. Use result.{key} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key)


class ActionTokenUsage(BaseModel):
    """Token usage for a single action"""

    goal: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


class LLMUsage(BaseModel):
    """Token usage for a single LLM call"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMStepData(BaseModel):
    """
    LLM interaction data for a single step in agent traces.

    This structure is used in step_end trace events to capture LLM response
    details for debugging and analysis in Sentience Studio.
    """

    response_text: str | None = Field(
        None,
        description="The LLM's response text for this step",
    )
    response_hash: str | None = Field(
        None,
        description="SHA256 hash of response_text for deduplication/indexing",
    )
    usage: LLMUsage | None = Field(
        None,
        description="Token usage statistics for this LLM call",
    )
    model: str | None = Field(
        None,
        description="Model identifier used for this call (e.g., 'gpt-4o', 'claude-3-5-sonnet')",
    )

    def to_trace_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format expected by TraceEventBuilder.

        Returns:
            Dict with response_text, response_hash, and usage fields
        """
        result: dict[str, Any] = {}
        if self.response_text is not None:
            result["response_text"] = self.response_text
        if self.response_hash is not None:
            result["response_hash"] = self.response_hash
        if self.usage is not None:
            result["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        if self.model is not None:
            result["model"] = self.model
        return result


class TokenStats(BaseModel):
    """Token usage statistics for an agent session"""

    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    by_action: list[ActionTokenUsage]


class ActionHistory(BaseModel):
    """Single history entry from agent execution"""

    goal: str
    action: str  # The raw action string from LLM
    result: dict  # Will be AgentActionResult but stored as dict for flexibility
    success: bool
    attempt: int
    duration_ms: int


class ProxyConfig(BaseModel):
    """
    Proxy configuration for browser networking.

    Supports HTTP, HTTPS, and SOCKS5 proxies with optional authentication.
    """

    server: str = Field(
        ...,
        description="Proxy server URL including scheme and port (e.g., 'http://proxy.example.com:8080')",
    )
    username: str | None = Field(
        None,
        description="Username for proxy authentication (optional)",
    )
    password: str | None = Field(
        None,
        description="Password for proxy authentication (optional)",
    )

    def to_playwright_dict(self) -> dict:
        """
        Convert to Playwright proxy configuration format.

        Returns:
            Dict compatible with Playwright's proxy parameter
        """
        config = {"server": self.server}
        if self.username and self.password:
            config["username"] = self.username
            config["password"] = self.password
        return config


# ========== Storage State Models (Auth Injection) ==========


class Cookie(BaseModel):
    """
    Cookie definition for storage state injection.

    Matches Playwright's cookie format for storage_state.
    """

    name: str = Field(..., description="Cookie name")
    value: str = Field(..., description="Cookie value")
    domain: str = Field(..., description="Cookie domain (e.g., '.example.com')")
    path: str = Field(default="/", description="Cookie path")
    expires: float | None = Field(None, description="Expiration timestamp (Unix epoch)")
    httpOnly: bool = Field(default=False, description="HTTP-only flag")
    secure: bool = Field(default=False, description="Secure (HTTPS-only) flag")
    sameSite: Literal["Strict", "Lax", "None"] = Field(
        default="Lax", description="SameSite attribute"
    )


class LocalStorageItem(BaseModel):
    """
    LocalStorage item for a specific origin.

    Playwright stores localStorage as an array of {name, value} objects.
    """

    name: str = Field(..., description="LocalStorage key")
    value: str = Field(..., description="LocalStorage value")


class OriginStorage(BaseModel):
    """
    Storage state for a specific origin (localStorage).

    Represents localStorage data for a single domain.
    """

    origin: str = Field(..., description="Origin URL (e.g., 'https://example.com')")
    localStorage: list[LocalStorageItem] = Field(
        default_factory=list, description="LocalStorage items for this origin"
    )


class StorageState(BaseModel):
    """
    Complete browser storage state (cookies + localStorage).

    This is the format used by Playwright's storage_state() method.
    Can be saved to/loaded from JSON files for session injection.
    """

    cookies: list[Cookie] = Field(
        default_factory=list, description="Cookies to inject (global scope)"
    )
    origins: list[OriginStorage] = Field(
        default_factory=list, description="LocalStorage data per origin"
    )

    @classmethod
    def from_dict(cls, data: dict) -> StorageState:
        """
        Create StorageState from dictionary (e.g., loaded from JSON).

        Args:
            data: Dictionary with 'cookies' and/or 'origins' keys

        Returns:
            StorageState instance
        """
        cookies = [
            Cookie(**cookie) if isinstance(cookie, dict) else cookie
            for cookie in data.get("cookies", [])
        ]
        origins = []
        for origin_data in data.get("origins", []):
            if isinstance(origin_data, dict):
                # Handle localStorage as array of {name, value} or as dict
                localStorage_data = origin_data.get("localStorage", [])
                if isinstance(localStorage_data, dict):
                    # Convert dict to list of LocalStorageItem
                    localStorage_items = [
                        LocalStorageItem(name=k, value=v) for k, v in localStorage_data.items()
                    ]
                else:
                    # Already a list
                    localStorage_items = [
                        LocalStorageItem(**item) if isinstance(item, dict) else item
                        for item in localStorage_data
                    ]
                origins.append(
                    OriginStorage(
                        origin=origin_data.get("origin", ""),
                        localStorage=localStorage_items,
                    )
                )
            else:
                origins.append(origin_data)
        return cls(cookies=cookies, origins=origins)

    def to_playwright_dict(self) -> dict:
        """
        Convert to Playwright-compatible dictionary format.

        Returns:
            Dictionary compatible with Playwright's storage_state parameter
        """
        return {
            "cookies": [cookie.model_dump() for cookie in self.cookies],
            "origins": [
                {
                    "origin": origin.origin,
                    "localStorage": [item.model_dump() for item in origin.localStorage],
                }
                for origin in self.origins
            ],
        }


# ========== Text Search Models (findTextRect) ==========


class TextRect(BaseModel):
    """
    Rectangle coordinates for text occurrence.
    Includes both absolute (page) and viewport-relative coordinates.
    """

    x: float = Field(..., description="Absolute X coordinate (page coordinate with scroll offset)")
    y: float = Field(..., description="Absolute Y coordinate (page coordinate with scroll offset)")
    width: float = Field(..., description="Rectangle width in pixels")
    height: float = Field(..., description="Rectangle height in pixels")
    left: float = Field(..., description="Absolute left position (same as x)")
    top: float = Field(..., description="Absolute top position (same as y)")
    right: float = Field(..., description="Absolute right position (x + width)")
    bottom: float = Field(..., description="Absolute bottom position (y + height)")


class ViewportRect(BaseModel):
    """Viewport-relative rectangle coordinates (without scroll offset)"""

    x: float = Field(..., description="Viewport-relative X coordinate")
    y: float = Field(..., description="Viewport-relative Y coordinate")
    width: float = Field(..., description="Rectangle width in pixels")
    height: float = Field(..., description="Rectangle height in pixels")


class TextContext(BaseModel):
    """Context text surrounding a match"""

    before: str = Field(..., description="Text before the match (up to 20 chars)")
    after: str = Field(..., description="Text after the match (up to 20 chars)")


class TextMatch(BaseModel):
    """A single text match with its rectangle and context"""

    text: str = Field(..., description="The matched text")
    rect: TextRect = Field(..., description="Absolute rectangle coordinates (with scroll offset)")
    viewport_rect: ViewportRect = Field(
        ..., description="Viewport-relative rectangle (without scroll offset)"
    )
    context: TextContext = Field(..., description="Surrounding text context")
    in_viewport: bool = Field(..., description="Whether the match is currently visible in viewport")


class TextRectSearchResult(BaseModel):
    """
    Result of findTextRect operation.
    Returns all occurrences of text on the page with their exact pixel coordinates.
    """

    status: Literal["success", "error"]
    query: str | None = Field(None, description="The search text that was queried")
    case_sensitive: bool | None = Field(None, description="Whether search was case-sensitive")
    whole_word: bool | None = Field(None, description="Whether whole-word matching was used")
    matches: int | None = Field(None, description="Number of matches found")
    results: list[TextMatch] | None = Field(
        None, description="List of text matches with coordinates"
    )
    viewport: Viewport | None = Field(None, description="Current viewport dimensions")
    error: str | None = Field(None, description="Error message if status is 'error'")


class ReadResult(BaseModel):
    """Result of read() or read_async() operation"""

    status: Literal["success", "error"]
    url: str
    format: Literal["raw", "text", "markdown"]
    content: str
    length: int
    error: str | None = None


class ExtractResult(BaseModel):
    """Result of extract() or extract_async() operation"""

    ok: bool
    data: Any | None = None
    raw: str | None = None
    error: str | None = None
    llm_response: Any | None = None


class TraceStats(BaseModel):
    """Execution statistics for trace completion"""

    total_steps: int
    total_events: int
    duration_ms: int | None = None
    final_status: Literal["success", "failure", "partial", "unknown"]
    started_at: str | None = None
    ended_at: str | None = None


class StepExecutionResult(BaseModel):
    """Result of executing a single step in ConversationalAgent"""

    success: bool
    action: str
    data: dict[str, Any]  # Flexible data field for step-specific results
    error: str | None = None


class ExtractionResult(BaseModel):
    """Result of extracting information from a page"""

    found: bool
    data: dict[str, Any]  # Extracted data fields
    summary: str  # Brief description of what was found


@dataclass
class ScreenshotMetadata:
    """
    Metadata for a stored screenshot.

    Used by CloudTraceSink to track screenshots before upload.
    All fields are required for type safety.
    """

    sequence: int
    format: Literal["png", "jpeg"]
    size_bytes: int
    step_id: str | None
    filepath: str
