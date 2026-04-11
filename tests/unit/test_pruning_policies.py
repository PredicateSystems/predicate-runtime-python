"""
Unit tests for deterministic pruning policies and serializer output.
"""

from predicate.models import BBox, Element, Snapshot, VisualCues
from predicate.pruning import PruningTaskCategory
from predicate.pruning.pruner import prune_snapshot_for_task
from predicate.pruning.serializer import serialize_pruned_snapshot


def make_element(
    *,
    id: int,
    role: str,
    text: str,
    importance: int,
    doc_y: float = 0.0,
    in_dominant_group: bool | None = None,
    href: str | None = None,
    nearby_text: str | None = None,
) -> Element:
    return Element(
        id=id,
        role=role,
        text=text,
        importance=importance,
        bbox=BBox(x=0, y=doc_y, width=100, height=24),
        visual_cues=VisualCues(is_primary=False, is_clickable=role in {"button", "link", "textbox", "searchbox"}),
        doc_y=doc_y,
        in_dominant_group=in_dominant_group,
        href=href,
        nearby_text=nearby_text,
    )


def make_snapshot(elements: list[Element]) -> Snapshot:
    return Snapshot(status="success", url="https://example.com", elements=elements)


class TestPruningPolicies:
    """Tests for category-specific pruning behavior."""

    def test_shopping_policy_keeps_price_and_add_to_cart(self) -> None:
        snap = make_snapshot(
            [
                make_element(
                    id=1,
                    role="link",
                    text="Rainbow Trout Trucker",
                    importance=900,
                    doc_y=100,
                    in_dominant_group=True,
                    href="/product/hat",
                ),
                make_element(
                    id=2,
                    role="text",
                    text="$32.50",
                    importance=850,
                    doc_y=120,
                    in_dominant_group=True,
                    nearby_text="Price",
                ),
                make_element(
                    id=3,
                    role="button",
                    text="Add to Cart",
                    importance=950,
                    doc_y=140,
                    in_dominant_group=True,
                ),
                make_element(
                    id=4,
                    role="link",
                    text="Privacy Policy",
                    importance=100,
                    doc_y=900,
                    href="/privacy",
                ),
            ]
        )

        ctx = prune_snapshot_for_task(
            snap,
            goal="add the product to cart",
            category=PruningTaskCategory.SHOPPING,
        )

        kept_ids = {node.id for node in ctx.nodes}
        assert {1, 2, 3}.issubset(kept_ids)
        assert 4 not in kept_ids

    def test_form_filling_policy_keeps_inputs_and_submit(self) -> None:
        snap = make_snapshot(
            [
                make_element(id=10, role="textbox", text="Email", importance=850, doc_y=100),
                make_element(id=11, role="textbox", text="Message", importance=800, doc_y=140),
                make_element(id=12, role="button", text="Submit", importance=900, doc_y=180),
                make_element(id=13, role="link", text="Company Blog", importance=200, doc_y=800, href="/blog"),
            ]
        )

        ctx = prune_snapshot_for_task(
            snap,
            goal="fill out the contact form and submit it",
            category=PruningTaskCategory.FORM_FILLING,
        )

        kept_ids = {node.id for node in ctx.nodes}
        assert {10, 11, 12}.issubset(kept_ids)
        assert 13 not in kept_ids

    def test_search_policy_keeps_search_box_and_results(self) -> None:
        snap = make_snapshot(
            [
                make_element(id=20, role="searchbox", text="Search", importance=950, doc_y=50),
                make_element(
                    id=21,
                    role="link",
                    text="Best Trail Shoes",
                    importance=900,
                    doc_y=180,
                    in_dominant_group=True,
                    href="/trail-shoes",
                ),
                make_element(id=22, role="text", text="Footer links", importance=50, doc_y=1200),
            ]
        )

        ctx = prune_snapshot_for_task(
            snap,
            goal="search for trail shoes and open the best result",
            category=PruningTaskCategory.SEARCH,
        )

        kept_ids = {node.id for node in ctx.nodes}
        assert 20 in kept_ids
        assert 21 in kept_ids
        assert 22 not in kept_ids

    def test_serializer_outputs_compact_skeleton_dom(self) -> None:
        snap = make_snapshot(
            [
                make_element(
                    id=30,
                    role="button",
                    text="Checkout",
                    importance=900,
                    doc_y=200,
                    in_dominant_group=True,
                )
            ]
        )

        ctx = prune_snapshot_for_task(
            snap,
            goal="go to checkout",
            category=PruningTaskCategory.CHECKOUT,
        )
        result = serialize_pruned_snapshot(ctx)

        assert "Category: checkout" in result
        assert "URL: https://example.com" in result
        assert "[30] button text=\"Checkout\"" in result
