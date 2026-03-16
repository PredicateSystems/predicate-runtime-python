#!/usr/bin/env python3
"""
PlannerExecutorAgent example with custom configuration.

This example demonstrates various configuration options:
- Snapshot escalation (enable/disable, custom step sizes)
- Scroll-after-escalation (find elements outside viewport)
- Retry configuration (timeouts, max attempts)
- Vision fallback settings
- Pre-step verification (skip if predicates pass)
- Recovery navigation (track last good URL)
- Modal dismissal (auto-dismiss overlays, custom patterns for i18n)

Usage:
    export OPENAI_API_KEY="sk-..."
    python custom_config_example.py
"""

from __future__ import annotations

import asyncio
import os

from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime
from predicate.agents import (
    CheckoutDetectionConfig,
    ModalDismissalConfig,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    RecoveryNavigationConfig,
    RetryConfig,
    SnapshotEscalationConfig,
)
from predicate.agents.browser_agent import VisionFallbackConfig
from predicate.backends.playwright_backend import PlaywrightBackend
from predicate.llm_provider import OpenAIProvider


async def example_default_config() -> None:
    """Default configuration: escalation enabled, step=30, scroll enabled."""
    print("\n--- Example 1: Default Config ---")
    print("Escalation: 60 -> 90 -> 120 -> 150 -> 180 -> 200")
    print("Scroll-after-escalation: down (x3), up (x3)")

    config = PlannerExecutorConfig()

    print(f"  snapshot.enabled: {config.snapshot.enabled}")
    print(f"  snapshot.limit_base: {config.snapshot.limit_base}")
    print(f"  snapshot.limit_step: {config.snapshot.limit_step}")
    print(f"  snapshot.limit_max: {config.snapshot.limit_max}")
    print(f"  snapshot.scroll_after_escalation: {config.snapshot.scroll_after_escalation}")
    print(f"  snapshot.scroll_max_attempts: {config.snapshot.scroll_max_attempts}")
    print(f"  snapshot.scroll_directions: {config.snapshot.scroll_directions}")


async def example_disabled_escalation() -> None:
    """Disable escalation: always use limit_base."""
    print("\n--- Example 2: Disabled Escalation ---")
    print("Escalation: disabled (always 60)")

    config = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(enabled=False),
    )

    print(f"  snapshot.enabled: {config.snapshot.enabled}")
    print(f"  snapshot.limit_base: {config.snapshot.limit_base}")


async def example_custom_step_size() -> None:
    """Custom step size for faster escalation."""
    print("\n--- Example 3: Custom Step Size ---")
    print("Escalation: 60 -> 110 -> 160 -> 200 (step=50)")

    config = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            limit_step=50,  # Larger steps = fewer iterations
        ),
    )

    print(f"  snapshot.limit_step: {config.snapshot.limit_step}")


async def example_custom_limits() -> None:
    """Custom base and max limits."""
    print("\n--- Example 4: Custom Limits ---")
    print("Escalation: 100 -> 125 -> 150 -> 175 -> 200 -> 225 -> 250")

    config = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            limit_base=100,  # Start higher
            limit_step=25,   # Smaller increments
            limit_max=250,   # Higher maximum
        ),
    )

    print(f"  snapshot.limit_base: {config.snapshot.limit_base}")
    print(f"  snapshot.limit_step: {config.snapshot.limit_step}")
    print(f"  snapshot.limit_max: {config.snapshot.limit_max}")


async def example_scroll_after_escalation() -> None:
    """Scroll-after-escalation configuration."""
    print("\n--- Example 4b: Scroll-after-Escalation ---")
    print("After exhausting limit escalation, scroll to find elements outside viewport")

    # Default: scroll down first, then up
    config_default = PlannerExecutorConfig()
    print(f"  Default scroll_directions: {config_default.snapshot.scroll_directions}")

    # Disable scroll-after-escalation (only use limit escalation)
    config_no_scroll = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            scroll_after_escalation=False,
        ),
    )
    print(f"  Disabled: scroll_after_escalation={config_no_scroll.snapshot.scroll_after_escalation}")

    # Custom: more scroll attempts, down only (useful for infinite scroll pages)
    config_down_only = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            scroll_after_escalation=True,
            scroll_max_attempts=5,       # More scrolls
            scroll_directions=("down",), # Only scroll down
        ),
    )
    print(f"  Down-only: scroll_directions={config_down_only.snapshot.scroll_directions}")
    print(f"  Down-only: scroll_max_attempts={config_down_only.snapshot.scroll_max_attempts}")

    # Custom: try up first (e.g., for elements at top of page)
    config_up_first = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            scroll_directions=("up", "down"),  # Try up before down
        ),
    )
    print(f"  Up-first: scroll_directions={config_up_first.snapshot.scroll_directions}")


async def example_retry_config() -> None:
    """Custom retry configuration."""
    print("\n--- Example 5: Retry Config ---")

    config = PlannerExecutorConfig(
        retry=RetryConfig(
            verify_timeout_s=15.0,       # Longer timeout for slow pages
            verify_poll_s=0.3,           # Faster polling
            verify_max_attempts=10,      # More verification attempts
            executor_repair_attempts=3,  # More repair attempts
            max_replans=2,               # Allow 2 replans on failure
        ),
    )

    print(f"  retry.verify_timeout_s: {config.retry.verify_timeout_s}")
    print(f"  retry.verify_max_attempts: {config.retry.verify_max_attempts}")
    print(f"  retry.max_replans: {config.retry.max_replans}")


async def example_vision_fallback() -> None:
    """Vision fallback configuration."""
    print("\n--- Example 6: Vision Fallback ---")

    config = PlannerExecutorConfig(
        vision=VisionFallbackConfig(
            enabled=True,
            max_vision_calls=5,                      # Up to 5 vision calls per run
            trigger_requires_vision=True,            # Trigger on require_vision status
            trigger_canvas_or_low_actionables=True,  # Trigger on canvas pages
        ),
    )

    print(f"  vision.enabled: {config.vision.enabled}")
    print(f"  vision.max_vision_calls: {config.vision.max_vision_calls}")


async def example_pre_step_verification() -> None:
    """Pre-step verification configuration."""
    print("\n--- Example 7: Pre-step Verification ---")

    # Default: enabled (steps are skipped if predicates already pass)
    config_enabled = PlannerExecutorConfig(
        pre_step_verification=True,  # Default
    )
    print(f"  pre_step_verification (default): {config_enabled.pre_step_verification}")

    # Disabled: always execute steps even if predicates pass
    config_disabled = PlannerExecutorConfig(
        pre_step_verification=False,
    )
    print(f"  pre_step_verification (disabled): {config_disabled.pre_step_verification}")
    print("  When disabled, all steps are executed even if already satisfied")


async def example_recovery_navigation() -> None:
    """Recovery navigation configuration."""
    print("\n--- Example 8: Recovery Navigation ---")

    config = PlannerExecutorConfig(
        recovery=RecoveryNavigationConfig(
            enabled=True,
            max_recovery_attempts=3,
            track_successful_urls=True,
        ),
    )

    print(f"  recovery.enabled: {config.recovery.enabled}")
    print(f"  recovery.max_recovery_attempts: {config.recovery.max_recovery_attempts}")
    print(f"  recovery.track_successful_urls: {config.recovery.track_successful_urls}")
    print("  Tracks last_known_good_url for recovery when agent gets off-track")


async def example_modal_dismissal() -> None:
    """Modal dismissal configuration for auto-dismissing overlays."""
    print("\n--- Example 9: Modal Dismissal ---")
    print("Auto-dismiss blocking modals, drawers, and popups")

    # Default: enabled with common English patterns
    config_default = PlannerExecutorConfig()
    print(f"  Default enabled: {config_default.modal.enabled}")
    print(f"  Default patterns: {config_default.modal.dismiss_patterns[:5]}...")

    # Disable modal dismissal
    config_disabled = PlannerExecutorConfig(
        modal=ModalDismissalConfig(enabled=False),
    )
    print(f"  Disabled: modal.enabled={config_disabled.modal.enabled}")

    # Custom patterns for German sites
    config_german = PlannerExecutorConfig(
        modal=ModalDismissalConfig(
            dismiss_patterns=(
                "nein danke",       # No thanks
                "nicht jetzt",      # Not now
                "abbrechen",        # Cancel
                "schließen",        # Close
                "überspringen",     # Skip
                "später",           # Later
                "ablehnen",         # Decline
                "weiter",           # Continue
            ),
            dismiss_icons=("x", "×", "✕"),  # Icons are universal
        ),
    )
    print(f"  German patterns: {config_german.modal.dismiss_patterns[:4]}...")

    # Custom patterns for Spanish sites
    config_spanish = PlannerExecutorConfig(
        modal=ModalDismissalConfig(
            dismiss_patterns=(
                "no gracias",       # No thanks
                "ahora no",         # Not now
                "cancelar",         # Cancel
                "cerrar",           # Close
                "omitir",           # Skip
                "más tarde",        # Later
                "rechazar",         # Reject
                "continuar",        # Continue
            ),
        ),
    )
    print(f"  Spanish patterns: {config_spanish.modal.dismiss_patterns[:4]}...")

    # Custom patterns for French sites
    config_french = PlannerExecutorConfig(
        modal=ModalDismissalConfig(
            dismiss_patterns=(
                "non merci",        # No thanks
                "pas maintenant",   # Not now
                "annuler",          # Cancel
                "fermer",           # Close
                "passer",           # Skip
                "plus tard",        # Later
                "refuser",          # Refuse
                "continuer",        # Continue
            ),
        ),
    )
    print(f"  French patterns: {config_french.modal.dismiss_patterns[:4]}...")

    # Custom patterns for Japanese sites
    config_japanese = PlannerExecutorConfig(
        modal=ModalDismissalConfig(
            dismiss_patterns=(
                "いいえ",           # No
                "後で",             # Later
                "閉じる",           # Close
                "キャンセル",       # Cancel
                "スキップ",         # Skip
                "続ける",           # Continue
                "結構です",         # No thank you
            ),
        ),
    )
    print(f"  Japanese patterns: {config_japanese.modal.dismiss_patterns[:4]}...")

    # Combined multilingual config
    config_multilingual = PlannerExecutorConfig(
        modal=ModalDismissalConfig(
            dismiss_patterns=(
                # English
                "no thanks", "not now", "close", "skip", "cancel",
                # German
                "nein danke", "schließen", "abbrechen",
                # Spanish
                "no gracias", "cerrar", "cancelar",
                # French
                "non merci", "fermer", "annuler",
            ),
        ),
    )
    print(f"  Multilingual: {len(config_multilingual.modal.dismiss_patterns)} patterns")


async def example_checkout_detection() -> None:
    """Checkout page detection configuration."""
    print("\n--- Example 10: Checkout Detection ---")
    print("Auto-detect checkout pages and trigger continuation replanning")

    # Default: enabled with common checkout patterns
    config_default = PlannerExecutorConfig()
    print(f"  Default enabled: {config_default.checkout.enabled}")
    print(f"  Default URL patterns: {config_default.checkout.url_patterns[:5]}...")

    # Disable checkout detection
    config_disabled = PlannerExecutorConfig(
        checkout=CheckoutDetectionConfig(enabled=False),
    )
    print(f"  Disabled: checkout.enabled={config_disabled.checkout.enabled}")

    # Custom patterns for German e-commerce sites
    config_german = PlannerExecutorConfig(
        checkout=CheckoutDetectionConfig(
            url_patterns=(
                "/warenkorb",       # Cart
                "/kasse",           # Checkout
                "/zahlung",         # Payment
                "/bestellung",      # Order
                "/anmelden",        # Sign-in
            ),
            element_patterns=(
                "zur kasse",        # To checkout
                "warenkorb",        # Shopping cart
                "jetzt kaufen",     # Buy now
                "anmelden",         # Sign in
            ),
        ),
    )
    print(f"  German URL patterns: {config_german.checkout.url_patterns[:3]}...")

    # Disable replan trigger (just detect, don't act)
    config_detect_only = PlannerExecutorConfig(
        checkout=CheckoutDetectionConfig(
            enabled=True,
            trigger_replan=False,  # Only detect, don't trigger continuation
        ),
    )
    print(f"  Detect-only: trigger_replan={config_detect_only.checkout.trigger_replan}")


async def example_full_custom() -> None:
    """Full custom configuration with all options."""
    print("\n--- Example 11: Full Custom Config ---")

    config = PlannerExecutorConfig(
        # Snapshot escalation with scroll-after-escalation
        snapshot=SnapshotEscalationConfig(
            enabled=True,
            limit_base=80,
            limit_step=40,
            limit_max=240,
            # Scroll to find elements outside viewport
            scroll_after_escalation=True,
            scroll_max_attempts=3,
            scroll_directions=("down", "up"),
        ),
        # Retry settings
        retry=RetryConfig(
            verify_timeout_s=12.0,
            verify_poll_s=0.4,
            verify_max_attempts=6,
            max_replans=2,
        ),
        # Vision fallback
        vision=VisionFallbackConfig(
            enabled=True,
            max_vision_calls=3,
        ),
        # Recovery navigation
        recovery=RecoveryNavigationConfig(
            enabled=True,
            max_recovery_attempts=2,
        ),
        # Modal dismissal (auto-dismiss blocking overlays)
        modal=ModalDismissalConfig(
            enabled=True,
            max_attempts=2,
        ),
        # Checkout detection (continue workflow on checkout pages)
        checkout=CheckoutDetectionConfig(
            enabled=True,
            trigger_replan=True,
        ),
        # Pre-step verification
        pre_step_verification=True,
        # Planner settings
        planner_max_tokens=3000,
        planner_temperature=0.0,
        # Executor settings
        executor_max_tokens=128,
        executor_temperature=0.0,
        # Tracing
        trace_screenshots=True,
        trace_screenshot_format="jpeg",
        trace_screenshot_quality=85,
    )

    print("  Full config created successfully!")
    print(f"  Escalation: {config.snapshot.limit_base} -> ... -> {config.snapshot.limit_max}")
    print(f"  Scroll-after-escalation: {config.snapshot.scroll_after_escalation}")
    print(f"  Max replans: {config.retry.max_replans}")
    print(f"  Vision enabled: {config.vision.enabled}")
    print(f"  Pre-step verification: {config.pre_step_verification}")
    print(f"  Recovery enabled: {config.recovery.enabled}")
    print(f"  Modal dismissal: {config.modal.enabled}")
    print(f"  Checkout detection: {config.checkout.enabled}")


async def example_run_with_config() -> None:
    """Run agent with custom config."""
    print("\n--- Example 12: Run Agent with Custom Config ---")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  Skipping (no OPENAI_API_KEY)")
        return

    predicate_api_key = os.getenv("PREDICATE_API_KEY")

    # Create config optimized for reliability
    config = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            enabled=True,
            limit_base=60,
            limit_step=30,
            limit_max=180,
        ),
        retry=RetryConfig(
            verify_timeout_s=10.0,
            max_replans=1,
        ),
    )

    planner = OpenAIProvider(model="gpt-4o")
    executor = OpenAIProvider(model="gpt-4o-mini")

    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
        config=config,
    )

    async with AsyncPredicateBrowser(
        api_key=predicate_api_key,
        headless=True,
    ) as browser:
        page = await browser.new_page()
        await page.goto("https://example.com")

        backend = PlaywrightBackend(page)
        runtime = AgentRuntime(backend=backend)

        result = await agent.run(
            runtime=runtime,
            task="Verify example.com is loaded",
        )

        print(f"  Success: {result.success}")
        print(f"  Steps: {result.steps_completed}/{result.steps_total}")


async def main() -> None:
    print("PlannerExecutorAgent Configuration Examples")
    print("=" * 50)

    await example_default_config()
    await example_disabled_escalation()
    await example_custom_step_size()
    await example_custom_limits()
    await example_scroll_after_escalation()
    await example_retry_config()
    await example_vision_fallback()
    await example_pre_step_verification()
    await example_recovery_navigation()
    await example_modal_dismissal()
    await example_checkout_detection()
    await example_full_custom()
    await example_run_with_config()

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
