"""
CAPTCHA Handling Example

Demonstrates different CAPTCHA solving strategies:
1. Abort policy: Fail immediately on CAPTCHA
2. Human handoff: Wait for manual solve
3. External solver: Integrate with 2Captcha, CapSolver, etc.
4. Custom handler: Implement your own solving logic

Prerequisites:
    pip install predicate-sdk openai

For external solvers:
    pip install 2captcha-python  # For 2Captcha
    pip install capsolver       # For CapSolver
"""

import asyncio
from typing import Any

from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime
from predicate.agents import (
    AutomationTask,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    TaskCategory,
)
from predicate.agents.browser_agent import CaptchaConfig
from predicate.captcha import CaptchaContext, CaptchaResolution
from predicate.captcha_strategies import (
    ExternalSolver,
    HumanHandoffSolver,
    VisionSolver,
)
from predicate.llm_provider import OpenAIProvider


def create_abort_config() -> PlannerExecutorConfig:
    """
    Abort policy: Fail immediately when CAPTCHA is detected.

    This is the default behavior. Use this when:
    - You want to fail fast and handle CAPTCHA externally
    - Your automation should not encounter CAPTCHAs (e.g., authenticated sessions)
    """
    return PlannerExecutorConfig(
        captcha=CaptchaConfig(
            policy="abort",
            min_confidence=0.7,  # Confidence threshold for CAPTCHA detection
        ),
    )


def create_human_handoff_config() -> PlannerExecutorConfig:
    """
    Human handoff: Wait for manual CAPTCHA solve.

    Use this when:
    - Running with a visible browser window
    - A human operator can solve CAPTCHAs manually
    - Using live session URLs for remote debugging
    """
    return PlannerExecutorConfig(
        captcha=CaptchaConfig(
            policy="callback",
            handler=HumanHandoffSolver(
                message="Please solve the CAPTCHA in the browser window",
                timeout_ms=180_000,  # 3 minute timeout
                poll_ms=2_000,  # Check every 2 seconds
            ),
        ),
    )


def create_external_solver_config() -> PlannerExecutorConfig:
    """
    External solver: Integrate with solving services.

    This example shows integration with 2Captcha.
    Similar patterns work for CapSolver, Anti-Captcha, etc.
    """

    def solve_with_2captcha(ctx: CaptchaContext) -> bool:
        """
        Solve CAPTCHA using 2Captcha service.

        This is a simplified example. In production:
        - Handle different CAPTCHA types (reCAPTCHA, hCaptcha, etc.)
        - Properly inject solutions into the page
        - Handle errors and timeouts
        """
        try:
            # Import 2Captcha (pip install 2captcha-python)
            # from twocaptcha import TwoCaptcha
            # solver = TwoCaptcha('YOUR_API_KEY')

            # Get CAPTCHA type from diagnostics
            captcha_type = getattr(ctx.captcha, "type", "unknown")
            print(f"CAPTCHA detected: {captcha_type}")
            print(f"URL: {ctx.url}")
            print(f"Screenshot: {ctx.screenshot_path}")

            # Example for reCAPTCHA v2
            if captcha_type == "recaptcha":
                sitekey = getattr(ctx.captcha, "sitekey", None)
                if sitekey:
                    # result = solver.recaptcha(sitekey=sitekey, url=ctx.url)
                    # solution = result['code']

                    # Inject solution using page_control
                    if ctx.page_control:
                        # await ctx.page_control.evaluate_js(f"""
                        #     document.getElementById('g-recaptcha-response').innerHTML = '{solution}';
                        # """)
                        pass

            return True  # Signal that solving was attempted

        except Exception as e:
            print(f"CAPTCHA solve failed: {e}")
            return False

    return PlannerExecutorConfig(
        captcha=CaptchaConfig(
            policy="callback",
            handler=ExternalSolver(
                resolver=solve_with_2captcha,
                message="Solving CAPTCHA via external service",
                timeout_ms=180_000,
                poll_ms=5_000,
            ),
        ),
    )


def create_custom_handler_config() -> PlannerExecutorConfig:
    """
    Custom handler: Full control over CAPTCHA handling.

    Use this for:
    - Complex solving logic
    - Integration with internal systems
    - Custom retry/escalation strategies
    """

    async def custom_captcha_handler(ctx: CaptchaContext) -> CaptchaResolution:
        """
        Custom CAPTCHA handler with full context access.

        CaptchaContext provides:
        - ctx.run_id: Current automation run ID
        - ctx.step_index: Current step being executed
        - ctx.url: Page URL where CAPTCHA appeared
        - ctx.source: Where CAPTCHA was detected
        - ctx.captcha: CaptchaDiagnostics with type, sitekey, etc.
        - ctx.screenshot_path: Path to screenshot
        - ctx.frames_dir: Directory with frame images
        - ctx.snapshot_path: Path to DOM snapshot
        - ctx.live_session_url: URL for live debugging
        - ctx.page_control: Hook for JS evaluation
        """

        print(f"[Custom Handler] CAPTCHA at step {ctx.step_index}")
        print(f"[Custom Handler] URL: {ctx.url}")
        print(f"[Custom Handler] Type: {getattr(ctx.captcha, 'type', 'unknown')}")

        # Example: Check if we have a live session for manual intervention
        if ctx.live_session_url:
            print(f"[Custom Handler] Live session: {ctx.live_session_url}")

            # Return wait_until_cleared for human intervention
            return CaptchaResolution(
                action="wait_until_cleared",
                message="CAPTCHA detected - please solve manually via live session",
                handled_by="human",
                timeout_ms=120_000,
                poll_ms=3_000,
            )

        # Example: For certain sites, retry with new session
        if "problematic-site.com" in ctx.url:
            return CaptchaResolution(
                action="retry_new_session",
                message="Retrying with fresh session",
                handled_by="unknown",
            )

        # Default: Abort if we can't handle
        return CaptchaResolution(
            action="abort",
            message="Cannot handle CAPTCHA automatically",
            handled_by="unknown",
        )

    return PlannerExecutorConfig(
        captcha=CaptchaConfig(
            policy="callback",
            handler=custom_captcha_handler,
            timeout_ms=300_000,  # 5 minute overall timeout
            min_confidence=0.7,
        ),
    )


async def run_with_captcha_handling():
    """Example: Run automation with CAPTCHA handling."""
    print("\n=== CAPTCHA Handling Example ===\n")

    planner = OpenAIProvider(model="gpt-4o")
    executor = OpenAIProvider(model="gpt-4o-mini")

    # Choose a CAPTCHA handling strategy
    # config = create_abort_config()            # Fail on CAPTCHA
    config = create_human_handoff_config()  # Wait for manual solve
    # config = create_external_solver_config()  # Use 2Captcha
    # config = create_custom_handler_config()   # Custom logic

    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
        config=config,
    )

    task = AutomationTask(
        task_id="captcha-test",
        starting_url="https://example.com",  # Replace with actual site
        task="Complete the signup form",
        category=TaskCategory.FORM_FILL,
        enable_recovery=True,
    )

    async with AsyncPredicateBrowser() as browser:
        page = await browser.new_page()
        await page.goto(task.starting_url)

        runtime = AgentRuntime.from_page(page)
        result = await agent.run(runtime, task)

        print(f"\nResult: {'Success' if result.success else 'Failed'}")
        if result.error:
            print(f"Error: {result.error}")


async def demonstrate_captcha_configs():
    """Show different CAPTCHA configurations without running."""
    print("=" * 60)
    print("CAPTCHA Configuration Examples")
    print("=" * 60)

    print("\n1. ABORT Policy (default)")
    print("-" * 40)
    config = create_abort_config()
    print(f"   Policy: {config.captcha.policy}")
    print(f"   Min confidence: {config.captcha.min_confidence}")

    print("\n2. HUMAN HANDOFF")
    print("-" * 40)
    config = create_human_handoff_config()
    print(f"   Policy: {config.captcha.policy}")
    print("   Handler: HumanHandoffSolver")
    print("   - Waits for manual CAPTCHA solve")
    print("   - Useful with visible browser or live sessions")

    print("\n3. EXTERNAL SOLVER")
    print("-" * 40)
    config = create_external_solver_config()
    print(f"   Policy: {config.captcha.policy}")
    print("   Handler: ExternalSolver")
    print("   - Integrates with 2Captcha, CapSolver, etc.")
    print("   - Requires API key from solving service")

    print("\n4. CUSTOM HANDLER")
    print("-" * 40)
    config = create_custom_handler_config()
    print(f"   Policy: {config.captcha.policy}")
    print("   Handler: custom async function")
    print("   - Full control over solving logic")
    print("   - Access to CaptchaContext with all details")

    print("\n" + "=" * 60)
    print("CaptchaContext fields:")
    print("=" * 60)
    print("  - run_id: Current automation run ID")
    print("  - step_index: Current step being executed")
    print("  - url: Page URL where CAPTCHA appeared")
    print("  - source: Where CAPTCHA was detected")
    print("  - captcha: CaptchaDiagnostics (type, sitekey, etc.)")
    print("  - screenshot_path: Path to screenshot")
    print("  - frames_dir: Directory with frame images")
    print("  - live_session_url: URL for live debugging")
    print("  - page_control: Hook for JS evaluation")

    print("\n" + "=" * 60)
    print("CaptchaResolution actions:")
    print("=" * 60)
    print("  - abort: Stop automation immediately")
    print("  - retry_new_session: Clear cookies and retry")
    print("  - wait_until_cleared: Poll until CAPTCHA is gone")


async def main():
    """Run examples."""
    # Show configuration options (no browser needed)
    await demonstrate_captcha_configs()

    # Uncomment to run with actual browser:
    # await run_with_captcha_handling()


if __name__ == "__main__":
    asyncio.run(main())
