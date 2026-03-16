# PlannerExecutorAgent User Manual

The `PlannerExecutorAgent` is a two-tier agent architecture for browser automation that separates planning from execution:

- **Planner**: Generates JSON execution plans with verification predicates (uses 7B+ model)
- **Executor**: Executes each step with snapshot-first verification (uses 3B-7B model)

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [AutomationTask](#automationtask)
4. [Configuration](#configuration)
5. [CAPTCHA Handling](#captcha-handling)
6. [Permissions](#permissions)
7. [Modal and Dialog Handling](#modal-and-dialog-handling)
8. [Recovery and Rollback](#recovery-and-rollback)
9. [Custom Heuristics](#custom-heuristics)
10. [Tracing](#tracing)
11. [Examples](#examples)

---

## Quick Start

```python
from predicate.agents import PlannerExecutorAgent, PlannerExecutorConfig, AutomationTask
from predicate.llm_provider import OpenAIProvider
from predicate import AsyncPredicateBrowser, AgentRuntime

# Initialize LLM providers
planner = OpenAIProvider(model="gpt-4o")
executor = OpenAIProvider(model="gpt-4o-mini")

# Create agent
agent = PlannerExecutorAgent(
    planner=planner,
    executor=executor,
    config=PlannerExecutorConfig(),
)

# Run automation
async with AsyncPredicateBrowser() as browser:
    runtime = AgentRuntime.from_browser(browser)

    result = await agent.run(
        runtime=runtime,
        task="Search for laptops on Amazon",
        start_url="https://amazon.com",
    )

    print(f"Success: {result.success}")
```

---

## Core Concepts

### Snapshot-First Verification

The agent uses a **snapshot-first** approach:
1. Capture DOM snapshot before each action
2. Select element using heuristics or LLM
3. Execute action
4. Verify result using predicates

### Predicate Verification

Each plan step includes verification predicates:

```json
{
  "id": 1,
  "goal": "Click search button",
  "action": "CLICK",
  "intent": "search",
  "verify": [
    {"predicate": "url_contains", "args": ["/search"]},
    {"predicate": "exists", "args": ["role=list"]}
  ]
}
```

Available predicates:
- `url_contains(substring)`: URL contains the given string
- `url_matches(pattern)`: URL matches regex pattern
- `exists(selector)`: Element matching selector exists
- `not_exists(selector)`: Element does not exist
- `element_count(selector, min, max)`: Element count within range
- `any_of(predicates...)`: Any predicate is true
- `all_of(predicates...)`: All predicates are true

Selectors: `role=button`, `role=link`, `text~'text'`, `role=textbox`, etc.

### Snapshot Escalation and Scroll-after-Escalation

When capturing DOM snapshots, the agent uses **incremental limit escalation** to ensure it captures enough elements:

1. Start with `limit_base` (default: 60 elements)
2. If element count is low (<10), escalate by `limit_step` (default: 30)
3. Continue until `limit_max` (default: 200) is reached

After exhausting limit escalation, if the target element is still not found, the agent can use **scroll-after-escalation** to find elements outside the current viewport. This feature only triggers when ALL conditions are met:

1. `scroll_after_escalation=True` (default)
2. The step action is `CLICK` (not TYPE_AND_SUBMIT, NAVIGATE, etc.)
3. The step has a specific `intent` field
4. Custom `intent_heuristics` are injected into the agent

When triggered:
1. Scroll down (up to `scroll_max_attempts` times, default: 3)
2. Take a new snapshot after each scroll
3. Check if target element is now visible using intent heuristics
4. If still not found, scroll up
5. Return the best snapshot found

This is particularly useful for elements like "Add to Cart" buttons that may be below the initial viewport on product pages, when you have custom heuristics to detect them.

```python
from predicate.agents import SnapshotEscalationConfig

# Default behavior: escalation + scroll enabled
config = SnapshotEscalationConfig()

# Disable scroll-after-escalation (only use limit escalation)
config = SnapshotEscalationConfig(scroll_after_escalation=False)

# Custom scroll settings
config = SnapshotEscalationConfig(
    scroll_after_escalation=True,
    scroll_max_attempts=5,           # More scrolls per direction
    scroll_directions=("down",),     # Only scroll down
)

# Try scrolling up first (useful for elements at top of page)
config = SnapshotEscalationConfig(
    scroll_directions=("up", "down"),  # Try up before down
)
```

---

## AutomationTask

The `AutomationTask` model provides a flexible way to define browser automation tasks:

```python
from predicate.agents import AutomationTask, TaskCategory, SuccessCriteria

# Basic task
task = AutomationTask(
    task_id="purchase-laptop-001",
    starting_url="https://amazon.com",
    task="Find a laptop under $1000 with good reviews and add to cart",
)

# Task with category hint
task = AutomationTask(
    task_id="purchase-laptop-001",
    starting_url="https://amazon.com",
    task="Find a laptop under $1000 with good reviews and add to cart",
    category=TaskCategory.TRANSACTION,
    max_steps=50,
    enable_recovery=True,
)

# Task with success criteria
task = task.with_success_criteria(
    {"predicate": "url_contains", "args": ["/cart"]},
    {"predicate": "exists", "args": [".cart-item"]},
)

# Run with AutomationTask
result = await agent.run(runtime, task)
```

### TaskCategory

Categories help the planner and executor make better decisions:

| Category | Use Case |
|----------|----------|
| `NAVIGATION` | Navigate to a destination |
| `SEARCH` | Search and find information |
| `FORM_FILL` | Fill out forms |
| `EXTRACTION` | Extract data from pages |
| `TRANSACTION` | Purchase, submit, create actions |
| `VERIFICATION` | Verify state/information exists |

### Extraction Tasks

For data extraction tasks:

```python
from predicate.agents import ExtractionSpec

task = AutomationTask(
    task_id="extract-product-info",
    starting_url="https://amazon.com/dp/B0...",
    task="Extract the product name, price, and rating",
    category=TaskCategory.EXTRACTION,
    extraction_spec=ExtractionSpec(
        output_schema={"name": "str", "price": "float", "rating": "float"},
        format="json",
    ),
)
```

---

## Configuration

### PlannerExecutorConfig

```python
from predicate.agents import (
    PlannerExecutorConfig,
    SnapshotEscalationConfig,
    RetryConfig,
    RecoveryNavigationConfig,
)
from predicate.agents.browser_agent import VisionFallbackConfig, CaptchaConfig

config = PlannerExecutorConfig(
    # Snapshot escalation: progressively increase limit on low element count
    # After exhausting limit escalation, scrolls to find elements outside viewport
    snapshot=SnapshotEscalationConfig(
        enabled=True,
        limit_base=60,      # Initial snapshot limit
        limit_step=30,      # Increment per escalation
        limit_max=200,      # Maximum limit
        # Scroll-after-escalation: find elements below/above viewport
        scroll_after_escalation=True,   # Enable scrolling after limit exhaustion
        scroll_max_attempts=3,          # Max scrolls per direction
        scroll_directions=("down", "up"),  # Directions to try
    ),

    # Retry configuration
    retry=RetryConfig(
        verify_timeout_s=10.0,
        verify_poll_s=0.5,
        verify_max_attempts=5,
        executor_repair_attempts=2,
        max_replans=1,
    ),

    # Vision fallback for canvas pages or low-confidence snapshots
    vision=VisionFallbackConfig(
        enabled=True,
        max_vision_calls=3,
        trigger_requires_vision=True,
        trigger_canvas_or_low_actionables=True,
    ),

    # CAPTCHA handling
    captcha=CaptchaConfig(),  # See CAPTCHA section below

    # Recovery navigation
    recovery=RecoveryNavigationConfig(
        enabled=True,
        max_recovery_attempts=2,
        track_successful_urls=True,
    ),

    # LLM settings
    planner_max_tokens=2048,
    planner_temperature=0.0,
    executor_max_tokens=96,
    executor_temperature=0.0,

    # Stabilization
    stabilize_enabled=True,
    stabilize_poll_s=0.35,
    stabilize_max_attempts=6,

    # Pre-step verification: skip step if predicates already pass
    pre_step_verification=True,

    # Tracing
    trace_screenshots=True,
    trace_screenshot_format="jpeg",
    trace_screenshot_quality=80,
)

agent = PlannerExecutorAgent(
    planner=planner,
    executor=executor,
    config=config,
)
```

---

## CAPTCHA Handling

The SDK provides flexible CAPTCHA handling through the `CaptchaConfig` system.

### CAPTCHA Policies

| Policy | Behavior |
|--------|----------|
| `abort` | Fail immediately when CAPTCHA is detected (default) |
| `callback` | Invoke a handler and wait for resolution |

### Using a CAPTCHA Handler

```python
from predicate.agents.browser_agent import CaptchaConfig
from predicate.captcha_strategies import (
    HumanHandoffSolver,
    ExternalSolver,
    VisionSolver,
)

# Option 1: Human handoff - waits for manual solve
config = PlannerExecutorConfig(
    captcha=CaptchaConfig(
        policy="callback",
        handler=HumanHandoffSolver(
            message="Please solve the CAPTCHA in the browser window",
            timeout_ms=120_000,  # 2 minute timeout
        ),
    ),
)

# Option 2: External solver integration (e.g., 2Captcha, CapSolver)
def solve_with_2captcha(ctx):
    # ctx.url - current page URL
    # ctx.screenshot_path - path to screenshot
    # ctx.captcha - CaptchaDiagnostics with type info

    # Call your CAPTCHA solving service here
    # Return when solved or raise on failure
    pass

config = PlannerExecutorConfig(
    captcha=CaptchaConfig(
        policy="callback",
        handler=ExternalSolver(
            resolver=solve_with_2captcha,
            message="Solving CAPTCHA via 2Captcha",
            timeout_ms=180_000,
        ),
    ),
)

# Option 3: Vision-based solving (requires vision model)
config = PlannerExecutorConfig(
    captcha=CaptchaConfig(
        policy="callback",
        handler=VisionSolver(
            message="Attempting vision-based CAPTCHA solve",
            timeout_ms=60_000,
        ),
    ),
)
```

### CaptchaContext

When the CAPTCHA handler is invoked, it receives a `CaptchaContext`:

```python
@dataclass
class CaptchaContext:
    run_id: str              # Current run ID
    step_index: int          # Current step index
    url: str                 # Current page URL
    source: CaptchaSource    # "extension" | "gateway" | "runtime"
    captcha: CaptchaDiagnostics  # CAPTCHA type and details
    screenshot_path: str | None  # Path to screenshot
    frames_dir: str | None   # Directory with frame images
    snapshot_path: str | None    # Path to snapshot
    live_session_url: str | None # URL for live debugging
    meta: dict | None        # Additional metadata
    page_control: PageControlHook | None  # JS evaluation hook
```

### CaptchaResolution Actions

| Action | Behavior |
|--------|----------|
| `abort` | Stop automation immediately |
| `retry_new_session` | Clear cookies and retry |
| `wait_until_cleared` | Poll until CAPTCHA is cleared |

### Implementing a Custom CAPTCHA Handler

```python
from predicate.captcha import CaptchaContext, CaptchaResolution, CaptchaHandler

async def my_captcha_handler(ctx: CaptchaContext) -> CaptchaResolution:
    """Custom CAPTCHA handler with external solving service."""

    # Example: Integrate with 2Captcha
    import requests

    # Read screenshot for solving
    if ctx.screenshot_path:
        with open(ctx.screenshot_path, 'rb') as f:
            image_data = f.read()

    # Submit to solving service
    # response = requests.post("https://2captcha.com/in.php", ...)
    # solution = poll_for_solution(response['captcha_id'])

    # Inject solution if page_control is available
    if ctx.page_control:
        await ctx.page_control.evaluate_js(f"""
            document.getElementById('captcha-input').value = '{solution}';
            document.getElementById('captcha-form').submit();
        """)

    return CaptchaResolution(
        action="wait_until_cleared",
        message="CAPTCHA solved via 2Captcha",
        handled_by="customer_system",
        timeout_ms=30_000,
        poll_ms=2_000,
    )

config = PlannerExecutorConfig(
    captcha=CaptchaConfig(
        policy="callback",
        handler=my_captcha_handler,
        timeout_ms=180_000,
        min_confidence=0.7,
    ),
)
```

### Dependencies for CAPTCHA Solving

For external CAPTCHA solving services:

```bash
# For 2Captcha integration
pip install 2captcha-python

# For CapSolver integration
pip install capsolver
```

Example with 2Captcha:

```python
from twocaptcha import TwoCaptcha

solver = TwoCaptcha('YOUR_API_KEY')

def solve_with_2captcha(ctx: CaptchaContext):
    if ctx.captcha.type == "recaptcha":
        result = solver.recaptcha(
            sitekey=ctx.captcha.sitekey,
            url=ctx.url,
        )
        # Inject the solution
        if ctx.page_control:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                ctx.page_control.evaluate_js(f"""
                    document.getElementById('g-recaptcha-response').innerHTML = '{result["code"]}';
                """)
            )
    return True
```

---

## Permissions

Chrome browser permissions (geolocation, notifications, etc.) are handled at two levels:

### Startup Permissions

Applied at browser/context creation:

```python
from predicate.permissions import PermissionPolicy

policy = PermissionPolicy(
    default="grant",  # "clear" | "deny" | "grant"
    auto_grant=["geolocation", "notifications", "camera", "microphone"],
    geolocation={"latitude": 37.7749, "longitude": -122.4194},  # Mock location
    origin="https://example.com",
)

# Apply via browser configuration (implementation-dependent)
```

### Recovery Permissions

For handling permission prompts during automation:

```python
from predicate.agents.browser_agent import PermissionRecoveryConfig

config = PlannerExecutorConfig(
    # Other config...
)

# PermissionRecoveryConfig is used at agent level
permission_recovery = PermissionRecoveryConfig(
    enabled=True,
    max_restarts=1,
    auto_grant=["geolocation", "notifications"],
    geolocation={"latitude": 37.7749, "longitude": -122.4194},
    origin="https://example.com",
)
```

### Common Permissions

| Permission | Description |
|------------|-------------|
| `geolocation` | Access to device location |
| `notifications` | Push notification access |
| `camera` | Camera access |
| `microphone` | Microphone access |
| `clipboard-read` | Read clipboard |
| `clipboard-write` | Write clipboard |

---

## Modal and Dialog Handling

Modal and dialog handling is done through plan steps with heuristic hints:

### Common Modal Patterns

The SDK includes common hints for dismissing modals:

```python
from predicate.agents import COMMON_HINTS, get_common_hint

# Built-in hints for common patterns
close_hint = get_common_hint("close")        # "close", "dismiss", "x", "cancel"
accept_cookies = get_common_hint("accept_cookies")  # "accept", "allow", "agree"
```

### Handling Cookie Consent

```python
task = AutomationTask(
    task_id="example",
    starting_url="https://example.com",
    task="Accept cookies and then search for products",
)

# The planner will generate steps with heuristic hints:
# {
#     "id": 1,
#     "goal": "Accept cookie consent",
#     "action": "CLICK",
#     "intent": "accept_cookies",
#     "heuristic_hints": [
#         {
#             "intent_pattern": "accept_cookies",
#             "text_patterns": ["accept", "accept all", "allow", "agree"],
#             "role_filter": ["button"]
#         }
#     ]
# }
```

### Custom Modal Handling Heuristics

For site-specific modals, provide custom heuristics:

```python
class ModalDismissHeuristics:
    def find_element_for_intent(self, intent, elements, url, goal):
        if "dismiss" in intent.lower() or "close" in intent.lower():
            # Look for close buttons
            for el in elements:
                text = (getattr(el, "text", "") or "").lower()
                role = getattr(el, "role", "") or ""

                # Common close button patterns
                if role == "button":
                    if any(p in text for p in ["close", "dismiss", "x", "no thanks", "cancel"]):
                        return getattr(el, "id", None)

                # Close icon (×)
                if text in ["×", "x", "✕", "✖"]:
                    return getattr(el, "id", None)

        return None

    def priority_order(self):
        return ["close", "dismiss", "accept_cookies"]

agent = PlannerExecutorAgent(
    planner=planner,
    executor=executor,
    intent_heuristics=ModalDismissHeuristics(),
)
```

### Optional Substeps for Modals

The planner can generate optional substeps for edge cases:

```json
{
  "id": 2,
  "goal": "Search for laptops",
  "action": "TYPE_AND_SUBMIT",
  "input": "laptop",
  "verify": [{"predicate": "url_contains", "args": ["/search"]}],
  "optional_substeps": [
    {
      "id": 201,
      "goal": "Dismiss any modal that may have appeared",
      "action": "CLICK",
      "intent": "close"
    }
  ]
}
```

### Automatic Modal Dismissal

The agent includes automatic modal/drawer dismissal that triggers after DOM changes. This handles common blocking scenarios:

- Product protection/warranty upsells (e.g., Amazon's "Add Protection Plan")
- Cookie consent banners
- Newsletter signup popups
- Promotional overlays
- Cart upsell drawers

**How It Works:**

When a CLICK action triggers a significant DOM change (5+ new elements), the agent:
1. Detects that a modal may have appeared
2. Scans for dismissal buttons using common text patterns
3. Clicks the best matching button to clear the overlay
4. Continues with the task

**Default Configuration:**

```python
from predicate.agents import PlannerExecutorConfig, ModalDismissalConfig

# Default: enabled with common English patterns
config = PlannerExecutorConfig()
print(config.modal.enabled)  # True
print(config.modal.dismiss_patterns[:3])  # ('no thanks', 'no, thanks', 'no thank you')
```

**Disabling Modal Dismissal:**

```python
config = PlannerExecutorConfig(
    modal=ModalDismissalConfig(enabled=False),
)
```

**Custom Patterns for Internationalization:**

The dismissal patterns are fully configurable for non-English sites:

```python
# German site configuration
config = PlannerExecutorConfig(
    modal=ModalDismissalConfig(
        dismiss_patterns=(
            "nein danke",       # No thanks
            "nicht jetzt",      # Not now
            "abbrechen",        # Cancel
            "schließen",        # Close
            "überspringen",     # Skip
            "später",           # Later
        ),
        dismiss_icons=("x", "×", "✕"),  # Icons are universal
    ),
)

# Spanish site configuration
config = PlannerExecutorConfig(
    modal=ModalDismissalConfig(
        dismiss_patterns=(
            "no gracias",       # No thanks
            "ahora no",         # Not now
            "cancelar",         # Cancel
            "cerrar",           # Close
            "omitir",           # Skip
        ),
    ),
)

# French site configuration
config = PlannerExecutorConfig(
    modal=ModalDismissalConfig(
        dismiss_patterns=(
            "non merci",        # No thanks
            "pas maintenant",   # Not now
            "annuler",          # Cancel
            "fermer",           # Close
            "passer",           # Skip
        ),
    ),
)

# Japanese site configuration
config = PlannerExecutorConfig(
    modal=ModalDismissalConfig(
        dismiss_patterns=(
            "いいえ",           # No
            "後で",             # Later
            "閉じる",           # Close
            "キャンセル",       # Cancel
            "スキップ",         # Skip
        ),
    ),
)
```

**Multilingual Configuration:**

For sites that may show modals in multiple languages:

```python
config = PlannerExecutorConfig(
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
```

**Pattern Matching:**

- **Word boundary matching**: Patterns use word boundary matching to avoid false positives (e.g., "close" won't match "enclosed")
- **Icon exact matching**: Short patterns like "x", "×" require exact match
- **Pattern ordering**: Earlier patterns in the list have higher priority

---

## Recovery and Rollback

The agent tracks checkpoints for recovery when steps fail:

### How Recovery Works

1. After each successful step verification, a checkpoint is recorded
2. If a step fails repeatedly, the agent attempts recovery:
   - Navigate back to the last successful URL
   - Re-verify the page state
   - Resume from the checkpoint step
3. Limited by `max_recovery_attempts`

### Recovery Configuration

```python
from predicate.agents import AutomationTask

task = AutomationTask(
    task_id="checkout-flow",
    starting_url="https://shop.com",
    task="Complete checkout process",
    enable_recovery=True,        # Enable recovery
    max_recovery_attempts=2,     # Max attempts
)
```

### RecoveryState API

For advanced use cases:

```python
from predicate.agents import RecoveryState, RecoveryCheckpoint

state = RecoveryState(max_recovery_attempts=2)

# Record checkpoint after successful step
checkpoint = state.record_checkpoint(
    url="https://shop.com/cart",
    step_index=2,
    snapshot_digest="abc123",
    predicates_passed=["url_contains('/cart')"],
)

# Check if recovery is possible
if state.can_recover():
    checkpoint = state.consume_recovery_attempt()
    # Navigate to checkpoint.url and resume
```

---

## Custom Heuristics

### IntentHeuristics Protocol

Implement domain-specific element selection:

```python
class EcommerceHeuristics:
    def find_element_for_intent(self, intent, elements, url, goal):
        intent_lower = intent.lower()

        if "add to cart" in intent_lower or "add_to_cart" in intent_lower:
            for el in elements:
                text = (getattr(el, "text", "") or "").lower()
                role = getattr(el, "role", "") or ""
                if role == "button" and any(p in text for p in ["add to cart", "add to bag"]):
                    return getattr(el, "id", None)

        if "checkout" in intent_lower:
            for el in elements:
                text = (getattr(el, "text", "") or "").lower()
                if "checkout" in text or "proceed" in text:
                    return getattr(el, "id", None)

        return None  # Fall back to LLM

    def priority_order(self):
        return ["add_to_cart", "checkout", "search", "login"]

agent = PlannerExecutorAgent(
    planner=planner,
    executor=executor,
    intent_heuristics=EcommerceHeuristics(),
)
```

### ExecutorOverride Protocol

Validate or override executor choices:

```python
class SafetyOverride:
    def validate_choice(self, element_id, action, elements, goal):
        # Block clicks on delete buttons
        for el in elements:
            if getattr(el, "id", None) == element_id:
                text = (getattr(el, "text", "") or "").lower()
                if "delete" in text and action == "CLICK":
                    return False, None, "blocked_delete_button"
        return True, None, None

agent = PlannerExecutorAgent(
    planner=planner,
    executor=executor,
    executor_override=SafetyOverride(),
)
```

### ComposableHeuristics

The agent uses `ComposableHeuristics` internally to compose from multiple sources:

1. Planner-provided `HeuristicHint` (per step, highest priority)
2. Common hints for well-known patterns
3. Static `IntentHeuristics` (user-injected)
4. `TaskCategory` defaults (lowest priority)

---

## Tracing

Enable tracing for Predicate Studio visualization:

```python
from predicate.tracing import Tracer

tracer = Tracer(output_dir="./traces")

agent = PlannerExecutorAgent(
    planner=planner,
    executor=executor,
    tracer=tracer,
    config=PlannerExecutorConfig(
        trace_screenshots=True,
        trace_screenshot_format="jpeg",
        trace_screenshot_quality=80,
    ),
)

# Run automation
result = await agent.run(runtime, task)

# Trace files saved to ./traces/
```

---

## Examples

### E-commerce Purchase Flow

```python
from predicate.agents import (
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    AutomationTask,
    TaskCategory,
)
from predicate.agents.browser_agent import CaptchaConfig
from predicate.captcha_strategies import HumanHandoffSolver
from predicate.llm_provider import OpenAIProvider

# Setup
planner = OpenAIProvider(model="gpt-4o")
executor = OpenAIProvider(model="gpt-4o-mini")

config = PlannerExecutorConfig(
    captcha=CaptchaConfig(
        policy="callback",
        handler=HumanHandoffSolver(timeout_ms=120_000),
    ),
)

agent = PlannerExecutorAgent(
    planner=planner,
    executor=executor,
    config=config,
)

task = AutomationTask(
    task_id="buy-laptop",
    starting_url="https://amazon.com",
    task="Search for 'laptop under $500', add the first result to cart, proceed to checkout",
    category=TaskCategory.TRANSACTION,
    enable_recovery=True,
    max_recovery_attempts=2,
)

task = task.with_success_criteria(
    {"predicate": "url_contains", "args": ["/cart"]},
    {"predicate": "exists", "args": [".cart-item"]},
)

async with AsyncPredicateBrowser() as browser:
    runtime = AgentRuntime.from_browser(browser)
    result = await agent.run(runtime, task)
    print(f"Success: {result.success}, Steps: {result.steps_completed}/{result.steps_total}")
```

### Form Filling with Extraction

```python
from predicate.agents import AutomationTask, TaskCategory, ExtractionSpec

task = AutomationTask(
    task_id="fill-contact-form",
    starting_url="https://example.com/contact",
    task="Fill the contact form with name 'John Doe', email 'john@example.com', and message 'Hello'",
    category=TaskCategory.FORM_FILL,
).with_success_criteria(
    {"predicate": "exists", "args": [".success-message"]},
)

result = await agent.run(runtime, task)
```

### Data Extraction

```python
task = AutomationTask(
    task_id="extract-prices",
    starting_url="https://shop.com/products",
    task="Extract all product names and prices from the product listing",
    category=TaskCategory.EXTRACTION,
    extraction_spec=ExtractionSpec(
        output_schema={
            "products": [{"name": "str", "price": "float"}]
        },
        format="json",
    ),
)

result = await agent.run(runtime, task)
```

---

## API Reference

### PlannerExecutorAgent

```python
class PlannerExecutorAgent:
    def __init__(
        self,
        *,
        planner: LLMProvider,           # LLM for generating plans
        executor: LLMProvider,          # LLM for executing steps
        vision_executor: LLMProvider | None = None,
        vision_verifier: LLMProvider | None = None,
        config: PlannerExecutorConfig | None = None,
        tracer: Tracer | None = None,
        context_formatter: Callable | None = None,
        intent_heuristics: IntentHeuristics | None = None,
        executor_override: ExecutorOverride | None = None,
    )

    async def run(
        self,
        runtime: AgentRuntime,
        task: AutomationTask | str,
        *,
        start_url: str | None = None,
        run_id: str | None = None,
    ) -> RunOutcome

    async def plan(
        self,
        task: str,
        *,
        start_url: str | None = None,
        max_attempts: int = 2,
    ) -> Plan

    async def step(
        self,
        runtime: AgentRuntime,
        step: PlanStep,
        step_index: int = 0,
    ) -> StepOutcome
```

### RunOutcome

```python
@dataclass
class RunOutcome:
    run_id: str
    task: str
    success: bool
    steps_completed: int
    steps_total: int
    replans_used: int
    step_outcomes: list[StepOutcome]
    total_duration_ms: int
    error: str | None
```

### StepOutcome

```python
@dataclass
class StepOutcome:
    step_id: int
    goal: str
    status: StepStatus  # SUCCESS, FAILED, SKIPPED, VISION_FALLBACK
    action_taken: str | None
    verification_passed: bool
    used_vision: bool
    error: str | None
    duration_ms: int
    url_before: str | None
    url_after: str | None
```
