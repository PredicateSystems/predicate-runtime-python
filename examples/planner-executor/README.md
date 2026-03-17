# Planner + Executor Agent Examples

This directory contains examples for the `PlannerExecutorAgent`, a two-tier agent
architecture with separate Planner (7B+) and Executor (3B-7B) models.

> **See also**: [Full User Manual](../../docs/PLANNER_EXECUTOR_AGENT.md) for comprehensive documentation.

## Examples

| File | Description |
|------|-------------|
| `minimal_example.py` | Basic usage with OpenAI models |
| `stepwise_example.py` | Stepwise (ReAct-style) planning for unfamiliar sites |
| `automation_task_example.py` | Using AutomationTask for flexible task definition |
| `captcha_example.py` | CAPTCHA handling with different solvers |
| `local_models_example.py` | Using local HuggingFace/MLX models |
| `custom_config_example.py` | Custom configuration (escalation, retry, vision) |
| `tracing_example.py` | Full tracing integration for Predicate Studio |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PlannerExecutorAgent                      │
├─────────────────────────────────────────────────────────────┤
│  Planner (7B+)              │  Executor (3B-7B)             │
│  ─────────────              │  ────────────────             │
│  • Generates JSON plan      │  • Executes each step         │
│  • Includes predicates      │  • Snapshot-first approach    │
│  • Handles replanning       │  • Vision fallback            │
│  • Stepwise (ReAct) mode    │                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      AgentRuntime                            │
│  • Snapshots with limit escalation                          │
│  • Predicate verification                                    │
│  • Tracing for Studio visualization                         │
└─────────────────────────────────────────────────────────────┘
```

## Planning Modes

### Upfront Planning (Default)

The planner generates a complete multi-step plan before execution. Use for well-known sites.

```python
result = await agent.run(runtime, task)
```

### Stepwise Planning (ReAct-style)

The planner decides one action at a time based on current page state. **Recommended for unfamiliar sites.**

```python
from predicate.agents import StepwisePlanningConfig

config = PlannerExecutorConfig(
    stepwise=StepwisePlanningConfig(
        max_steps=30,
        action_history_limit=5,
    ),
)

agent = PlannerExecutorAgent(planner=planner, executor=executor, config=config)
result = await agent.run_stepwise(runtime, task)
```

### Auto-Fallback (Default Behavior)

By default, `agent.run()` automatically falls back to stepwise planning when upfront planning fails:

```python
# Default: auto_fallback_to_stepwise=True
result = await agent.run(runtime, task)

# Check if fallback was used
if result.fallback_used:
    print("Automatically switched to stepwise planning")

# Disable auto-fallback
config = PlannerExecutorConfig(auto_fallback_to_stepwise=False)
```

## Quick Start

```python
from predicate.agents import PlannerExecutorAgent, PlannerExecutorConfig
from predicate.llm_provider import OpenAIProvider
from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime

# Create LLM providers
planner = OpenAIProvider(model="gpt-4o")
executor = OpenAIProvider(model="gpt-4o-mini")

# Create agent
agent = PlannerExecutorAgent(
    planner=planner,
    executor=executor,
)

# Run task
async with AsyncPredicateBrowser() as browser:
    page = await browser.new_page()
    await page.goto("https://example.com")

    runtime = AgentRuntime.from_page(page)
    result = await agent.run(
        runtime=runtime,
        task="Find the main heading on this page",
    )
    print(f"Success: {result.success}")
```

## Configuration

### Snapshot Escalation

Control how the agent increases snapshot limits when elements are missing:

```python
from predicate.agents import SnapshotEscalationConfig

# Default: 60 -> 90 -> 120 -> 150 -> 180 -> 200
config = PlannerExecutorConfig()

# Disable escalation (always use 60)
config = PlannerExecutorConfig(
    snapshot=SnapshotEscalationConfig(enabled=False)
)

# Custom step size: 60 -> 110 -> 160 -> 200
config = PlannerExecutorConfig(
    snapshot=SnapshotEscalationConfig(limit_step=50)
)
```

### Retry Configuration

```python
from predicate.agents import RetryConfig

config = PlannerExecutorConfig(
    retry=RetryConfig(
        verify_timeout_s=15.0,      # Verification timeout
        verify_max_attempts=8,       # Max verification attempts
        max_replans=2,               # Max replanning attempts
    )
)
```

### Vision Fallback

```python
from predicate.agents.browser_agent import VisionFallbackConfig

config = PlannerExecutorConfig(
    vision=VisionFallbackConfig(
        enabled=True,
        max_vision_calls=5,
    )
)
```

## Tracing for Predicate Studio

To visualize agent runs in Predicate Studio:

```python
from predicate.tracer_factory import create_tracer

tracer = create_tracer(
    api_key="sk_...",
    upload_trace=True,
    goal="Search and add to cart",
    agent_type="PlannerExecutorAgent",
)

agent = PlannerExecutorAgent(
    planner=planner,
    executor=executor,
    tracer=tracer,  # Pass tracer for visualization
)

# ... run agent ...

tracer.close()  # Upload trace to Studio
```

## AutomationTask

Use `AutomationTask` for flexible task definition with built-in recovery:

```python
from predicate.agents import AutomationTask, TaskCategory

# Basic task
task = AutomationTask(
    task_id="search-products",
    starting_url="https://amazon.com",
    task="Search for laptops and add the first result to cart",
    category=TaskCategory.TRANSACTION,
    enable_recovery=True,
)

# Add success criteria
task = task.with_success_criteria(
    {"predicate": "url_contains", "args": ["/cart"]},
    {"predicate": "exists", "args": [".cart-item"]},
)

result = await agent.run(runtime, task)
```

## Permissions

Grant browser permissions to prevent permission dialogs from interrupting automation:

```python
from predicate import AsyncPredicateBrowser

# Grant permissions to avoid "Allow this site to access your location?" dialogs
permission_policy = {
    "auto_grant": [
        "geolocation",      # Store locators, local inventory
        "notifications",    # Push notification prompts
        "clipboard-read",   # Paste coupon codes
        "clipboard-write",  # Copy product info
    ],
    "geolocation": {"latitude": 47.6762, "longitude": -122.2057},  # Mock location
}

async with AsyncPredicateBrowser(
    permission_policy=permission_policy,
) as browser:
    # Run automation without permission dialogs
    ...
```

## CAPTCHA Handling

Configure CAPTCHA solving with different strategies:

```python
from predicate.agents.browser_agent import CaptchaConfig
from predicate.captcha_strategies import HumanHandoffSolver, ExternalSolver

# Human handoff: wait for manual solve
config = PlannerExecutorConfig(
    captcha=CaptchaConfig(
        policy="callback",
        handler=HumanHandoffSolver(timeout_ms=120_000),
    ),
)

# External solver: integrate with 2Captcha, CapSolver, etc.
def solve_captcha(ctx):
    # Call your CAPTCHA solving service
    pass

config = PlannerExecutorConfig(
    captcha=CaptchaConfig(
        policy="callback",
        handler=ExternalSolver(resolver=solve_captcha),
    ),
)
```

## Modal/Drawer Dismissal

Automatic modal and drawer dismissal is enabled by default in both upfront and stepwise planning modes.

After successful CLICK actions, the agent automatically detects and dismisses blocking overlays:

```python
from predicate.agents import PlannerExecutorConfig, ModalDismissalConfig

# Default: enabled with common patterns (works in both modes)
config = PlannerExecutorConfig()

# Custom patterns for non-English sites
config = PlannerExecutorConfig(
    modal=ModalDismissalConfig(
        dismiss_patterns=(
            "no thanks", "not now", "close", "skip",  # English
            "nein danke", "schließen",  # German
            "no gracias", "cerrar",  # Spanish
        ),
    ),
)

# Disable modal dismissal
config = PlannerExecutorConfig(
    modal=ModalDismissalConfig(enabled=False),
)
```

This handles common e-commerce scenarios like:
- Amazon's "Add Protection Plan" drawer after Add to Cart
- Cookie consent banners
- Newsletter signup popups
- Promotional overlays
