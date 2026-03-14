# Planner + Executor Agent Examples

This directory contains examples for the `PlannerExecutorAgent`, a two-tier agent
architecture with separate Planner (7B+) and Executor (3B-7B) models.

## Examples

| File | Description |
|------|-------------|
| `minimal_example.py` | Basic usage with OpenAI models |
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
