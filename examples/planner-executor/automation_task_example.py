"""
AutomationTask Example

Demonstrates using AutomationTask for flexible task definition with:
- Task categories for better heuristics
- Success criteria for verification
- Recovery configuration for rollback
- Extraction specification for data extraction tasks

Prerequisites:
    pip install predicate-sdk openai
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime
from predicate.agents import (
    AutomationTask,
    ExtractionSpec,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    TaskCategory,
)
from predicate.llm_provider import OpenAIProvider


async def basic_task_example():
    """Basic AutomationTask usage."""
    print("\n=== Basic AutomationTask Example ===\n")

    # Create LLM providers
    planner = OpenAIProvider(model="gpt-4o")
    executor = OpenAIProvider(model="gpt-4o-mini")

    # Create agent
    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
    )

    # Create a basic task
    task = AutomationTask(
        task_id="search-example",
        starting_url="https://example.com",
        task="Find the main heading on the page",
    )

    print(f"Task ID: {task.task_id}")
    print(f"Starting URL: {task.starting_url}")
    print(f"Task: {task.task}")

    async with AsyncPredicateBrowser() as browser:
        page = await browser.new_page()
        await page.goto(task.starting_url)

        runtime = AgentRuntime.from_page(page)
        result = await agent.run(runtime, task)

        print(f"\nResult: {'Success' if result.success else 'Failed'}")
        print(f"Steps completed: {result.steps_completed}/{result.steps_total}")


async def transaction_task_example():
    """E-commerce transaction task with recovery."""
    print("\n=== Transaction Task Example ===\n")

    planner = OpenAIProvider(model="gpt-4o")
    executor = OpenAIProvider(model="gpt-4o-mini")

    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
    )

    # Create a transaction task with category and recovery
    task = AutomationTask(
        task_id="purchase-laptop",
        starting_url="https://amazon.com",
        task="Search for 'laptop under $500' and add the first result to cart",
        category=TaskCategory.TRANSACTION,  # Helps with element selection
        enable_recovery=True,  # Enable rollback on failure
        max_recovery_attempts=2,
        max_steps=50,
    )

    # Add success criteria
    task = task.with_success_criteria(
        {"predicate": "url_contains", "args": ["/cart"]},
        {"predicate": "exists", "args": [".cart-item, .sc-list-item"]},
    )

    print(f"Task: {task.task}")
    print(f"Category: {task.category}")
    print(f"Recovery enabled: {task.enable_recovery}")
    print(f"Success criteria: {task.success_criteria}")

    async with AsyncPredicateBrowser() as browser:
        page = await browser.new_page()
        await page.goto(task.starting_url)

        runtime = AgentRuntime.from_page(page)
        result = await agent.run(runtime, task)

        print(f"\nResult: {'Success' if result.success else 'Failed'}")
        print(f"Steps completed: {result.steps_completed}/{result.steps_total}")
        print(f"Replans used: {result.replans_used}")

        if result.error:
            print(f"Error: {result.error}")


async def extraction_task_example():
    """Data extraction task with schema."""
    print("\n=== Extraction Task Example ===\n")

    planner = OpenAIProvider(model="gpt-4o")
    executor = OpenAIProvider(model="gpt-4o-mini")

    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
    )

    # Create an extraction task with output schema
    task = AutomationTask(
        task_id="extract-product-info",
        starting_url="https://amazon.com/dp/B0EXAMPLE",
        task="Extract the product name, price, and rating",
        category=TaskCategory.EXTRACTION,
        extraction_spec=ExtractionSpec(
            output_schema={
                "name": "str",
                "price": "float",
                "rating": "float",
                "num_reviews": "int",
            },
            format="json",
            require_evidence=True,
        ),
    )

    print(f"Task: {task.task}")
    print(f"Category: {task.category}")
    print(f"Output schema: {task.extraction_spec.output_schema}")
    print(f"Format: {task.extraction_spec.format}")

    # Note: This example won't run successfully as the URL is fake
    # In real usage, provide a valid product URL


async def form_fill_task_example():
    """Form filling task example."""
    print("\n=== Form Fill Task Example ===\n")

    planner = OpenAIProvider(model="gpt-4o")
    executor = OpenAIProvider(model="gpt-4o-mini")

    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
    )

    # Create a form fill task
    task = AutomationTask(
        task_id="contact-form",
        starting_url="https://example.com/contact",
        task="Fill the contact form with name 'John Doe', email 'john@example.com', and message 'Hello, I have a question'",
        category=TaskCategory.FORM_FILL,
    )

    # Add success criteria for form submission
    task = task.with_success_criteria(
        {"predicate": "any_of", "args": [
            {"predicate": "exists", "args": [".success-message"]},
            {"predicate": "url_contains", "args": ["/thank-you"]},
        ]},
    )

    print(f"Task: {task.task}")
    print(f"Category: {task.category}")


async def from_string_example():
    """Create task from simple string."""
    print("\n=== From String Example ===\n")

    # Quick task creation from string
    task = AutomationTask.from_string(
        "Search for 'headphones' and filter by price under $50",
        "https://amazon.com",
        category=TaskCategory.SEARCH,
    )

    print(f"Task ID: {task.task_id}")  # Auto-generated UUID
    print(f"Task: {task.task}")
    print(f"Starting URL: {task.starting_url}")
    print(f"Category: {task.category}")


async def with_extraction_example():
    """Add extraction to existing task."""
    print("\n=== With Extraction Example ===\n")

    # Create basic task
    task = AutomationTask(
        task_id="product-search",
        starting_url="https://amazon.com",
        task="Search for the cheapest laptop",
    )

    # Add extraction specification using fluent API
    task_with_extraction = task.with_extraction(
        output_schema={"product_name": "str", "price": "float"},
        format="json",
    )

    print(f"Original category: {task.category}")
    print(f"After with_extraction: {task_with_extraction.category}")
    print(f"Extraction spec: {task_with_extraction.extraction_spec}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("AutomationTask Examples")
    print("=" * 60)

    # Show task creation patterns (no browser needed)
    await from_string_example()
    await with_extraction_example()

    # These require browser and API keys
    # Uncomment to run:
    # await basic_task_example()
    # await transaction_task_example()
    # await form_fill_task_example()


if __name__ == "__main__":
    asyncio.run(main())
