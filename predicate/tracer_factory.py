"""
Tracer factory with automatic tier detection.

Provides convenient factory function for creating tracers with cloud upload support.

Key Features:
- Automatic cloud upload when API key is provided
- Auto-close on process exit (atexit) to prevent data loss
- Context manager support for both sync and async workflows
- Orphaned trace recovery from previous crashes
"""

import atexit
import gzip
import os
import uuid
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import requests

from predicate.cloud_tracing import CloudTraceSink, SentienceLogger
from predicate.constants import PREDICATE_API_URL
from predicate.tracing import JsonlTraceSink, Tracer

# Global registry of active tracers for atexit cleanup
# Using a set of tracer IDs mapped to weak references
_active_tracers: dict[int, weakref.ref[Tracer]] = {}
_atexit_registered = False


def _cleanup_tracers_on_exit() -> None:
    """
    Cleanup handler called on process exit.

    Closes all active tracers to ensure trace data is uploaded to cloud.
    This prevents data loss when users forget to call tracer.close().
    """
    for tracer_id, tracer_ref in list(_active_tracers.items()):
        tracer = tracer_ref()
        if tracer is not None:
            try:
                tracer.close()
            except Exception:
                pass  # Best effort - don't raise during exit


def _register_tracer_for_cleanup(tracer: Tracer) -> None:
    """
    Register a tracer for automatic cleanup on process exit.

    Args:
        tracer: Tracer instance to register
    """
    global _atexit_registered

    # Use id() as key to avoid hashability issues
    tracer_id = id(tracer)
    _active_tracers[tracer_id] = weakref.ref(tracer)

    # Set callback on tracer so it unregisters itself when closed
    tracer._on_close_callback = _unregister_tracer

    # Register atexit handler on first tracer creation
    if not _atexit_registered:
        atexit.register(_cleanup_tracers_on_exit)
        _atexit_registered = True


def _unregister_tracer(tracer: Tracer) -> None:
    """
    Unregister a tracer from cleanup (called when tracer.close() is invoked).

    Args:
        tracer: Tracer instance to unregister
    """
    tracer_id = id(tracer)
    _active_tracers.pop(tracer_id, None)


def _emit_run_start(
    tracer: Tracer,
    agent_type: str | None,
    llm_model: str | None,
    goal: str | None,
    start_url: str | None,
) -> None:
    """
    Helper to emit run_start event with available metadata.
    """
    try:
        config: dict[str, Any] = {}
        if goal:
            config["goal"] = goal
        if start_url:
            config["start_url"] = start_url

        tracer.emit_run_start(
            agent=agent_type or "SentienceAgent",
            llm_model=llm_model,
            config=config if config else None,
        )
    except Exception:
        pass  # Tracing must be non-fatal


def create_tracer(
    api_key: str | None = None,
    run_id: str | None = None,
    api_url: str | None = None,
    logger: SentienceLogger | None = None,
    upload_trace: bool | None = None,
    goal: str | None = None,
    agent_type: str | None = None,
    llm_model: str | None = None,
    start_url: str | None = None,
    screenshot_processor: Callable[[str], str] | None = None,
    auto_emit_run_start: bool = True,
) -> Tracer:
    """
    Create tracer with automatic tier detection and auto-cleanup.

    Tier Detection:
    - If api_key is provided: Try to initialize CloudTraceSink (Pro/Enterprise)
    - If cloud init fails or no api_key: Fall back to JsonlTraceSink (Free tier)

    Auto-Cleanup:
    - Tracers are automatically registered for cleanup on process exit (atexit)
    - This ensures trace data is uploaded even if tracer.close() is not called
    - For best practice, still call tracer.close() explicitly or use context manager

    Args:
        api_key: Sentience API key (e.g., "sk_pro_xxxxx")
                 - Free tier: None or empty
                 - Pro/Enterprise: Valid API key
        run_id: Unique identifier for this agent run. If not provided, generates UUID.
        api_url: Sentience API base URL (default: https://api.sentienceapi.com)
        logger: Optional logger instance for logging file sizes and errors
        upload_trace: Enable cloud trace upload. When None (default), automatically
                      enables cloud upload if api_key is provided. When True and api_key
                      is provided, traces will be uploaded to cloud. When False, traces
                      are saved locally only regardless of api_key.
        goal: User's goal/objective for this trace run. This will be displayed as the
              trace name in the frontend. Should be descriptive and action-oriented.
              Example: "Add wireless headphones to cart on Amazon"
        agent_type: Type of agent running (e.g., "SentienceAgent", "CustomAgent")
        llm_model: LLM model used (e.g., "gpt-4-turbo", "claude-3-5-sonnet")
        start_url: Starting URL of the agent run (e.g., "https://amazon.com")
        screenshot_processor: Optional function to process screenshots before upload.
                            Takes base64 string, returns processed base64 string.
                            Useful for PII redaction or custom image processing.
        auto_emit_run_start: If True (default), automatically emit run_start event
                            with the provided metadata. This ensures traces have
                            complete structure for Studio visualization.

    Returns:
        Tracer configured with appropriate sink

    Example:
        >>> # RECOMMENDED: Use as context manager (auto-closes on exit)
        >>> with create_tracer(api_key="sk_pro_xyz", goal="Add to cart") as tracer:
        ...     agent = SentienceAgent(browser, llm, tracer=tracer)
        ...     agent.act("Click search")
        >>> # tracer.close() called automatically
        >>>
        >>> # ALTERNATIVE: Manual close (still safe - atexit cleanup as fallback)
        >>> tracer = create_tracer(api_key="sk_pro_xyz", goal="Add to cart")
        >>> try:
        ...     agent = SentienceAgent(browser, llm, tracer=tracer)
        ...     agent.act("Click search")
        ... finally:
        ...     tracer.close()  # Best practice: explicit close
        >>>
        >>> # Pro tier with all metadata
        >>> tracer = create_tracer(
        ...     api_key="sk_pro_xyz",
        ...     run_id="demo",
        ...     goal="Add headphones to cart",
        ...     agent_type="SentienceAgent",
        ...     llm_model="gpt-4-turbo",
        ...     start_url="https://amazon.com"
        ... )
        >>>
        >>> # With screenshot processor for PII redaction
        >>> def redact_pii(screenshot_base64: str) -> str:
        ...     # Your custom redaction logic
        ...     return redacted_screenshot
        >>>
        >>> tracer = create_tracer(
        ...     api_key="sk_pro_xyz",
        ...     screenshot_processor=redact_pii
        ... )
        >>>
        >>> # Free tier user (local-only traces)
        >>> tracer = create_tracer(run_id="demo")
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    if api_url is None:
        api_url = PREDICATE_API_URL

    # Default upload_trace to True when api_key is provided
    # This ensures tracing is enabled automatically for Pro/Enterprise tiers
    should_upload = upload_trace if upload_trace is not None else (api_key is not None)

    # 0. Check for orphaned traces from previous crashes (if api_key provided and upload enabled)
    if api_key and should_upload:
        _recover_orphaned_traces(api_key, api_url)

    # 1. Try to initialize Cloud Sink (Pro/Enterprise tier) if upload enabled
    if api_key and should_upload:
        try:
            # Build metadata object for trace initialization
            # Only include non-empty fields to avoid sending empty strings
            metadata: dict[str, str] = {}
            if goal and goal.strip():
                metadata["goal"] = goal.strip()
            if agent_type and agent_type.strip():
                metadata["agent_type"] = agent_type.strip()
            if llm_model and llm_model.strip():
                metadata["llm_model"] = llm_model.strip()
            if start_url and start_url.strip():
                metadata["start_url"] = start_url.strip()

            # Build request payload
            payload: dict[str, Any] = {"run_id": run_id}
            if metadata:
                payload["metadata"] = metadata

            # Request pre-signed upload URL from backend
            response = requests.post(
                f"{api_url}/v1/traces/init",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                upload_url = data.get("upload_url")

                if upload_url:
                    print("☁️  [Sentience] Cloud tracing enabled (Pro tier)")
                    tracer = Tracer(
                        run_id=run_id,
                        sink=CloudTraceSink(
                            upload_url=upload_url,
                            run_id=run_id,
                            api_key=api_key,
                            api_url=api_url,
                            logger=logger,
                        ),
                        screenshot_processor=screenshot_processor,
                    )
                    # Register for atexit cleanup (safety net for forgotten close())
                    _register_tracer_for_cleanup(tracer)
                    # Auto-emit run_start for complete trace structure
                    if auto_emit_run_start:
                        _emit_run_start(tracer, agent_type, llm_model, goal, start_url)
                    return tracer
                else:
                    print("⚠️  [Sentience] Cloud init response missing upload_url")
                    print(f"   Response data: {data}")
                    print("   Falling back to local-only tracing")

            elif response.status_code == 403:
                print("⚠️  [Sentience] Cloud tracing requires Pro tier")
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error") or error_data.get("message", "")
                    if error_msg:
                        print(f"   API Error: {error_msg}")
                except Exception:
                    pass
                print("   Falling back to local-only tracing")
            elif response.status_code == 401:
                print("⚠️  [Sentience] Cloud init failed: HTTP 401 Unauthorized")
                print("   API key is invalid or expired")
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error") or error_data.get("message", "")
                    if error_msg:
                        print(f"   API Error: {error_msg}")
                except Exception:
                    pass
                print("   Falling back to local-only tracing")
            else:
                print(f"⚠️  [Sentience] Cloud init failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error") or error_data.get(
                        "message", "Unknown error"
                    )
                    print(f"   Error: {error_msg}")
                    if "tier" in error_msg.lower() or "subscription" in error_msg.lower():
                        print(f"   💡 This may be a tier/subscription issue")
                except Exception:
                    print(f"   Response: {response.text[:200]}")
                print("   Falling back to local-only tracing")

        except requests.exceptions.Timeout:
            print("⚠️  [Sentience] Cloud init timeout")
            print("   Falling back to local-only tracing")
        except requests.exceptions.ConnectionError:
            print("⚠️  [Sentience] Cloud init connection error")
            print("   Falling back to local-only tracing")
        except Exception as e:
            print(f"⚠️  [Sentience] Cloud init error: {e}")
            print("   Falling back to local-only tracing")

    # 2. Fallback to Local Sink (Free tier / Offline mode)
    traces_dir = Path("traces")
    traces_dir.mkdir(exist_ok=True)

    local_path = traces_dir / f"{run_id}.jsonl"
    print(f"💾 [Sentience] Local tracing: {local_path}")

    tracer = Tracer(
        run_id=run_id,
        sink=JsonlTraceSink(str(local_path)),
        screenshot_processor=screenshot_processor,
    )

    # Register for atexit cleanup (ensures file is properly closed)
    _register_tracer_for_cleanup(tracer)

    # Auto-emit run_start for complete trace structure
    if auto_emit_run_start:
        _emit_run_start(tracer, agent_type, llm_model, goal, start_url)

    return tracer


def _recover_orphaned_traces(api_key: str, api_url: str = PREDICATE_API_URL) -> None:
    """
    Attempt to upload orphaned traces from previous crashed runs.

    Scans ~/.sentience/traces/pending/ for un-uploaded trace files and
    attempts to upload them using the provided API key.

    Args:
        api_key: Sentience API key for authentication
        api_url: Sentience API base URL (defaults to PREDICATE_API_URL)
    """
    pending_dir = Path.home() / ".sentience" / "traces" / "pending"

    if not pending_dir.exists():
        return

    orphaned = list(pending_dir.glob("*.jsonl"))

    if not orphaned:
        return

    # Filter out test files (run_ids that start with "test-" or are clearly test data)
    # These are likely from local testing and shouldn't be uploaded
    test_patterns = ["test-", "test_", "test."]
    valid_orphaned = [
        f
        for f in orphaned
        if not any(f.stem.startswith(pattern) for pattern in test_patterns)
        and not f.stem.startswith("test")
    ]

    if not valid_orphaned:
        return

    print(f"⚠️  [Sentience] Found {len(valid_orphaned)} un-uploaded trace(s) from previous runs")
    print("   Attempting to upload now...")

    for trace_file in valid_orphaned:
        try:
            # Extract run_id from filename (format: {run_id}.jsonl)
            run_id = trace_file.stem

            # Request new upload URL for this run_id
            response = requests.post(
                f"{api_url}/v1/traces/init",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"run_id": run_id},
                timeout=10,
            )

            if response.status_code != 200:
                # HTTP 409 means trace already exists (already uploaded)
                # Treat as success and delete local file
                if response.status_code == 409:
                    print(f"✅ Trace {run_id} already exists in cloud (skipping re-upload)")
                    # Delete local file since it's already in cloud
                    try:
                        os.remove(trace_file)
                    except Exception:
                        pass  # Ignore cleanup errors
                    continue
                # HTTP 422 typically means invalid run_id (e.g., test files)
                # Skip silently for 422, but log other errors
                if response.status_code == 422:
                    # Likely a test file or invalid run_id, skip silently
                    continue
                print(f"❌ Failed to get upload URL for {run_id}: HTTP {response.status_code}")
                continue

            data = response.json()
            upload_url = data.get("upload_url")

            if not upload_url:
                print(f"❌ Upload URL missing for {run_id}")
                continue

            # Read and compress trace file
            with open(trace_file, "rb") as f:
                trace_data = f.read()

            compressed_data = gzip.compress(trace_data)

            # Upload to cloud
            upload_response = requests.put(
                upload_url,
                data=compressed_data,
                headers={
                    "Content-Type": "application/x-gzip",
                    "Content-Encoding": "gzip",
                },
                timeout=60,
            )

            if upload_response.status_code == 200:
                print(f"✅ Uploaded orphaned trace: {run_id}")
                # Delete file on successful upload
                try:
                    os.remove(trace_file)
                except Exception:
                    pass  # Ignore cleanup errors
            else:
                print(f"❌ Failed to upload {run_id}: HTTP {upload_response.status_code}")

        except requests.exceptions.Timeout:
            print(f"❌ Timeout uploading {trace_file.name}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error uploading {trace_file.name}")
        except Exception as e:
            print(f"❌ Error uploading {trace_file.name}: {e}")
