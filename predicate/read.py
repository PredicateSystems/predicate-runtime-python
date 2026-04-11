"""
Read page content - supports raw HTML, text, and markdown formats
"""

import json
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from .browser import AsyncSentienceBrowser, SentienceBrowser
from .browser_evaluator import BrowserEvaluator
from .llm_provider import LLMProvider
from .models import ExtractResult, ReadResult

_READ_EVAL_JS = r"""
(options) => {
  const fmt = (options && options.format) ? options.format : "raw";
  try {
    const api =
      (typeof globalThis !== "undefined" && globalThis && globalThis.sentience)
        ? globalThis.sentience
        : (typeof window !== "undefined" && window && window.sentience)
          ? window.sentience
          : null;

    if (!api || typeof api.read !== "function") {
      return {
        status: "error",
        url: (typeof location !== "undefined" && location && location.href) ? location.href : "",
        format: fmt,
        content: "",
        length: 0,
        error: "sentience extension not available (window.sentience.read missing)"
      };
    }

    const res = api.read(options || { format: fmt });
    if (!res || typeof res !== "object") {
      return {
        status: "error",
        url: (typeof location !== "undefined" && location && location.href) ? location.href : "",
        format: fmt,
        content: "",
        length: 0,
        error: "predicate.read returned non-object"
      };
    }

    // Normalize to the ReadResult schema expected by SDK consumers.
    // If the extension returns an error without an explicit status, treat it as error.
    if (!res.status) res.status = (res.error ? "error" : "success");
    if (!res.url) res.url = (typeof location !== "undefined" && location && location.href) ? location.href : "";
    if (!res.format) res.format = fmt;
    if (typeof res.content !== "string") res.content = String(res.content ?? "");
    if (typeof res.length !== "number") res.length = res.content.length;
    if (!("error" in res)) res.error = null;
    return res;
  } catch (e) {
    const msg =
      (e && (e.stack || e.message)) ? (e.stack || e.message) : String(e);
    return {
      status: "error",
      url: (typeof location !== "undefined" && location && location.href) ? location.href : "",
      format: fmt,
      content: "",
      length: 0,
      error: msg
    };
  }
}
"""


def _looks_empty_content(content: str) -> bool:
    # Some pages can legitimately be short, but for "read" the empty/near-empty
    # case is almost always an integration failure (extension returned ""/"\n"/" ").
    if content is None:
        return True
    if not isinstance(content, str):
        content = str(content)
    return len(content.strip()) == 0


def _debug_read(msg: str) -> None:
    if os.environ.get("SENTIENCE_DEBUG_READ", "").strip():
        print(f"[sentience][read] {msg}")


def _fallback_read_from_page_sync(
    page,
    *,
    output_format: Literal["raw", "text", "markdown"],
) -> ReadResult | None:
    """
    Fallback reader that does NOT rely on the extension.
    Uses Playwright primitives directly.
    """
    try:
        url = getattr(page, "url", "") or ""
        if output_format == "raw":
            html = page.content()
            if not isinstance(html, str) or _looks_empty_content(html):
                return None
            return ReadResult(
                status="success", url=url, format="raw", content=html, length=len(html)
            )

        if output_format == "text":
            text = page.evaluate(
                "() => (document && document.body) ? (document.body.innerText || '') : ''"
            )
            if not isinstance(text, str) or _looks_empty_content(text):
                return None
            return ReadResult(
                status="success", url=url, format="text", content=text, length=len(text)
            )

        if output_format == "markdown":
            try:
                from markdownify import markdownify  # type: ignore
            except Exception:
                return None
            html = page.content()
            if not isinstance(html, str) or _looks_empty_content(html):
                return None
            md = markdownify(html, heading_style="ATX", wrap=True)
            if not isinstance(md, str) or _looks_empty_content(md):
                return None
            return ReadResult(
                status="success", url=url, format="markdown", content=md, length=len(md)
            )
    except Exception:
        return None
    return None


async def _fallback_read_from_page_async(
    page,
    *,
    output_format: Literal["raw", "text", "markdown"],
) -> ReadResult | None:
    """
    Async variant of `_fallback_read_from_page_sync`.
    """
    try:
        url = getattr(page, "url", "") or ""
        if output_format == "raw":
            html = await page.content()
            if not isinstance(html, str) or _looks_empty_content(html):
                return None
            return ReadResult(
                status="success", url=url, format="raw", content=html, length=len(html)
            )

        if output_format == "text":
            text = await page.evaluate(
                "() => (document && document.body) ? (document.body.innerText || '') : ''"
            )
            if not isinstance(text, str) or _looks_empty_content(text):
                return None
            return ReadResult(
                status="success", url=url, format="text", content=text, length=len(text)
            )

        if output_format == "markdown":
            try:
                from markdownify import markdownify  # type: ignore
            except Exception:
                return None
            html = await page.content()
            if not isinstance(html, str) or _looks_empty_content(html):
                return None
            md = markdownify(html, heading_style="ATX", wrap=True)
            if not isinstance(md, str) or _looks_empty_content(md):
                return None
            return ReadResult(
                status="success", url=url, format="markdown", content=md, length=len(md)
            )
    except Exception:
        return None
    return None


def read(
    browser: SentienceBrowser,
    output_format: Literal["raw", "text", "markdown"] = "raw",
    enhance_markdown: bool = True,
) -> ReadResult:
    """
    Read page content as raw HTML, text, or markdown

    Args:
        browser: SentienceBrowser instance
        output_format: Output format - "raw" (default, returns HTML for external processing),
                        "text" (plain text), or "markdown" (lightweight or enhanced markdown).
        enhance_markdown: If True and output_format is "markdown", uses markdownify for better conversion.
                          If False, uses the extension's lightweight markdown converter.

    Returns:
        dict with:
            - status: "success" or "error"
            - url: Current page URL
            - format: "raw", "text", or "markdown"
            - content: Page content as string
            - length: Content length in characters
            - error: Error message if status is "error"

    Examples:
        # Get raw HTML (default) - can be used with markdownify for better conversion
        result = read(browser)
        html_content = result["content"]

        # Get high-quality markdown (uses markdownify internally)
        result = read(browser, output_format="markdown")
        markdown = result["content"]

        # Get plain text
        result = read(browser, output_format="text")
        text = result["content"]
    """
    if not browser.page:
        raise RuntimeError("Browser not started. Call browser.start() first.")

    # Best-effort: wait for extension injection, like snapshot/text_search do.
    # This prevents transient "window.sentience undefined" right after navigation.
    try:
        BrowserEvaluator.wait_for_extension(browser.page, timeout_ms=5000)
    except Exception:
        pass

    if output_format == "markdown" and enhance_markdown:
        # Get raw HTML from the extension first
        raw_html_result = browser.page.evaluate(
            _READ_EVAL_JS,
            {"format": "raw"},
        )

        if raw_html_result.get("status") == "success":
            html_content = raw_html_result["content"]
            try:
                # Use markdownify for enhanced markdown conversion
                from markdownify import markdownify  # type: ignore

                try:
                    # Some markdownify versions don't expose MarkdownifyError.
                    from markdownify import MarkdownifyError  # type: ignore
                except Exception:  # pragma: no cover
                    MarkdownifyError = Exception  # type: ignore[misc,assignment]

                markdown_content = markdownify(html_content, heading_style="ATX", wrap=True)
                if _looks_empty_content(markdown_content):
                    # Extension returned empty/near-empty HTML; try Playwright fallback.
                    fb = _fallback_read_from_page_sync(browser.page, output_format="markdown")
                    if fb is not None:
                        _debug_read("fallback=playwright reason=empty_markdown_from_extension")
                        return fb
                return ReadResult(
                    status="success",
                    url=raw_html_result["url"],
                    format="markdown",
                    content=markdown_content,
                    length=len(markdown_content),
                )
            except ImportError:
                print(
                    "Warning: 'markdownify' not installed. Install with 'pip install markdownify' for enhanced markdown. Falling back to extension's markdown."
                )
            except MarkdownifyError as e:
                print(f"Warning: markdownify failed ({e}), falling back to extension's markdown.")
            except Exception as e:
                print(
                    f"Warning: An unexpected error occurred with markdownify ({e}), falling back to extension's markdown."
                )
        else:
            # Extension raw read failed; try Playwright fallback for markdown if possible.
            fb = _fallback_read_from_page_sync(browser.page, output_format="markdown")
            if fb is not None:
                _debug_read("fallback=playwright reason=extension_raw_failed format=markdown")
                return fb

    # If not enhanced markdown, or fallback, call extension with requested format
    result = browser.page.evaluate(
        _READ_EVAL_JS,
        {"format": output_format},
    )

    # Convert dict result to ReadResult model
    rr = ReadResult(**result)
    if rr.status == "success" and _looks_empty_content(rr.content):
        fb = _fallback_read_from_page_sync(browser.page, output_format=output_format)
        if fb is not None:
            _debug_read(
                f"fallback=playwright reason=empty_content_from_extension format={output_format}"
            )
            return fb
        # If we couldn't fallback, treat near-empty as error so callers don't
        # mistakenly treat it as a successful read.
        return ReadResult(
            status="error",
            url=rr.url,
            format=rr.format,
            content=rr.content,
            length=rr.length,
            error="empty_content",
        )
    return rr


def read_best_effort(
    browser: SentienceBrowser,
    output_format: Literal["raw", "text", "markdown"] = "raw",
    enhance_markdown: bool = True,
) -> ReadResult:
    """
    Best-effort read.

    This function exists to give callers a stable API contract:
    - Prefer the extension-backed `read()` path (when available)
    - If the extension returns `success` but empty/near-empty content, fallback to
      Playwright primitives (page.content()/innerText and HTML→markdownify for markdown)

    Today, `read()` already implements this best-effort behavior; this wrapper
    is intentionally thin so we can extend the fallback chain without changing
    semantics for callers that want explicit "best effort" behavior.
    """
    return read(browser, output_format=output_format, enhance_markdown=enhance_markdown)


async def read_async(
    browser: AsyncSentienceBrowser,
    output_format: Literal["raw", "text", "markdown"] = "raw",
    enhance_markdown: bool = True,
) -> ReadResult:
    """
    Read page content as raw HTML, text, or markdown (async)

    Args:
        browser: AsyncSentienceBrowser instance
        output_format: Output format - "raw" (default, returns HTML for external processing),
                        "text" (plain text), or "markdown" (lightweight or enhanced markdown).
        enhance_markdown: If True and output_format is "markdown", uses markdownify for better conversion.
                          If False, uses the extension's lightweight markdown converter.

    Returns:
        dict with:
            - status: "success" or "error"
            - url: Current page URL
            - format: "raw", "text", or "markdown"
            - content: Page content as string
            - length: Content length in characters
            - error: Error message if status is "error"

    Examples:
        # Get raw HTML (default) - can be used with markdownify for better conversion
        result = await read_async(browser)
        html_content = result["content"]

        # Get high-quality markdown (uses markdownify internally)
        result = await read_async(browser, output_format="markdown")
        markdown = result["content"]

        # Get plain text
        result = await read_async(browser, output_format="text")
        text = result["content"]
    """
    if not browser.page:
        raise RuntimeError("Browser not started. Call await browser.start() first.")

    # Best-effort: wait for extension injection, like snapshot/text_search do.
    try:
        await BrowserEvaluator.wait_for_extension_async(browser.page, timeout_ms=5000)
    except Exception:
        pass

    if output_format == "markdown" and enhance_markdown:
        # Get raw HTML from the extension first
        raw_html_result = await browser.page.evaluate(
            _READ_EVAL_JS,
            {"format": "raw"},
        )

        if raw_html_result.get("status") == "success":
            html_content = raw_html_result["content"]
            try:
                # Use markdownify for enhanced markdown conversion
                from markdownify import markdownify  # type: ignore

                try:
                    from markdownify import MarkdownifyError  # type: ignore
                except Exception:  # pragma: no cover
                    MarkdownifyError = Exception  # type: ignore[misc,assignment]

                markdown_content = markdownify(html_content, heading_style="ATX", wrap=True)
                if _looks_empty_content(markdown_content):
                    fb = await _fallback_read_from_page_async(
                        browser.page, output_format="markdown"
                    )
                    if fb is not None:
                        _debug_read("fallback=playwright reason=empty_markdown_from_extension")
                        return fb
                return ReadResult(
                    status="success",
                    url=raw_html_result["url"],
                    format="markdown",
                    content=markdown_content,
                    length=len(markdown_content),
                )
            except ImportError:
                print(
                    "Warning: 'markdownify' not installed. Install with 'pip install markdownify' for enhanced markdown. Falling back to extension's markdown."
                )
            except MarkdownifyError as e:
                print(f"Warning: markdownify failed ({e}), falling back to extension's markdown.")
            except Exception as e:
                print(
                    f"Warning: An unexpected error occurred with markdownify ({e}), falling back to extension's markdown."
                )
        else:
            fb = await _fallback_read_from_page_async(browser.page, output_format="markdown")
            if fb is not None:
                _debug_read("fallback=playwright reason=extension_raw_failed format=markdown")
                return fb

    # If not enhanced markdown, or fallback, call extension with requested format
    result = await browser.page.evaluate(
        _READ_EVAL_JS,
        {"format": output_format},
    )

    rr = ReadResult(**result)
    if rr.status == "success" and _looks_empty_content(rr.content):
        fb = await _fallback_read_from_page_async(browser.page, output_format=output_format)
        if fb is not None:
            _debug_read(
                f"fallback=playwright reason=empty_content_from_extension format={output_format}"
            )
            return fb
        return ReadResult(
            status="error",
            url=rr.url,
            format=rr.format,
            content=rr.content,
            length=rr.length,
            error="empty_content",
        )
    return rr


async def read_best_effort_async(
    browser: AsyncSentienceBrowser,
    output_format: Literal["raw", "text", "markdown"] = "raw",
    enhance_markdown: bool = True,
) -> ReadResult:
    """
    Async best-effort read. See `read_best_effort()` for semantics.
    """
    return await read_async(browser, output_format=output_format, enhance_markdown=enhance_markdown)


def _extract_json_payload(text: str) -> dict[str, Any]:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))
    inline = re.search(r"(\{.*\})", text, re.DOTALL)
    if inline:
        return json.loads(inline.group(1))
    return json.loads(text)


def extract(
    browser: SentienceBrowser,
    llm: LLMProvider,
    query: str,
    schema: type[BaseModel] | None = None,
    max_chars: int = 12000,
) -> ExtractResult:
    """
    Extract structured data from the current page using read() markdown + LLM.
    """
    result = read(browser, output_format="markdown", enhance_markdown=True)
    if result.status != "success":
        return ExtractResult(ok=False, error=result.error)

    content = result.content[:max_chars]
    schema_desc = ""
    if schema is not None:
        schema_desc = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    system = "You extract structured data from markdown content. Return only JSON. No prose."
    user = f"QUERY:\n{query}\n\nSCHEMA:\n{schema_desc}\n\nCONTENT:\n{content}"
    response = llm.generate(system, user)
    raw = response.content.strip()

    if schema is None:
        return ExtractResult(ok=True, data={"text": raw}, raw=raw, llm_response=response)

    try:
        payload = _extract_json_payload(raw)
        validated = schema.model_validate(payload)
        return ExtractResult(ok=True, data=validated, raw=raw, llm_response=response)
    except (json.JSONDecodeError, ValidationError) as exc:
        return ExtractResult(ok=False, error=str(exc), raw=raw, llm_response=response)


async def extract_async(
    browser: AsyncSentienceBrowser,
    llm: LLMProvider,
    query: str,
    schema: type[BaseModel] | None = None,
    max_chars: int = 12000,
) -> ExtractResult:
    """
    Async version of extract().
    """
    result = await read_async(browser, output_format="markdown", enhance_markdown=True)
    if result.status != "success":
        return ExtractResult(ok=False, error=result.error)

    content = result.content[:max_chars]
    schema_desc = ""
    if schema is not None:
        schema_desc = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    system = "You extract structured data from markdown content. Return only JSON. No prose."
    user = f"QUERY:\n{query}\n\nSCHEMA:\n{schema_desc}\n\nCONTENT:\n{content}"
    response = await llm.generate_async(system, user)
    raw = response.content.strip()

    if schema is None:
        return ExtractResult(ok=True, data={"text": raw}, raw=raw, llm_response=response)

    try:
        payload = _extract_json_payload(raw)
        validated = schema.model_validate(payload)
        return ExtractResult(ok=True, data=validated, raw=raw, llm_response=response)
    except (json.JSONDecodeError, ValidationError) as exc:
        return ExtractResult(ok=False, error=str(exc), raw=raw, llm_response=response)
