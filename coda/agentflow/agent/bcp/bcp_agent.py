"""BrowseComp-Plus multi-turn retrieval agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from coda.agentflow.agent import register_agent
from coda.agentflow.agent.base_agent import BaseAgent
from coda.agentflow.utils import CONTEXT_LENGTH_EXCEEDED
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry constants for open_page (aligned with ref/rl open_page_tool.py)
# ---------------------------------------------------------------------------

_OPEN_PAGE_MAX_RETRIES = 2          # open_page: 2 retries (ref/rl open_page_tool.py)
_SEARCH_MAX_RETRIES = 10            # search: 10 retries (ref/rl call_search_api MAX_RETRIES)
_RETRY_DELAY = 1                    # seconds; delay for attempt N = _RETRY_DELAY * N
# 4xx client errors are the caller's fault — do not retry
_NON_RETRYABLE_STATUS: frozenset[int] = frozenset({400, 401, 403, 404, 422})
# 5xx server errors that are transient — retry on these
_RETRYABLE_SERVER_STATUS: frozenset[int] = frozenset({500, 502, 503, 504})

# ---------------------------------------------------------------------------
# Tool schemas — adapted from ref/rl search_tool_config.yaml
# ---------------------------------------------------------------------------

_SEARCH_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Performs a search over the BrowseComp-Plus corpus. Supply 'query_list' (a list "
            "containing a single query string) and optional 'topk' (default 3). Returns "
            "top-k document snippets (truncated to ~512 tokens each) with their docid. "
            "Use open_page to read a full document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 1,
                    "description": "A list containing a single search query.",
                },
                "topk": {
                    "type": "integer",
                    "description": "Return the top k pages (default 3).",
                },
            },
            "required": ["query_list"],
        },
    },
}

_OPEN_PAGE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "open_page",
        "description": (
            "Retrieve the most relevant passages from a specific document. "
            "Provide the docid from prior search results and a query describing "
            "what information you need. Returns the top matching chunks instead of the full document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "docid": {
                    "type": "string",
                    "description": "Document ID from search results, e.g. the value shown in [docid: xxx].",
                },
                "query": {
                    "type": "string",
                    "description": "The question or keywords to search for within this document.",
                },
            },
            "required": ["docid", "query"],
        },
    },
}

_FINISH_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Submit your final answer. Call this when you have a definitive answer "
            "or cannot progress further."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your final answer. Be concise and precise.",
                },
            },
            "required": ["answer"],
        },
    },
}

_TOOL_SCHEMAS: list[dict] = [_SEARCH_SCHEMA, _OPEN_PAGE_SCHEMA, _FINISH_SCHEMA]


# ---------------------------------------------------------------------------
# Retrieval result formatting (aligned with ref/rl search_r1_like_utils.py)
# ---------------------------------------------------------------------------


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to approximately *max_tokens* tokens.

    Uses the heuristic 1 token ≈ 4 characters (same as ref/rl).
    """
    char_limit = max_tokens * 4
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + "..."


def _passages2string(retrieval_result: list[dict], max_tokens_per_doc: int = 512) -> str:
    """Format a list of retrieval results into a readable string for the LLM.

    Each entry in *retrieval_result* is expected to have the shape returned by
    the ``/retrieve`` endpoint::

        {"docid": "...", "document": {"contents": "<title>\\n<body>"}}

    The body of each document is truncated to *max_tokens_per_doc* tokens so
    that the combined search result stays within the context budget.
    """
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        text = _truncate_to_tokens(text, max_tokens_per_doc)
        docid = doc_item.get("docid", "")
        if docid:
            format_reference += f"Doc {idx + 1} (Title: {title}) [docid: {docid}]\n{text}\n\n"
        else:
            format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@register_agent("bcp")
class BCPAgent(BaseAgent):
    """BrowseComp-Plus multi-turn retrieval agent.

    Three-tool setup:
    1. Sends the question as a user message alongside SYSTEM_PROMPT to the LLM.
    2. Executes search / open_page tool calls against the retrieval service.
    3. Terminates when the model calls the finish tool with its final answer.
    4. Computes reward via the injected reward_fn.

    Config keys (injected via ``data_source.agent`` / ``data_sources[].agent``):
        retrieval_service_url (str): Base URL of the retrieval server.
        search_topk (int): Documents returned per search call (default: 3).
        max_queries_per_call (int): Max queries sent per search call (default: 1).
        topk_chunks (int): Chunks returned per open_page call (default: 3).
        chunk_size (int): Chunk size in tokens for open_page (default: 512).
        chunk_overlap (int): Chunk overlap in tokens for open_page (default: 32).
        max_turns (int): Maximum LLM turns before forced termination (default: 20).
        tool_timeout (float): HTTP timeout for tool calls in seconds (default: 30.0).
    """

    SYSTEM_PROMPT = """You are an expert investigative research agent. Your task is to solve complex multi-hop
queries using a search engine.

CRITICAL RULES:
1. ACT, DON'T JUST THINK: Every <think> block MUST be followed immediately by a <tool_call>. Keep thinking
   under 3-5 sentences. Never plan multiple searches in one think block — just pick the single best query
   and execute it.
2. KEYWORD SEARCHING: The search tool is keyword-based. Extract rare, specific entities from the question as keywords.
   - BAD: ["author who published 3 articles in 2019 and was interviewed"] (too conversational)
   - GOOD: ["Tribeca Festival Gotham Week audio selection"] (specific named entities)
3. MULTI-HOP: Break the question into sub-questions. Solve one hop per search. Find unique entities first,
   then use them to find the next hop.
4. RETRY WITH DIFFERENT ANGLES: If a search fails, do NOT repeat similar keywords. Try a completely
   different angle — use different entities from the question, drop modifiers, or search for a different
   hop entirely.
5. ANTI-HALLUCINATION: Never guess. You only know what the documents tell you.
6. VERIFICATION: Use `open_page` with both a docid and a query to retrieve relevant passages from a promising document.
7. KNOW WHEN TO STOP: If you have a well-supported answer, call finish immediately. If after several
   searches you cannot find the answer, call finish with your best guess rather than searching endlessly.

Always use the finish tool to submit your final answer."""

    def __init__(
        self,
        router_url: str = "http://localhost:8000",
        reward_fn: Any = None,
        completion_params: dict = None,
        max_response_len_per_trajectory: int = 0,
        temperature: float = 0.7,
        max_turns: int = 20,
        retrieval_service_url: str = "http://127.0.0.1:9000",
        search_topk: int = 3,
        max_queries_per_call: int = 1,
        topk_chunks: int = 3,
        chunk_size: int = 512,
        chunk_overlap: int = 32,
        tool_timeout: float = 30.0,
        search_timeout: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            router_url,
            completion_params=completion_params,
            max_response_len_per_trajectory=max_response_len_per_trajectory,
            temperature=temperature,
            **kwargs,
        )
        self.reward_fn = reward_fn
        self.max_turns = max_turns
        self.retrieval_service_url = retrieval_service_url.rstrip("/")
        self.search_topk = search_topk
        self.max_queries_per_call = max_queries_per_call
        self.topk_chunks = topk_chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._closed = False
        # Separate clients: LLM (very slow), open_page, search (tight timeout)
        self.llm_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        self.tool_client = httpx.AsyncClient(timeout=httpx.Timeout(tool_timeout))
        self.search_client = httpx.AsyncClient(timeout=httpx.Timeout(search_timeout))
        logger.info(
            "BCPAgent init: router=%s retrieval=%s max_turns=%d search_topk=%d",
            router_url,
            retrieval_service_url,
            max_turns,
            search_topk,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    async def run_trajectory(self, trajectory: dict[str, Any]) -> Reward:
        """Run one BCP trajectory and return the reward.

        Args:
            trajectory: Dict with keys ``prompt`` (question text or message list)
                    and ``label`` (ground-truth answer dict).

        Returns:
            A :class:`~coda.reward.reward.Reward` instance with the final score.
        """
        prompt = trajectory.get("prompt")
        label = trajectory.get("label") or {}

        if isinstance(prompt, list):
            messages: list[dict] = [dict(m) for m in prompt]
        elif isinstance(prompt, str):
            messages: list[dict] = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        else:
            messages: list[dict] = [{"role": "user", "content": str(prompt)}]

        completed_turns = sum(message.get("role") == "assistant" for message in messages)
        for turn in range(completed_turns, self.max_turns):
            try:
                # The Router owns the trajectory-wide token budget and clamps
                # this request to the true remaining response length.
                response = await self._call_llm(
                    messages, self.max_response_len_per_trajectory
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == httpx.codes.BAD_REQUEST:
                    try:
                        err = e.response.json().get("error", {})
                    except Exception:
                        err = {}
                    if err.get("type") == CONTEXT_LENGTH_EXCEEDED:
                        logger.warning(
                            "BCPAgent: max_response_len_per_trajectory exhausted (router): %s",
                            err.get("message", ""),
                        )
                        break
                raise

            choice = response["choices"][0]
            msg = choice["message"]
            content: str = msg.get("content") or ""
            tool_calls: list[dict] = msg.get("tool_calls") or []

            if not tool_calls:
                # Model produced text without any tool call — treat as end of trajectory.
                messages.append({"role": "assistant", "content": content})
                break

            # Append the assistant turn (keep full tool_calls structure for reward function).
            messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            })

            # Execute all tool calls uniformly — including finish.
            # Mirror ref/rl: append every tool response first, then check for
            # finish/budget termination (ref/rl _handle_processing_tools_state).
            finished = False
            turn_tool_results: list[str] = []
            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                try:
                    arguments = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                tool_result = await self._execute_tool(name, arguments)
                turn_tool_results.append(tool_result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_result,
                })
                logger.debug(
                    "tool=%s result_len=%d",
                    name, len(tool_result),
                )

                # Terminate after finish tool response is appended (mirrors ref/rl).
                if name == "finish":
                    logger.debug("tool=finish answer=%s", arguments.get("answer", "")[:80])
                    finished = True
                    break

            if finished:
                break
        else:
            logger.warning("BCPAgent: max_turns=%d reached without finish call", self.max_turns)

        if self.reward_fn is not None:
            return self.reward_fn(messages, label, {})
        return Reward(final_reward=0.0)

    async def clear(self) -> None:
        """Release HTTP clients. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        await self.llm_client.aclose()
        await self.tool_client.aclose()
        await self.search_client.aclose()
        logger.info("BCPAgent resources cleared")

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_llm(self, messages: list[dict], turn_max_tokens: int) -> dict:
        body = {
            "model": "default",
            "messages": messages,
            "tools": _TOOL_SCHEMAS,
            "tool_choice": "auto",
            "temperature": self.temperature,
            **self.completion_params,
        }
        body["max_tokens"] = turn_max_tokens
        resp = await self.llm_client.post(
            f"{self.router_url}/v1/chat/completions", json=body
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(self, name: str, arguments: dict) -> str:
        """Dispatch a tool call to the retrieval service.

        Args:
            name: Tool name — ``"search"`` or ``"open_page"``.
            arguments: Parsed tool arguments dict.

        Returns:
            JSON-encoded tool result string.
        """
        try:
            if name == "search":
                query_list = arguments.get("query_list") or [arguments.get("query", "")]
                topk = int(arguments.get("topk") or self.search_topk)
                return await self._do_search(query_list, topk=topk)
            elif name == "open_page":
                return await self._do_open_page(
                    arguments.get("docid", ""),
                    arguments.get("query", ""),
                )
            elif name == "finish":
                answer = arguments.get("answer", "")
                if not answer:
                    return json.dumps({"error": "Missing 'answer' parameter."})
                return json.dumps({"result": f"Answer submitted: {answer}"}, ensure_ascii=False)
            return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as exc:
            logger.warning("Tool %s failed: %s", name, exc)
            return json.dumps({"error": str(exc)})

    async def _do_search(self, query_list: list[str], topk: int | None = None) -> str:
        """POST to /retrieve and return formatted search results.

        Aligned with ref/rl ``perform_single_search_batch`` / ``call_search_api``:
        - Retries up to ``_SEARCH_MAX_RETRIES`` times on 5xx / network errors.
        - Multiple query results are formatted independently and joined with
          ``"\\n---\\n"`` (same as ref/rl), not merged/deduped.
        - Returns ``{"result": "Search error: ..."}`` on failure (not an exception).

        Args:
            query_list: Queries supplied by the model.
            topk: Number of results to return; falls back to ``self.search_topk``.

        Returns:
            JSON-encoded ``{"result": "<formatted passages>"}`` string.
        """
        if not query_list:
            return json.dumps({"result": "Search error: query_list is empty."})

        # Limit to max_queries_per_call to control context budget
        queries = query_list
        if len(query_list) > self.max_queries_per_call:
            queries = [str(q) for q in query_list[: self.max_queries_per_call]]
        payload = {"queries": queries, "topk": int(topk or self.search_topk), "return_scores": True}
        last_error: str = ""

        for attempt in range(_SEARCH_MAX_RETRIES):
            try:
                resp = await self.search_client.post(
                    f"{self.retrieval_service_url}/retrieve", json=payload
                )

                # 5xx — transient server error, retry
                if resp.status_code in _RETRYABLE_SERVER_STATUS:
                    last_error = f"Server error HTTP {resp.status_code}"
                    logger.warning(
                        "search server error attempt=%d/%d status=%d",
                        attempt + 1, _SEARCH_MAX_RETRIES, resp.status_code,
                    )
                    if attempt < _SEARCH_MAX_RETRIES - 1:
                        await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
                    continue

                resp.raise_for_status()
                data = resp.json()
                raw_results: list = data.get("result", [])

                if not raw_results:
                    return json.dumps({"result": "No search results found."}, ensure_ascii=False)

                # Format each query's results independently, join with separator
                # (mirrors ref/rl: "\n---\n".join(pretty_results))
                pretty_results = [_passages2string(q_docs) for q_docs in raw_results]
                final_result = "\n---\n".join(pretty_results)
                return json.dumps({"result": final_result}, ensure_ascii=False)

            except httpx.ConnectError:
                last_error = "Connection refused — retrieval server may be down"
                logger.warning(
                    "search connection error attempt=%d/%d", attempt + 1, _SEARCH_MAX_RETRIES
                )
            except httpx.TimeoutException:
                last_error = f"Request timed out (attempt {attempt + 1}/{_SEARCH_MAX_RETRIES})"
                logger.warning(
                    "search timeout attempt=%d/%d", attempt + 1, _SEARCH_MAX_RETRIES
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "search error attempt=%d/%d: %s", attempt + 1, _SEARCH_MAX_RETRIES, exc
                )

            if attempt < _SEARCH_MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAY * (attempt + 1))

        return json.dumps(
            {"result": f"Search error: {last_error}"}, ensure_ascii=False
        )

    async def _do_open_page(self, docid: str, query: str) -> str:
        """POST to /get_doc_chunks and format the chunk response.

        Aligned with ref/rl open_page_tool.py:
        - Validates docid and query before making the HTTP call.
        - Passes explicit chunk parameters to the server.
        - Does not retry on 4xx (client errors); retries up to ``_OPEN_PAGE_MAX_RETRIES``
          times on network/timeout errors with linear back-off.

        Args:
            docid: Document ID from prior search results.
            query: Question or keywords to rank chunks against.

        Returns:
            JSON-encoded ``{"result": "<formatted chunks>"}`` or ``{"error": "..."}`` string.
        """
        # Sanitise: strip copy-paste artifacts like "[docid: 12345]"
        if not isinstance(docid, str):
            docid = str(docid)
        if not isinstance(query, str):
            query = str(query)

        docid = docid.strip().strip("[]")
        if docid.startswith("docid:"):
            docid = docid[len("docid:"):].strip()

        if not docid:
            return json.dumps({"error": "Missing 'docid' parameter."})
        if not query:
            return json.dumps({
                "error": (
                    "Missing 'query' parameter. "
                    "Provide the question you want to answer from this document."
                )
            })

        payload = {
            "docid": docid,
            "query": query,
            "topk": self.topk_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        last_error: str = ""

        for attempt in range(_OPEN_PAGE_MAX_RETRIES):
            try:
                resp = await self.tool_client.post(
                    f"{self.retrieval_service_url}/get_doc_chunks", json=payload
                )

                # 4xx errors are non-retryable client errors
                if resp.status_code in _NON_RETRYABLE_STATUS:
                    try:
                        detail = resp.json()
                        msg = detail.get("detail", detail.get("error", resp.text[:200]))
                    except Exception:
                        msg = resp.text[:200]
                    return json.dumps({
                        "error": (
                            f"Document retrieval failed (HTTP {resp.status_code}): {msg}. "
                            "Do NOT retry with the same docid. Try a different docid or search query."
                        )
                    })

                resp.raise_for_status()
                data = resp.json()

                if data.get("error"):
                    # Semantic error from server (e.g. docid not found) — no retry
                    return json.dumps({
                        "error": f"{data['error']}. Try a different docid from search results."
                    })

                # Format chunks into readable text
                title = data.get("title", "Unknown")
                chunks = data.get("chunks", [])
                total = data.get("total_chunks", 0)
                parts = [
                    f"Document (Title: {title}) [docid: {docid}] "
                    f"— showing {len(chunks)}/{total} most relevant chunks:\n"
                ]
                for chunk in chunks:
                    parts.append(
                        f"--- Chunk {chunk['chunk_index'] + 1}/{total} "
                        f"(score: {chunk['score']:.3f}) ---\n"
                        f"{chunk['text']}\n"
                    )
                return json.dumps({"result": "\n".join(parts)}, ensure_ascii=False)

            except httpx.ConnectError:
                last_error = "Connection refused — retrieval server may be down"
                logger.warning(
                    "open_page connection error docid=%s attempt=%d/%d",
                    docid, attempt + 1, _OPEN_PAGE_MAX_RETRIES,
                )
            except httpx.TimeoutException:
                last_error = f"Request timed out (attempt {attempt + 1}/{_OPEN_PAGE_MAX_RETRIES})"
                logger.warning(
                    "open_page timeout docid=%s attempt=%d/%d",
                    docid, attempt + 1, _OPEN_PAGE_MAX_RETRIES,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "open_page error docid=%s attempt=%d/%d: %s",
                    docid, attempt + 1, _OPEN_PAGE_MAX_RETRIES, exc,
                )

            if attempt < _OPEN_PAGE_MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAY * (attempt + 1))

        return json.dumps({
            "error": (
                f"Failed to fetch document '{docid}' after {_OPEN_PAGE_MAX_RETRIES} attempts: "
                f"{last_error}. Try a different docid or search query."
            )
        })
