import os
import time
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    relevance_score: float          # 0.0 to 1.0


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int
    failure_mode: Optional[str] = None   # none, timeout, empty, malformed


def web_search(query: str, timeout: float = 5.0) -> SearchResponse:
    start = time.time()

    # malformed input
    if not query or not isinstance(query, str):
        return SearchResponse(
            query=str(query),
            results=[],
            total_found=0,
            failure_mode="malformed",
        )

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return SearchResponse(
            query=query,
            results=[],
            total_found=0,
            failure_mode="malformed",
        )

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=5, timeout=timeout)

        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                relevance_score=r.get("score", 0.0),
            )
            for r in response.get("results", [])
        ]

        if not results:
            return SearchResponse(
                query=query,
                results=[],
                total_found=0,
                failure_mode="empty",
            )

    except Exception:
        return SearchResponse(
            query=query,
            results=[],
            total_found=0,
            failure_mode="timeout",
        )

    latency = (time.time() - start) * 1000
    print(f"[web_search] query='{query}' results={len(results)} latency={latency:.0f}ms")

    return SearchResponse(
        query=query,
        results=results,
        total_found=len(results),
        failure_mode="none",
    )
