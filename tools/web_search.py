import time
import random
from typing import Optional
from pydantic import BaseModel


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


# stub data — simulates a real search engine returning results
STUB_RESULTS = {
    "climate change": [
        SearchResult(
            title="Effects of Climate Change - NASA",
            url="https://climate.nasa.gov/effects",
            snippet="Global temperatures have risen 1.1°C since pre-industrial times, causing sea level rise and extreme weather.",
            relevance_score=0.95,
        ),
        SearchResult(
            title="Ocean Acidification Overview - NOAA",
            url="https://www.noaa.gov/ocean-acidification",
            snippet="Oceans absorb 30% of CO2 emissions, causing acidification that harms coral reefs and marine life.",
            relevance_score=0.88,
        ),
    ],
    "machine learning": [
        SearchResult(
            title="What is Machine Learning? - IBM",
            url="https://www.ibm.com/topics/machine-learning",
            snippet="Machine learning is a branch of AI that enables systems to learn from data without being explicitly programmed.",
            relevance_score=0.92,
        ),
        SearchResult(
            title="Machine Learning Basics - Google",
            url="https://developers.google.com/machine-learning",
            snippet="ML models find patterns in data and use those patterns to make predictions on new data.",
            relevance_score=0.85,
        ),
    ],
    "default": [
        SearchResult(
            title="General Reference - Wikipedia",
            url="https://en.wikipedia.org/wiki/Main_Page",
            snippet="Wikipedia is a free online encyclopedia with millions of articles on a wide range of topics.",
            relevance_score=0.60,
        ),
        SearchResult(
            title="Research Overview - Britannica",
            url="https://www.britannica.com",
            snippet="Britannica provides authoritative content covering science, history, culture and more.",
            relevance_score=0.55,
        ),
    ],
}


def web_search(query: str, timeout: float = 5.0) -> SearchResponse:
    """
    Stub web search tool.

    Failure contracts:
    - timeout: if query contains 'timeout_test', simulates a timeout
    - empty:   if query contains 'empty_test', returns no results
    - malformed: if query is blank or non-string, returns malformed failure
    """
    start = time.time()

    # malformed input
    if not query or not isinstance(query, str):
        return SearchResponse(
            query=str(query),
            results=[],
            total_found=0,
            failure_mode="malformed",
        )

    # simulated timeout
    if "timeout_test" in query.lower():
        time.sleep(timeout + 0.1)
        return SearchResponse(
            query=query,
            results=[],
            total_found=0,
            failure_mode="timeout",
        )

    # simulated empty result
    if "empty_test" in query.lower():
        return SearchResponse(
            query=query,
            results=[],
            total_found=0,
            failure_mode="empty",
        )

    # match stub data by keyword
    matched_key = "default"
    for key in STUB_RESULTS:
        if key in query.lower():
            matched_key = key
            break

    results = STUB_RESULTS[matched_key]

    # simulate small network delay
    time.sleep(random.uniform(0.1, 0.4))

    latency = (time.time() - start) * 1000
    print(f"[web_search] query='{query}' results={len(results)} latency={latency:.0f}ms")

    return SearchResponse(
        query=query,
        results=results,
        total_found=len(results),
        failure_mode="none",
    )