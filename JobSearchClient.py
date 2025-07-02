from tavily import TavilyClient
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename="langgraph_debug.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

class JobSearchClient:
    def __init__(self, tavily_api_key: str = None, serper_api_key: str = None):
        """Initialize the job search client with Tavily and Serper API keys."""
        self.tavily_api_key = tavily_api_key
        self.serper_api_key = serper_api_key
        if not tavily_api_key:
            logging.error("TAVILY_API_KEY is missing")
        if not serper_api_key:
            logging.error("SERPER_API_KEY is missing")
        self.tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def _make_tavily_request(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Make a Tavily API request with retries."""
        for attempt in range(self.max_retries):
            try:
                # Truncate query to 400 characters to avoid Tavily's limit
                truncated_query = query[:400] if len(query) > 400 else query
                search_params = {
                    "query": f"{truncated_query} jobs India",
                    "search_depth": "advanced",
                    "include_domains": ["naukri.com", "linkedin.com", "indeed.com", "monsterindia.com", "timesjobs.com", "shine.com", "glassdoor.com"],
                    "max_results": max_results
                }
                logging.debug(f"Tavily search parameters (attempt {attempt + 1}): {search_params}")
                response = self.tavily_client.search(**search_params)
                logging.debug(f"Tavily API raw response: {response}")
                job_listings = []
                results = response.get("results", [])
                if not results:
                    logging.warning("Tavily API returned empty results")
                    return []
                for result in results[:max_results]:
                    title = result.get("title", "N/A")
                    if any(keyword in title.lower() for keyword in ["job", "hiring", "vacancy", "career", "recruitment"]):
                        job_listings.append({
                            "title": title,
                            "company": result.get("source", "N/A") or result.get("snippet", "N/A")[:50],
                            "location": "India",
                            "link": result.get("url", "#")
                        })
                logging.info(f"Found {len(job_listings)} job listings via Tavily API")
                return job_listings
            except Exception as e:
                logging.error(f"Tavily API error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return [{"error": f"Tavily API failed after {self.max_retries} attempts: {str(e)}"}]
        return []

    def _make_serper_request(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Make a Serper API request with retries."""
        for attempt in range(self.max_retries):
            try:
                url = "https://google.serper.dev/search"
                headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
                payload = {
                    "q": f"{query} jobs India",
                    "num": max_results
                }
                logging.debug(f"Serper API payload (attempt {attempt + 1}): {payload}")
                response = requests.post(url, headers=headers, json=payload)
                logging.debug(f"Serper API raw response: {response.text}")
                if response.status_code == 200:
                    data = response.json()
                    jobs = data.get("organic", [])
                    if not jobs:
                        logging.warning("Serper API returned empty organic results")
                        return []
                    job_listings = []
                    for job in jobs[:max_results]:
                        title = job.get("title", "N/A")
                        if any(keyword in title.lower() for keyword in ["job", "hiring", "vacancy", "career", "recruitment"]):
                            snippet = job.get("snippet", "N/A")
                            company = snippet.split(" - ")[0] if " - " in snippet else snippet[:50]
                            job_listings.append({
                                "title": title,
                                "company": company,
                                "location": "India",
                                "link": job.get("link", "#")
                            })
                    logging.info(f"Found {len(job_listings)} job listings via Serper API")
                    return job_listings
                else:
                    logging.error(f"Serper API failed: {response.status_code} - {response.text}")
                    return [{"error": f"Serper API failed: {response.status_code}"}]
            except Exception as e:
                logging.error(f"Serper API error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return [{"error": f"Serper API error after {self.max_retries} attempts: {str(e)}"}]
        return []

    def search_jobs(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for job listings in India using Tavily and Serper APIs.
        Args:
            query: The job search query (e.g., "Data Science").
            max_results: Maximum number of results to return.
        Returns:
            List of job dictionaries with title, company, location, and link.
        """
        job_listings = []

        # Try Tavily API first
        if self.tavily_client:
            job_listings = self._make_tavily_request(query, max_results)
            if job_listings and "error" not in job_listings[0]:
                return job_listings

        # Fallback to Serper API if Tavily fails or returns no results
        if self.serper_api_key:
            job_listings = self._make_serper_request(query, max_results)
            if job_listings and "error" not in job_listings[0]:
                return job_listings

        # Fallback to sample jobs if both APIs fail
        logging.warning("Both APIs failed or returned no results, returning sample jobs")
        job_listings = [
            {
                "title": "Sample Data Scientist",
                "company": "Sample Company",
                "location": "India",
                "link": "https://www.example.com/apply"
            },
            {
                "title": "Sample Software Engineer",
                "company": "Sample Tech Inc.",
                "location": "India",
                "link": "https://www.example.com/apply"
            }
        ]
        logging.debug("Returning sample job listings as fallback")
        return job_listings