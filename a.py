import os
import httpx
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- 1. Tavily Test ----
def test_tavily():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY is missing")
        return {"status": "failed", "error": "Missing TAVILY_API_KEY"}
    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={"query": "current price of bricks"},  # ✅ Correct payload
            headers={
                "Authorization": f"Bearer {api_key}",     # ✅ Correct auth header
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        return {
            "status": "success",
            "status_code": response.status_code,
            "output": response.json()
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"Tavily API error: {str(e)}")
        return {
            "status": "failed",
            "status_code": e.response.status_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error in Tavily API: {str(e)}")
        return {"status": "failed", "error": str(e)}

# ---- 2. Serper Test ----
def test_serper():
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        logger.error("SERPER_API_KEY is missing")
        return {"status": "failed", "error": "Missing SERPER_API_KEY"}
    try:
        response = httpx.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,                    # ✅ Proper auth header
                "Content-Type": "application/json"
            },
            json={"q": "current cement price"}          # ✅ Valid minimal query
        )
        response.raise_for_status()
        return {
            "status": "success",
            "status_code": response.status_code,
            "output": response.json()
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"Serper API error: {str(e)}")
        return {
            "status": "failed",
            "status_code": e.response.status_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error in Serper API: {str(e)}")
        return {"status": "failed", "error": str(e)}

# ---- 3. Gemini Test ----
def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is missing")
        return {"status": "failed", "error": "Missing GEMINI_API_KEY"}
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-1.5-flash")  # ✅ Updated model name
        response = model.generate_content("What is the current cost of concrete per cubic meter?")
        if response.text:
            return {"status": "success", "output": response.text}
        else:
            logger.warning("Gemini API returned empty response")
            return {"status": "failed", "error": "Empty response from Gemini"}
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return {"status": "failed", "error": str(e)}

# ---- Run All Tests ----
if __name__ == "__main__":
    print("\n--- Testing Tavily API ---")
    tavily_result = test_tavily()
    print(f"Status: {tavily_result['status']}")
    if tavily_result["status"] == "success":
        print(f"Status Code: {tavily_result['status_code']}")
        print(f"Output: {tavily_result['output']}")
    else:
        print(f"Error: {tavily_result['error']}")

    print("\n--- Testing Serper API ---")
    serper_result = test_serper()
    print(f"Status: {serper_result['status']}")
    if serper_result["status"] == "success":
        print(f"Status Code: {serper_result['status_code']}")
        print(f"Output: {serper_result['output']}")
    else:
        print(f"Error: {serper_result['error']}")

    print("\n--- Testing Gemini API ---")
    gemini_result = test_gemini()
    print(f"Status: {gemini_result['status']}")
    if gemini_result["status"] == "success":
        print(f"Output: {gemini_result['output']}")
    else:
        print(f"Error: {gemini_result['error']}")
