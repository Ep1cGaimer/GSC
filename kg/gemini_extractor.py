import os
import json
import time
import httpx
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class GeminiEntityExtractor:
    """Uses Google Gemini API to extract structured supply chain entities from unstructured alerts.
    
    This replaces traditional NER models like spaCy with a more robust, zero-shot 
    extraction that understands supply chain context (SDG 9/12/13).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set as an environment variable.")

        self.timeout_seconds = max(10.0, float(os.getenv("GEMINI_TIMEOUT_SECONDS", "20")))
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.max_retries = max(1, int(os.getenv("GEMINI_MAX_RETRIES", "2")))

        # Ignore broken shell proxy env vars for Gemini API traffic.
        http_client = httpx.Client(trust_env=False)
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                timeout=int(self.timeout_seconds * 1000),
                httpxClient=http_client,
            ),
        )

    def extract(self, raw_text: str) -> list[dict]:
        """Extracts entities from raw text.
        
        Returns:
            list of dicts: [{"name": "ChemicalSpill", "type": "HAZARD", "location": "Port of JNPT"}]
        """
        prompt = f"""
        Analyze the following supply chain alert or news snippet. 
        Extract any operational hazards, climate events, locations, or affected suppliers.
        
        Return the results as a JSON array of objects. 
        Each object should have:
        - "name": The canonical name of the entity (e.g., "Flood", "ChemicalSpill", "LaborStrike").
        - "type": One of [HAZARD, CLIMATE, LOGISTICS, LOCATION, SUPPLIER].
        - "description": A brief summary of the impact.
        
        If no relevant entities are found, return an empty array [].
        
        Alert text: "{raw_text}"
        
        JSON:
        """
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0,
                    ),
                )
                entities = json.loads(response.text or "[]")
                return entities
            except Exception as e:
                print(f"Error extracting entities with Gemini (attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    return []
                time.sleep(1.5 * attempt)

if __name__ == "__main__":
    # Test
    extractor = GeminiEntityExtractor()
    test_text = "Severe flooding reported near the Mumbai warehouse, expected to block road transport for 48 hours."
    results = extractor.extract(test_text)
    print(json.dumps(results, indent=2))
