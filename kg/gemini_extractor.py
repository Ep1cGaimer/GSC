import os
import json
import google.generativeai as genai
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
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

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
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            entities = json.loads(response.text)
            return entities
        except Exception as e:
            print(f"Error extracting entities with Gemini: {e}")
            return []

if __name__ == "__main__":
    # Test
    extractor = GeminiEntityExtractor()
    test_text = "Severe flooding reported near the Mumbai warehouse, expected to block road transport for 48 hours."
    results = extractor.extract(test_text)
    print(json.dumps(results, indent=2))
