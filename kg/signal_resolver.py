from kg.gemini_extractor import GeminiEntityExtractor
from kg.neo4j_client import KnowledgeGraphClient
import numpy as np

class SignalResolver:
    """Integrates Gemini Extraction with Knowledge Graph Grounding.
    
    This is the core middleware that converts raw text alerts into 
    actionable, deterministic operational concepts.
    """

    def __init__(self):
        self.extractor = GeminiEntityExtractor()
        self.kg_client = KnowledgeGraphClient()

    def resolve(self, raw_text: str) -> dict:
        """Processes a raw text alert.
        
        Steps:
        1. Extract entities using Gemini (LLM Path).
        2. Ground entities in Neo4j (Deterministic Path).
        3. Attach protocols and confidence scores.
        """
        # 1. Extraction
        extracted_entities = self.extractor.extract(raw_text)
        if not extracted_entities:
            return {
                "raw_text": raw_text,
                "results": [],
                "overall_confidence": 0.0,
                "note": "No entities extracted. Gemini may be unavailable, timed out, or the alert contained no actionable entities.",
            }
        
        results = []
        confidence_sum = 0
        
        # 2. Grounding
        for entity in extracted_entities:
            name = entity["name"]
            
            # Try to find exact match in KG
            try:
                protocols = self.kg_client.get_protocol_for_entity(name)
                affected_locs = self.kg_client.get_affected_locations(name)
            except Exception as exc:
                protocols = []
                affected_locs = []
                print(f"Neo4j grounding failed for {name}: {exc}")
            
            is_grounded = len(protocols) > 0
            
            results.append({
                "entity": entity,
                "grounded": is_grounded,
                "protocols": protocols if is_grounded else [],
                "affected_locations": affected_locs if is_grounded else []
            })
            
            confidence_sum += 1.0 if is_grounded else 0.5 # Half confidence if not in KG

        avg_confidence = confidence_sum / len(results) if results else 0.0
        
        return {
            "raw_text": raw_text,
            "results": results,
            "overall_confidence": avg_confidence
        }

if __name__ == "__main__":
    # Test
    resolver = SignalResolver()
    output = resolver.resolve("Chemical spill detected at Port of JNPT! Heavy disruption expected.")
    import json
    print(json.dumps(output, indent=2))
