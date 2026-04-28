import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class KnowledgeGraphClient:
    """Client for interacting with the Neo4j Knowledge Graph.
    
    Provides deterministic grounding for entities extracted by Gemini.
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        if not self.password:
            print("WARNING: NEO4J_PASSWORD not set. Connection may fail.")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=3,
        )

    def close(self):
        self.driver.close()

    def get_protocol_for_entity(self, entity_name: str) -> list[dict]:
        """Retrieves safety protocols and actions for a given entity name."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})-[:REQUIRES]->(p:Protocol)
                RETURN p.name as protocol, p.severity as severity, p.actions as actions
                """,
                name=entity_name
            )
            return [record.data() for record in result]

    def get_affected_locations(self, entity_name: str) -> list[dict]:
        """Retrieves locations affected by a given entity."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})-[:AFFECTS]->(l:Location)
                RETURN l.name as location, l.type as type, l.lat as lat, l.lng as lng
                """,
                name=entity_name
            )
            return [record.data() for record in result]

    def query_semantic_match(self, entity_type: str) -> list[dict]:
        """Finds all entities of a certain type in the KG."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {type: $type}) RETURN e.name as name",
                type=entity_type
            )
            return [record.data() for record in result]

if __name__ == "__main__":
    # Test
    client = KnowledgeGraphClient()
    try:
        protocols = client.get_protocol_for_entity("ChemicalSpill")
        print(f"Protocols for ChemicalSpill: {protocols}")
    except Exception as e:
        print(f"Neo4j connection error: {e}")
    finally:
        client.close()
