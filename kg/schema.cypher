// ChainGuard AI Knowledge Graph Schema
// Defines relationships between operational hazards, climate events, and safety protocols.

// 1. Constraints
CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;
CREATE CONSTRAINT protocol_name IF NOT EXISTS FOR (p:Protocol) REQUIRE p.name IS UNIQUE;

// 2. Sample Data: Hazards and Protocols
// Operational Hazards
MERGE (h1:Entity {name: "ChemicalSpill", type: "HAZARD", description: "Accidental release of hazardous chemicals"})
MERGE (p1:Protocol {name: "HazmatRouting", severity: "CRITICAL", actions: ["reroute", "alert_authorities", "containment_check"]})
MERGE (h1)-[:REQUIRES]->(p1)

MERGE (h2:Entity {name: "TrafficCongestion", type: "LOGISTICS", description: "Severe traffic delays on major corridors"})
MERGE (p2:Protocol {name: "RouteOptimization", severity: "LOW", actions: ["recalculate_eta", "notify_retailers"]})
MERGE (h2)-[:REQUIRES]->(p2)

// Climate Events (SDG 13 Alignment)
MERGE (c1:Entity {name: "Flood", type: "CLIMATE", description: "Heavy flooding blocking primary routes"})
MERGE (p3:Protocol {name: "FloodContingency", severity: "HIGH", actions: ["activate_backup_warehouse", "delay_shipments", "re-route_via_rail"]})
MERGE (c1)-[:REQUIRES]->(p3)

MERGE (c2:Entity {name: "Cyclone", type: "CLIMATE", description: "High wind speeds and heavy rain affecting port operations"})
MERGE (p4:Protocol {name: "PortClosureProtocol", severity: "CRITICAL", actions: ["secure_containers", "suspend_unloading", "reroute_vessels"]})
MERGE (c2)-[:REQUIRES]->(p4)

// 3. Suppliers and Locations
MERGE (l1:Location {name: "PortOfJNPT", type: "PORT", lat: 18.95, lng: 72.95})
MERGE (l2:Location {name: "Mumbai_Warehouse", type: "WAREHOUSE", lat: 19.076, lng: 72.877})
MERGE (l1)-[:CONNECTED_VIA {mode: "road"}]->(l2)

// 4. Critical Dependencies
MERGE (h1)-[:AFFECTS]->(l1)
MERGE (c1)-[:AFFECTS]->(l2)
