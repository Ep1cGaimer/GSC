"""
Graph Builder: Converts YAML topology configs into PyTorch Geometric HeteroData objects.
Handles dynamic topologies — nodes/edges can be added or removed between episodes.
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import HeteroData


# Node type feature dimensions
NODE_FEATURES = {
    "factory": ["capacity", "co2_per_unit"],
    "warehouse": ["capacity", "storage_cost"],
    "port": ["capacity", "handling_cost"],
    "retailer": ["demand_mean", "demand_std"],
}

EDGE_MODE_ENCODING = {"road": 0, "rail": 1, "sea": 2}


class GraphBuilder:
    """Builds PyG HeteroData from topology YAML configs.
    
    Each node type (factory, warehouse, port, retailer) gets its own feature matrix.
    Edge types are (source_type, mode, dest_type), e.g. ('factory', 'road', 'warehouse').
    
    This design ensures the GNN can handle variable numbers of nodes per type
    without retraining — PyG message-passing is topology-agnostic.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.node_id_to_idx = {}   # Maps node ID string to (type, local_index)
        self.node_id_to_data = {}  # Maps node ID to full node dict
        self._index_nodes()

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _index_nodes(self):
        """Build lookup tables from node IDs to typed indices."""
        type_counters = {}
        for node in self.config["nodes"]:
            ntype = node["type"]
            if ntype not in type_counters:
                type_counters[ntype] = 0
            self.node_id_to_idx[node["id"]] = (ntype, type_counters[ntype])
            self.node_id_to_data[node["id"]] = node
            type_counters[ntype] += 1

    def build(self, inventory_state: dict = None, disruptions: dict = None) -> HeteroData:
        """Build a HeteroData graph from the topology config.
        
        Args:
            inventory_state: Optional dict mapping node_id -> current_inventory (runtime state).
            disruptions: Optional dict with keys:
                - 'disabled_edges': list of (from_id, to_id) tuples
                - 'capacity_multipliers': dict mapping node_id -> multiplier
                - 'lead_time_multipliers': dict mapping edge key -> multiplier
        
        Returns:
            HeteroData with typed nodes and edges, ready for PyG message passing.
        """
        data = HeteroData()
        disruptions = disruptions or {}
        inventory_state = inventory_state or {}

        # --- Build node features per type ---
        nodes_by_type = {}
        coords_by_type = {}
        ids_by_type = {}

        for node in self.config["nodes"]:
            ntype = node["type"]
            if ntype not in nodes_by_type:
                nodes_by_type[ntype] = []
                coords_by_type[ntype] = []
                ids_by_type[ntype] = []

            feature_keys = NODE_FEATURES[ntype]
            features = [float(node.get(k, 0.0)) for k in feature_keys]

            # Apply capacity disruption
            cap_mult = disruptions.get("capacity_multipliers", {}).get(node["id"], 1.0)
            features[0] *= cap_mult  # capacity is always first feature

            # Add runtime inventory if available
            inv = float(inventory_state.get(node["id"], 0.0))
            features.append(inv)

            # Add normalized coordinates as features
            features.append(node["lat"] / 90.0)   # Normalize lat to [-1, 1]
            features.append(node["lng"] / 180.0)   # Normalize lng to [-1, 1]

            nodes_by_type[ntype].append(features)
            coords_by_type[ntype].append([node["lat"], node["lng"]])
            ids_by_type[ntype].append(node["id"])

        # Assign to HeteroData
        for ntype, features_list in nodes_by_type.items():
            data[ntype].x = torch.tensor(features_list, dtype=torch.float32)
            data[ntype].coords = torch.tensor(coords_by_type[ntype], dtype=torch.float32)
            data[ntype].node_ids = ids_by_type[ntype]

        # --- Build edge index per edge type ---
        disabled_edges = set()
        for e in disruptions.get("disabled_edges", []):
            disabled_edges.add((e[0], e[1]))

        edges_by_type = {}  # (src_type, mode, dst_type) -> {"src": [], "dst": [], "attr": []}
        lt_multipliers = disruptions.get("lead_time_multipliers", {})

        for edge in self.config["edges"]:
            from_id = edge["from"]
            to_id = edge["to"]

            # Skip disabled edges
            if (from_id, to_id) in disabled_edges:
                continue

            src_type, src_idx = self.node_id_to_idx[from_id]
            dst_type, dst_idx = self.node_id_to_idx[to_id]
            mode = edge["mode"]

            edge_type_key = (src_type, f"{mode}_to", dst_type)
            if edge_type_key not in edges_by_type:
                edges_by_type[edge_type_key] = {"src": [], "dst": [], "attr": []}

            edges_by_type[edge_type_key]["src"].append(src_idx)
            edges_by_type[edge_type_key]["dst"].append(dst_idx)

            # Edge features: [distance_km, lead_time_days, cost_per_unit, co2_per_km]
            edge_key = f"{from_id}->{to_id}"
            lt_mult = lt_multipliers.get(edge_key, 1.0)
            edge_attr = [
                edge["distance_km"] / 1500.0,          # Normalize
                edge["lead_time_days"] * lt_mult / 5.0, # Normalize
                edge["cost_per_unit"] / 3.0,             # Normalize
                edge["co2_per_km"] / 0.15,               # Normalize
            ]
            edges_by_type[edge_type_key]["attr"].append(edge_attr)

        # Assign edges to HeteroData
        for edge_type, edge_data in edges_by_type.items():
            src_indices = torch.tensor(edge_data["src"], dtype=torch.long)
            dst_indices = torch.tensor(edge_data["dst"], dtype=torch.long)
            data[edge_type].edge_index = torch.stack([src_indices, dst_indices], dim=0)
            data[edge_type].edge_attr = torch.tensor(edge_data["attr"], dtype=torch.float32)

        return data

    def get_node_ids_by_type(self, node_type: str) -> list:
        """Get all node IDs for a given type."""
        return [n["id"] for n in self.config["nodes"] if n["type"] == node_type]

    def get_all_agent_ids(self) -> list:
        """Get all node IDs that act as agents (warehouses)."""
        return self.get_node_ids_by_type("warehouse")

    def get_node_data(self, node_id: str) -> dict:
        """Get full node data dict by ID."""
        return self.node_id_to_data.get(node_id, {})

    def get_hazmat_zones(self) -> set:
        """Get set of node IDs that are hazmat-restricted zones."""
        return set(self.config.get("hazmat_zones", []))

    def get_edges_from(self, node_id: str) -> list:
        """Get all edges originating from a node."""
        return [e for e in self.config["edges"] if e["from"] == node_id]

    def get_edges_to(self, node_id: str) -> list:
        """Get all edges going to a node."""
        return [e for e in self.config["edges"] if e["to"] == node_id]

    def extract_subgraph(self, center_id: str, k_hops: int = 2) -> HeteroData:
        """Extract a k-hop subgraph around a node for decentralized actor input.
        
        Returns a HeteroData containing only nodes within k hops of center_id
        and the edges between them.
        """
        # BFS to find k-hop neighborhood
        visited = {center_id}
        frontier = {center_id}

        for _ in range(k_hops):
            next_frontier = set()
            for nid in frontier:
                for edge in self.config["edges"]:
                    if edge["from"] == nid and edge["to"] not in visited:
                        next_frontier.add(edge["to"])
                        visited.add(edge["to"])
                    if edge["to"] == nid and edge["from"] not in visited:
                        next_frontier.add(edge["from"])
                        visited.add(edge["from"])
            frontier = next_frontier

        # Build subgraph with only visited nodes
        sub_config = {
            "nodes": [n for n in self.config["nodes"] if n["id"] in visited],
            "edges": [e for e in self.config["edges"]
                      if e["from"] in visited and e["to"] in visited],
        }

        # Create temporary builder for subgraph
        sub_builder = GraphBuilder.__new__(GraphBuilder)
        sub_builder.config = sub_config
        sub_builder.node_id_to_idx = {}
        sub_builder.node_id_to_data = {}
        sub_builder._index_nodes()

        return sub_builder.build()

    @property
    def num_nodes(self) -> int:
        return len(self.config["nodes"])

    @property
    def num_edges(self) -> int:
        return len(self.config["edges"])

    @property
    def metadata(self) -> dict:
        return self.config.get("metadata", {})
