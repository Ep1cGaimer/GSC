"""
GNN-MAPPO: Graph Neural Network actor-critic for multi-agent supply chain RL.

Actor: 2-layer GATConv on local k-hop subgraph → policy head
Critic: 3-layer HeteroConv (GATConv per edge type) on full graph → value head

The GNN backbone handles variable-size topologies natively —
the same weights process 40, 50, or 75 node graphs without retraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch_geometric.nn import GATConv, HeteroConv, global_mean_pool
from torch_geometric.data import HeteroData, Batch


class GNNEncoder(nn.Module):
    """Shared GNN encoder using GATConv for homogeneous subgraph encoding.
    
    Used by the decentralized actor on local k-hop subgraphs.
    Attention weights learn which neighbors matter (sole-source supplier
    vs. one-of-many), which is critical for hub-and-spoke supply chains.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 128, heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, concat=False))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=False))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Output projection
        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge index [2, num_edges]
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
        return self.out_proj(x)


class HeteroGNNCritic(nn.Module):
    """Heterogeneous GNN critic for centralized value estimation.
    
    Uses HeteroConv to handle different node types (factory, warehouse, port, retailer)
    and edge types (road_to, rail_to). Processes the FULL global graph.
    
    Only used during training (CTDE paradigm).
    """

    def __init__(self, node_type_dims: dict, hidden_channels: int = 128,
                 out_channels: int = 256, heads: int = 4, num_layers: int = 3):
        super().__init__()

        # Input projections per node type (different feature dimensions)
        self.input_projs = nn.ModuleDict()
        for ntype, dim in node_type_dims.items():
            self.input_projs[ntype] = nn.Linear(dim, hidden_channels)

        # Heterogeneous conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # We'll build the actual HeteroConv dynamically in forward
            # since edge types depend on the input graph
            self.convs.append(None)  # Placeholder — built in _build_conv
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads

        # Output MLP
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Store dynamically built convs — use ModuleDict so parameters are tracked
        self._conv_cache = nn.ModuleDict()

    def _get_conv(self, edge_types, device):
        """Build or retrieve cached HeteroConv for given edge types."""
        # Use a stable string key for nn.ModuleDict compatibility
        str_key = "|".join(f"{s}__{r}__{d}" for s, r, d in sorted(edge_types))
        if str_key not in self._conv_cache:
            conv_dict = {}
            for et in edge_types:
                conv_dict[et] = GATConv(
                    self.hidden_channels, self.hidden_channels,
                    heads=self.heads, concat=False
                ).to(device)
            self._conv_cache[str_key] = HeteroConv(conv_dict, aggr="mean")
        return self._conv_cache[str_key]

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Args:
            data: HeteroData with typed nodes and edges
        Returns:
            Global value estimate [1]
        """
        # Project each node type to shared hidden dimension
        x_dict = {}
        for ntype in data.node_types:
            if hasattr(data[ntype], 'x') and data[ntype].x is not None:
                if ntype in self.input_projs:
                    x_dict[ntype] = torch.relu(self.input_projs[ntype](data[ntype].x))
                else:
                    # Unknown node type — project with zero-initialized linear
                    dim = data[ntype].x.shape[-1]
                    proj = nn.Linear(dim, self.hidden_channels).to(data[ntype].x.device)
                    nn.init.zeros_(proj.weight)
                    self.input_projs[ntype] = proj
                    x_dict[ntype] = torch.relu(proj(data[ntype].x))

        # Message passing
        edge_types = data.edge_types
        for i in range(self.num_layers):
            conv = self._get_conv(edge_types, next(iter(x_dict.values())).device)
            x_dict = conv(x_dict, data.edge_index_dict)
            # Apply norm and ReLU
            for ntype in x_dict:
                x_dict[ntype] = torch.relu(x_dict[ntype])

        # Global pooling: mean across all node types
        all_embeddings = []
        for ntype, x in x_dict.items():
            all_embeddings.append(x.mean(dim=0))

        global_embedding = torch.stack(all_embeddings).mean(dim=0)  # [hidden_channels]

        return self.value_head(global_embedding).squeeze(-1)


class GNNActor(nn.Module):
    """Actor with optional GNN encoder on local subgraph.

    Two modes:
      - Flat: MLP encoder on flat observation vector (backward compatible)
      - Graph: GNNEncoder on homogeneous k-hop subgraph around agent

    Set use_graph=True and pass graph_obs dict to use GNN path.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 embed_dim: int = 128, gnn_heads: int = 4, use_graph: bool = False):
        super().__init__()
        self.use_graph = use_graph

        if use_graph:
            self.gnn_encoder = GNNEncoder(
                in_channels=5, hidden_channels=hidden_dim,
                out_channels=embed_dim, heads=gnn_heads
            )
            self.obs_encoder = None
        else:
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Tanh(),
            )
            self.gnn_encoder = None

        self.alpha_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, graph_obs=None):
        """
        Args:
            obs: Flat observation tensor [batch, obs_dim] (ignored if use_graph)
            graph_obs: For graph mode: dict with 'x', 'edge_index', 'center_idx'
                       or list of such dicts for batched agents
        Returns:
            alpha, beta
        """
        if self.use_graph and graph_obs is not None:
            if isinstance(graph_obs, list):
                embeddings = []
                for go in graph_obs:
                    x = go["x"]  # [num_nodes, 5]
                    ei = go["edge_index"]  # [2, num_edges]
                    center = go["center_idx"]
                    if x.shape[0] == 0:
                        embeddings.append(torch.zeros(1, self.alpha_head[0].in_features,
                                                       device=x.device if hasattr(x, 'device') else 'cpu'))
                        continue
                    node_embs = self.gnn_encoder(x, ei)  # [num_nodes, embed_dim]
                    embeddings.append(node_embs[center].unsqueeze(0))
                embedding = torch.cat(embeddings, dim=0)
            else:
                x = graph_obs["x"]
                ei = graph_obs["edge_index"]
                center = graph_obs["center_idx"]
                node_embs = self.gnn_encoder(x, ei)
                embedding = node_embs[center].unsqueeze(0)
        else:
            embedding = self.obs_encoder(obs)

        alpha = F.softplus(self.alpha_head(embedding)) + 1.0
        beta = F.softplus(self.beta_head(embedding)) + 1.0
        return alpha, beta

    def get_action_and_value(self, obs, action=None, action_mask=None,
                              deterministic=False, graph_obs=None):
        """Get a bounded action from policy and compute log probability."""
        alpha, beta = self.forward(obs, graph_obs=graph_obs)
        dist = Beta(alpha, beta)

        if action is None:
            if deterministic:
                action = alpha / (alpha + beta)
            else:
                action = dist.rsample()

        action = torch.clamp(action, 1e-6, 1.0 - 1e-6)
        if action_mask is None:
            valid_mask = torch.ones_like(action)
        else:
            valid_mask = action_mask.to(action.device).float()

        # For masked (invalid) dims, use the Beta mean as a safe value
        # to avoid NaN/Inf in log_prob that could corrupt gradients
        safe_action = torch.where(
            valid_mask > 0.5,
            action,
            (alpha / (alpha + beta)).detach(),
        )
        log_prob = (dist.log_prob(safe_action) * valid_mask).sum(-1)
        entropy = (dist.entropy() * valid_mask).sum(-1)
        action = action * valid_mask

        return action, log_prob, entropy


class GNNCritic(nn.Module):
    """Centralized critic using flat observation (simplified for prototype).
    
    For full graph-based critic, use HeteroGNNCritic.
    This simpler version takes concatenated observations from all agents.
    """

    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * num_agents, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, all_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_obs: Concatenated observations [batch, obs_dim * num_agents]
        Returns:
            Value estimate [batch, 1]
        """
        return self.net(all_obs)
