# ChainGuard AI - PowerPoint Presentation Instructions

This file provides detailed content and structure for creating a comprehensive PowerPoint presentation about ChainGuard AI.

---

## SLIDE 1: Title Slide

**Title:** ChainGuard AI — Resilient Supply Chain Intelligence

**Subtitle:** Multi-Agent Reinforcement Learning for Supply Chain Resilience

**Event:** Google Solution Challenge 2026

**Tags:** SDGs 9 (Industry, Innovation, Infrastructure), 12 (Responsible Consumption), 13 (Climate Action)

**Visual:** Logo placeholder + background image of supply chain/logistics network

---

## SLIDE 2: The Problem

**Headline:** Global Supply Chains Are Vulnerable

**Key Points:**
- Black-swan events (pandemics, wars, natural disasters)
- Climate shocks increasing in frequency and intensity
- Existing AI solutions fail when network topology changes
- Oscillation under safety constraints
- Lack of explainability ("why did this happen?")

**Statistics to mention (optional):**
- X% of companies experienced supply chain disruptions in 2024
- Average cost of supply chain disruption: $X million

**Visual:** Infographic showing types of supply chain disruptions

---

## SLIDE 3: The Solution Overview

**Headline:** ChainGuard AI - Multi-Agent RL System

**Core Capabilities:**
1. **GNN-MAPPO**: Graph Neural Network-based RL that adapts to dynamic network shapes without retraining
2. **Safety Shield**: Mathematical action masking guaranteeing 0 violations of sustainability (CO2) and budget caps
3. **Adversarial Robustness**: Minimax adversary that proactively finds vulnerabilities
4. **Gemini & KG Grounding**: Deterministic grounding of unstructured alerts into operational protocols
5. **Conformal Prediction**: Single-pass uncertainty quantification for human-in-the-loop escalation

**Visual:** System architecture diagram showing all 5 components

---

## SLIDE 4: Technical Architecture - High Level

**Headline:** Four-Layer System Architecture

**Layer 1: World Model**
- YAML topology configuration
- SupplyChainEnv (PettingZoo ParallelEnv)
- Tracks: inventory, costs, CO2 emissions, demand fulfillment

**Layer 2: Decision Layer**
- GNNActor policy (neural network)
- Adversary policy (minimax)
- Conformal uncertainty estimation

**Layer 3: Safety & Grounding**
- Safety Shield (hard constraints)
- Gemini (entity extraction)
- Neo4j Knowledge Graph (grounding)

**Layer 4: Operator Interface**
- Flask REST API
- React Dashboard with Google Maps

**Visual:** Layered architecture diagram

---

## SLIDE 5: Topology & Simulation

**Headline:** India-Wide Logistics Network Simulation

**Current Topology (india_50_nodes.yaml):**
- **40 nodes** across 4 types:
  - Factories (F01-F08): Source of supply
  - Ports (P01-P03): Import/export points
  - Warehouses (W01-W10): Decision-making agents
  - Retailers (R01-R19): Demand endpoints
- **44 edges** (transport routes)
  - Road, Rail, Sea modes
  - Each edge has: distance, lead time, cost/unit, CO2/km

**Simulation Details:**
- Max steps per episode: 100
- Stochastic demand generation
- Optional live traffic integration (Google Maps)
- Disruption types: disabled edges, capacity changes, delays, demand shocks

**Visual:** Map of India showing node locations with colored edges

---

## SLIDE 6: Multi-Agent Reinforcement Learning

**Headline:** GNN-MAPPO Agent Architecture

**Algorithm:** Multi-Agent Proximal Policy Optimization (MAPPO)

**Agent Setup:**
- Each warehouse (W01-W10) is an independent agent
- Shared policy network across all agents
- Flat observation vector per agent:
  - Normalized inventory
  - Cumulative cost & CO2
  - Current step progress
  - Inbound supply availability
  - Outbound route status
  - Traffic pressure
  - Time remaining

**Action Space:**
- Continuous vector in [0, 1]
- Split into:
  - Inbound order quantities (from factories/ports)
  - Outbound allocation fractions (to retailers)

**Training Configuration:**
- Rollout steps: 200
- Minibatches: 8
- Epochs per update: 5
- Gamma: 0.99
- GAE Lambda: 0.95
- Clip range: 0.2

**Visual:** Agent decision flow diagram

---

## SLIDE 7: Training Process

**Headline:** PPO-Style Training Pipeline

**Training Loop:**
1. Collect rollouts from all agents
2. Compute Generalized Advantage Estimation (GAE)
3. Update policy with clipped objective
4. Value function loss
5. Entropy bonus for exploration
6. Log to TensorBoard

**Hyperparameters:**
- Learning rate: 3e-4 → 5e-5 (linear decay)
- Entropy: 0.05 → 0.001 (annealed over 100K steps)
- Total timesteps: 1,000,000 (default)
- Reward scale: 1e-3

**Output:**
- Checkpoints saved to `models/` directory
- TensorBoard logs in `runs/` directory

**Visual:** Training pipeline diagram with rollout collection, advantage computation, and policy update steps

---

## SLIDE 8: Safety Shield

**Headline:** Mathematical Safety Guarantees

**Shield Function:**
- Runs AFTER policy proposes actions
- Runs BEFORE environment executes actions
- Deterministic (does not learn)

**Constraints Enforced:**
1. **CO2 Cap**: Projected emissions must stay below threshold
2. **Budget Cap**: Projected costs must stay below threshold
3. **Hazmat Routing**: Certain goods cannot use specific routes
4. **Over-ordering**: Prevents ordering more than capacity allows

**Output:**
- `safe_action`: Modified action if needed
- `intervened`: Boolean flag
- `event`: Description of intervention

**Key Point:** Guarantees 0 violations even when RL policy is untrained or wrong

**Visual:** Shield filtering flow diagram

---

## SLIDE 9: Adversarial Robustness

**Headline:** Proactive Stress Testing

**Purpose:** Discover system vulnerabilities before real disruptions occur

**Adversary Capabilities:**
- Generate demand multipliers (demand shocks)
- Generate disabled edges (route closures)
- Generate capacity multipliers (supply disruptions)

**Constraints:**
- Disruption budget (cannot break everything)
- Targets specific vulnerabilities

**Training:**
- Zero-sum objective: adversary reward = -protagonist reward
- If protagonist performs poorly → adversary gets rewarded

**Visual:** Adversary attack scenarios diagram

---

## SLIDE 10: Knowledge Graph & Gemini

**Headline:** From Text Alerts to Operations

**Problem:** Unstructured alerts like "Flood at Mumbai Port" need to become operational data

**Pipeline:**
1. **Input:** Raw text alert
2. **Gemini Extraction:** JSON entities from unstructured text
   - Temperature: 0 (deterministic)
   - Output: structured entities
3. **Neo4j Grounding:** Match entities to knowledge graph
   - Schema includes: nodes, locations, protocols
4. **Resolution:** Attach protocols and affected locations

**Example:**
- Input: "Flood at Mumbai port, truck strike in Pune"
- Output: Grounded entities with operational protocols

**Required APIs:**
- Gemini API (gemini-2.5-flash default)
- Neo4j database

**Visual:** Signal resolution flow diagram

---

## SLIDE 11: Conformal Prediction

**Headline:** Uncertainty Quantification

**Purpose:** Decide when human operator intervention is needed

**Method:**
- MAPIE-based prediction intervals
- Distribution-free guarantees
- Fallback: residual-based estimation

**Current State:**
- Not calibrated in demo (fail-safe mode)
- Returns "escalate" when uncertain

**Future:** Calibrate with held-out rollout data

**Visual:** Prediction interval diagram showing safe vs. escalate zones

---

## SLIDE 12: Dashboard & Visualization

**Headline:** Operator Interface

**Features:**
1. **Header Metrics**: Revenue, CO2, current step, road route coverage
2. **Control Panel**: Step, Reset, Disrupt buttons
3. **Google Maps Overlay**:
   - Nodes: factories (factory), warehouses (warehouse), ports (📦), retailers (🏪)
   - Edges: colored by mode (road=blue, rail=gray, sea=purple)
   - Disruptions: red dashed lines
   - Live traffic: congestion coloring
4. **Route Details**: Selected route metrics
5. **Endpoint Status**: API health monitoring
6. **Quality Flags**: Topology validation warnings
7. **Shield Feed**: Recent interventions
8. **Signal Resolver**: Gemini/KG integration panel

**Tech Stack:**
- React + Vite
- Google Maps JavaScript API
- Real-time polling (3-second intervals)

**Visual:** Screenshot of dashboard

---

## SLIDE 13: Live Traffic Integration

**Headline:** Connecting Real-World Data

**Process:**
1. Dashboard samples road edges (up to unlimited with caching)
2. Google Directions API queries:
   - Base duration (no traffic)
   - Traffic duration (current conditions)
3. Compute multiplier: traffic_duration / base_duration
4. Send to backend via `/api/traffic`
5. Environment applies to road edge metrics:
   - Lead time increases
   - Cost increases
   - CO2 increases (gentler)

**Visual:** Traffic integration flow diagram

---

## SLIDE 14: Disruption Handling

**Headline:** Response to Network Changes

**Disruption Types:**
1. **disabled_edges**: Route closures (e.g., [["P01", "W01"]])
2. **capacity_multipliers**: Node capacity changes (e.g., {"F01": 0.5})
3. **lead_time_multipliers**: Route delays (e.g., {"W01->R01": 1.5})
4. **demand_multipliers**: Demand shocks (e.g., {"R03": 2.0})
5. **traffic_multipliers**: Congestion (e.g., {"F01->W01": 1.3})

**Visual Demo:**
- Red dashed lines appear on disabled edges
- Banner shows: "Disabled edges: X→Y" and "Constrained nodes: ..."

**Visual:** Before/after disruption map screenshots

---

## SLIDE 15: Key Technologies

**Headline:** Technology Stack

**AI/ML:**
- PyTorch
- PyTorch Geometric (GNN)
- PettingZoo (MARL)
- MAPIE (Conformal Prediction)

**Cloud Services (Google):**
- Vertex AI (training)
- Gemini API (entity extraction)
- Google Maps API (visualization & traffic)
- Firebase (optional backend)

**Data:**
- Neo4j (Knowledge Graph)
- YAML (topology config)
- TensorBoard (logging)

**Backend:**
- Python 3.10+
- Flask

**Frontend:**
- React
- Vite

---

## SLIDE 16: Repository Structure

**Headline:** Project Organization

```
GSC/
├── README.md                    # Overview
├── PROJECT_EXPLANATION.md       # Detailed documentation
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
│
├── env/                         # Environment simulation
│   ├── supply_chain_env.py      # PettingZoo parallel environment
│   ├── graph_builder.py         # Topology & graph construction
│   └── topology_configs/
│       └── india_50_nodes.yaml  # 40-node India network config
│
├── agents/                      # AI models
│   ├── gnn_mappo.py            # Actor & Critic networks
│   ├── shield.py               # Safety constraints
│   ├── adversary.py            # Attack policy
│   └── conformal.py            # Uncertainty estimation
│
├── kg/                          # Knowledge Graph
│   ├── gemini_extractor.py    # Gemini entity extraction
│   ├── neo4j_client.py        # Neo4j queries
│   ├── signal_resolver.py     # Full resolution pipeline
│   └── schema.cypher          # Graph schema
│
├── training/                   # Training scripts
│   ├── train_mappo.py         # Protagonist training
│   ├── train_adversary.py    # Adversary training
│   └── training_utils.py     # Helper functions
│
├── serving/                    # Backend server
│   └── local_server.py        # Flask API server
│
├── dashboard/                  # Frontend React app
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── main.jsx          # Entry point
│   │   └── styles.css        # Styling
│   └── dist/                  # Production build
│
├── models/                     # Trained model checkpoints
└── runs/                      # TensorBoard logs
```

---

## SLIDE 17: How to Run Locally

**Headline:** Getting Started

**Prerequisites:**
- Python 3.10+
- Node.js (for dashboard)
- (Optional) Neo4j, Google Cloud account

**Setup Steps:**

```bash
# 1. Clone and install
git clone <repo_url>
cd chainguard-ai
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (Google Maps, Gemini if used)

# 3. Install dashboard
cd dashboard
npm install
cd ..

# 4. Run backend
python serving/local_server.py

# 5. Open in browser
# http://localhost:5000
```

**For development:**
```bash
cd dashboard
npm run dev
```

**Visual:** Terminal commands screenshot

---

## SLIDE 18: Demo Walkthrough

**Headline:** Using the System

1. **Initial State**: Empty inventory, all routes functional
2. **Click "Step"**: RL policy orders stock, fulfills demand, earns revenue
3. **Click "Auto"**: Continuous stepping with animation
4. **Click "Disrupt"**: Inject demo disruption (disabled route)
   - Map shows red dashed line
   - Banner shows affected edges/nodes
5. **Traffic Sync**: Optional Google Maps traffic integration
6. **Signal Resolve**: Enter text alert for Gemini/KG processing

**Key Metrics Tracked:**
- Revenue (total earned)
- CO2 (total emissions)
- Cost (total spending)
- Inventory levels per node

**Visual:** Step-by-step screenshots

---

## SLIDE 19: Results & Performance

**Headline:** Training Results

**Include if available:**
- Training curves (reward vs timesteps)
- Comparison: trained vs random policy
- Shield intervention statistics
- Disruption recovery metrics

**Metrics to showcase:**
- Average episode reward
- CO2 constraint violations: 0
- Budget constraint violations: 0
- Training time to convergence

**Visual:** TensorBoard graphs, performance charts

---

## SLIDE 20: Differentiation & Innovation

**Headline:** What Makes ChainGuard Unique

**Compared to Traditional Solutions:**

| Feature | Traditional | ChainGuard AI |
|---------|-------------|---------------|
| Network Adaptation | Fixed topology | Dynamic GNN |
| Safety Guarantees | Heuristic | Mathematical shield |
| Attack Resistance | Reactive | Proactive adversary |
| Explainability | Black box | KG grounded |
| Uncertainty | None | Conformal prediction |

**Key Innovations:**
1. First supply chain RL system with hard safety guarantees
2. End-to-end from unstructured text to operational response
3. Adversarial training for resilience
4. Graph-based representation for arbitrary topologies

---

## SLIDE 21: Future Work

**Headline:** Roadmap

**Short-term:**
- Calibrate conformal prediction with real data
- Add more topology configurations
- Improve GNN encoder integration

**Long-term:**
- Multi-region deployment
- Real-time API integration
- Production-grade Neo4j
- User authentication (Firebase)

**Potential Partners:**
- Logistics companies
- Supply chain managers
- Government agencies

---

## SLIDE 22: Team & Credits

**Headline:** Team

**Team Members:**
- [Name 1] - Role
- [Name 2] - Role
- [Name 3] - Role
- ...

**Acknowledgments:**
- Google Solution Challenge mentors
- University/academic advisors
- Open source projects: PettingZoo, PyTorch Geometric, React

**Contact:** [email/link to repo]

---

## SLIDE 23: Q&A

**Headline:** Questions?

**Placeholder for audience questions**

---

## Design Tips for the PowerPoint

### Color Scheme
- **Primary:** Blue (#2563eb) - trust, technology
- **Secondary:** Green (#22c55e) - sustainability, CO2
- **Alert:** Red (#b91c1c) - disruptions
- **Background:** White or light gray
- **Text:** Dark gray (#334155)

### Visual Style
- Clean, modern, professional
- Use diagrams for architecture
- Use screenshots for dashboard
- Consistent iconography
- Avoid clutter - one key message per slide

### Animation
- Use subtle transitions
- Reveal bullet points one at a time
- Animate diagrams for flow explanation

### Charts/Graphs
- Use real TensorBoard screenshots if available
- Create simple bar/line charts for metrics
- Use tables for comparisons

### Map Visuals
- Include India map with node locations
- Show before/after disruption screenshots
- Highlight traffic congestion colors

---

## Additional Content Sources

**For technical details, refer to:**
- `README.md` - Quick overview
- `PROJECT_EXPLANATION.md` - Detailed technical doc
- `agents/gnn_mappo.py` - Model architecture
- `env/supply_chain_env.py` - Environment details
- `training/train_mappo.py` - Training parameters
- `dashboard/src/App.jsx` - Dashboard implementation
- `serving/local_server.py` - API endpoints

---

## Presentation Length Recommendation

- **Total slides:** 20-25
- **Timing:** 15-20 minutes presentation + 5-10 minutes Q&A
- **Focus:** Architecture, innovation, and demo are most important
- **Skip:** Deep mathematical derivations unless specifically asked

---

*End of PPT Instructions*