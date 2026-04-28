# ChainGuard AI Project Explanation

This file is a future-reference guide for how this project is structured, what each part does, and how the pieces work together at runtime. It is based on the current repository implementation.

## 1. What This Project Is

ChainGuard AI is a supply-chain resilience demo. It simulates an India-wide logistics network with factories, ports, warehouses, retailers, transport routes, disruptions, live traffic signals, and a dashboard for operators.

The main idea is:

1. A supply-chain environment models inventory, demand, cost, CO2, routes, and disruptions.
2. Each warehouse acts as an agent that decides how much to order and how to allocate outbound fulfillment.
3. A reinforcement-learning policy proposes actions.
4. A deterministic safety shield checks and modifies unsafe actions before the environment accepts them.
5. The dashboard shows topology, state, traffic, disruptions, endpoint health, shield interventions, and signal resolution.
6. Optional Gemini and Neo4j integration turns unstructured text alerts into grounded operational information.

The current app can run even without a trained model, Gemini, Neo4j, or Google Maps. When optional systems are unavailable, it falls back where possible.

## 2. Top-Level Repository Map

```text
.
|-- README.md
|-- requirements.txt
|-- .env.example
|-- agents/
|   |-- gnn_mappo.py
|   |-- shield.py
|   |-- adversary.py
|   |-- conformal.py
|-- env/
|   |-- graph_builder.py
|   |-- supply_chain_env.py
|   |-- topology_configs/
|       |-- india_50_nodes.yaml
|-- kg/
|   |-- gemini_extractor.py
|   |-- neo4j_client.py
|   |-- signal_resolver.py
|   |-- schema.cypher
|-- serving/
|   |-- local_server.py
|-- training/
|   |-- train_mappo.py
|   |-- train_adversary.py
|   |-- vertex_job.yaml
|-- dashboard/
|   |-- package.json
|   |-- vite.config.js
|   |-- index.html
|   |-- src/
|       |-- App.jsx
|       |-- main.jsx
|       |-- styles.css
|-- models/
|-- runs/
```

The Python backend lives in `serving/`, `env/`, `agents/`, `kg/`, and `training/`. The frontend lives in `dashboard/`.

## 3. Runtime Flow

The normal local demo path starts in `serving/local_server.py`.

1. Flask starts on port `5000`.
2. The server loads `.env`.
3. It builds `SupplyChainEnv` using `env/topology_configs/india_50_nodes.yaml`.
4. It identifies warehouse nodes as agents.
5. It creates a `GNNActor` policy.
6. It looks in `models/` for the newest `*_actor.pt` checkpoint.
7. If a compatible checkpoint exists, it loads that actor. If not, it uses the untrained random policy.
8. It creates a `SafetyShield`.
9. It creates a `ConformalEscalation` object. In the current local server this is not calibrated, so uncertainty escalation is fail-safe.
10. It serves API endpoints used by the dashboard.
11. It serves either `dashboard/dist` if a production frontend build exists, or the raw `dashboard` directory.

The dashboard calls backend APIs to fetch config, topology, state, and to run actions such as step, reset, disrupt, traffic update, and signal resolve.

## 4. Supply Chain Topology

The topology is defined in:

```text
env/topology_configs/india_50_nodes.yaml
```

It contains:

- `metadata`: name, description, region.
- `nodes`: factories, warehouses, ports, and retailers.
- `edges`: transport links between nodes.
- optional safety or domain metadata such as hazmat zones.

The topology is currently described as an India 50-node baseline with 72 edges in metadata. The backend includes a quality-flag check that compares the metadata expectation against the actual node and edge counts loaded from YAML. If the numbers do not match, the dashboard shows a warning under "Quality flags".

Node types mean:

- `factory`: source of supply.
- `port`: logistics import/export or transfer point.
- `warehouse`: decision-making agent node.
- `retailer`: demand endpoint.

Edge fields include route mode, distance, lead time, cost per unit, and CO2 per kilometer. Road edges can be affected by live traffic multipliers.

## 5. Graph Builder

File:

```text
env/graph_builder.py
```

`GraphBuilder` loads the YAML topology and creates lookup tables:

- node ID to typed local index.
- node ID to full node data.
- edges from each node.
- edges to each node.

It can also build a PyTorch Geometric `HeteroData` graph. The graph is heterogeneous because node types and edge types are different. For example, factories, warehouses, ports, and retailers each have their own feature matrices, and edges are grouped by source type, transport mode, and destination type.

Important behavior:

- Disabled edges are skipped when graph data is built.
- Capacity multipliers modify node capacity features.
- Lead-time and traffic multipliers modify edge features.
- Road traffic changes effective lead time, cost, and CO2.
- Coordinates are included as normalized node features.

Even though the current local actor uses flat observations, the graph builder is still important for topology management, future GNN critic/actor work, and map data.

## 6. Environment

File:

```text
env/supply_chain_env.py
```

`SupplyChainEnv` is a PettingZoo `ParallelEnv`. Each warehouse is an agent.

### Agents

Agents are all warehouse node IDs returned by:

```python
graph_builder.get_all_agent_ids()
```

That means warehouses such as `W01`, `W02`, etc. independently choose actions at each step.

### Observation

Each agent gets a flat numeric observation vector containing:

- normalized inventory.
- normalized cumulative cost.
- normalized cumulative CO2.
- current step progress.
- inbound supply availability.
- outbound route active/disabled status.
- outbound traffic pressure.
- time remaining.

The observation is padded to a consistent size based on the maximum inbound and outbound connections among all warehouse agents.

### Action

Each agent outputs a continuous vector in `[0, 1]`.

The vector is split into:

- inbound order quantities for supplier routes.
- outbound route allocation fractions for retailer/customer routes.

The number of inbound and outbound slots is also padded to the global max so all agents share the same action shape.

### Step Logic

On each environment step:

1. Current timestep increases.
2. Retailer demand is generated stochastically.
3. Optional stochastic traffic may refresh.
4. Each warehouse action is parsed.
5. Inbound orders add inventory and accumulate cost/CO2.
6. Outbound allocations fulfill demand and reduce inventory.
7. Revenue is earned from fulfilled demand.
8. Storage cost and stockout penalty are applied.
9. Reward is calculated as revenue minus costs and penalties.
10. New observations and info dictionaries are returned.

The episode ends when `current_step >= max_steps`.

### Disruptions

The environment stores disruptions in:

```python
self.disruptions
```

Supported disruption categories:

- `disabled_edges`: route closures.
- `capacity_multipliers`: source/warehouse capacity changes.
- `lead_time_multipliers`: route delays.
- `demand_multipliers`: demand shocks.
- `traffic_multipliers`: live or synthetic road congestion.

`inject_disruption()` merges new disruption data into the current disruption state. `clear_disruptions()` resets disruption state.

### Traffic

`update_traffic()` accepts multipliers keyed by `"FROM->TO"`. It only accepts valid edges, clamps values to a safe range, and stores them in `traffic_multipliers`.

Effective edge metrics are calculated in `_effective_edge_metrics()`:

- road traffic multiplies lead time.
- congestion increases cost.
- congestion increases CO2 more gently than cost.
- non-road edges ignore traffic multipliers.

## 7. Policy Model

File:

```text
agents/gnn_mappo.py
```

The main inference policy used by the local server is `GNNActor`.

Despite the filename and comments describing GNN-MAPPO, the current actor used in local inference is a flat-observation neural policy:

1. Observation vector goes through an MLP encoder.
2. Policy head outputs action means through a sigmoid, so means are in `[0, 1]`.
3. A learned `log_std` defines a Normal distribution.
4. Sampled actions are clamped to `[0, 1]`.

The actor is shared across all warehouse agents. Each agent receives its own observation, but all use the same network weights.

The file also contains:

- `GNNEncoder`: a GAT-based encoder for homogeneous subgraphs.
- `HeteroGNNCritic`: a heterogeneous graph critic for full graph value estimation.
- `GNNCritic`: a simpler flat-observation critic.

The training script currently uses the flat actor and a flat critic path.

## 8. Safety Shield

File:

```text
agents/shield.py
```

The safety shield is deterministic. It runs after the policy proposes an action and before the environment executes the action.

The local server calls:

```python
safe_action, intervened, event = shield.filter(...)
```

The shield checks:

1. Projected CO2 cap.
2. Projected budget cap.
3. Hazmat routing.
4. Over-ordering.

If a constraint would be violated, the shield modifies the action and returns a `ShieldEvent`. The dashboard displays recent shield interventions.

Important point: the shield does not learn. It is a hard safety layer that can protect the system even when the RL policy is untrained or wrong.

## 9. Conformal Escalation

File:

```text
agents/conformal.py
```

`ConformalEscalation` is intended to estimate uncertainty and decide when a human operator should be involved.

If calibrated, it can use MAPIE to produce distribution-free prediction intervals around value/reward estimates. If MAPIE is unavailable, it has a residual-based fallback.

In the current local server, it is constructed without calibration data:

```python
conformal = ConformalEscalation()
```

That means:

- `is_calibrated` is false.
- `should_escalate()` returns fail-safe escalation.
- interval width is infinite.
- threshold is zero.

This is correct for a safety-first demo, but future production work should calibrate it with held-out rollout data.

## 10. Adversary

Files:

```text
agents/adversary.py
training/train_adversary.py
```

The adversary is a policy that learns how to stress the protagonist supply-chain policy.

It can generate:

- demand multipliers.
- disabled edges.
- capacity multipliers.

It has a disruption budget so it cannot simply break everything. The goal is to discover targeted vulnerabilities.

`training/train_adversary.py` loads the latest trained protagonist actor if available, freezes it, then trains the adversary using a zero-sum objective:

```text
adversary reward = negative protagonist reward
```

So if the protagonist performs badly under a disruption, the adversary gets rewarded.

## 11. Training

Main protagonist training file:

```text
training/train_mappo.py
```

This trains the shared actor and critic with PPO-style updates:

- rollout collection over all warehouse agents.
- generalized advantage estimation.
- clipped policy loss.
- value loss.
- entropy bonus.
- TensorBoard logging.

Key output locations:

```text
runs/
models/
```

`runs/` stores TensorBoard logs. `models/` stores actor and critic checkpoints.

The local server automatically searches `models/` for the newest `*_actor.pt`.

## 12. Backend Server

File:

```text
serving/local_server.py
```

The Flask server is the integration point between environment, model, shield, Gemini/KG, and dashboard.

### Important Startup Objects

- `env`: `SupplyChainEnv`.
- `agents`: warehouse agent IDs.
- `actor`: `GNNActor`.
- `shield`: `SafetyShield`.
- `conformal`: `ConformalEscalation`.
- `_resolver`: lazily created `SignalResolver`.
- `current_obs`: latest observations for all agents.

### Static Serving

The server serves:

1. `dashboard/dist` if a built frontend exists.
2. otherwise `dashboard`.

For Vite development, the dashboard can also be run separately with `npm run dev`.

### API Endpoints

#### `GET /api/config`

Returns browser-safe runtime config:

- Maps API key.
- Google Maps map ID.
- whether Maps is enabled.
- topology config name.
- traffic runtime label.

#### `GET /api/topology`

Returns nodes, edges, metadata, and quality flags for map rendering.

Each edge includes base metrics and effective metrics:

- distance.
- lead time.
- cost.
- CO2.
- traffic multiplier.
- effective lead time.
- effective cost.
- effective CO2.

#### `GET /api/state`

Returns current simulation state:

- current step and max steps.
- inventory.
- revenue.
- CO2.
- cost.
- total revenue.
- total CO2.
- total cost.
- disruptions.
- traffic multipliers.
- quality flags.

#### `POST /api/step`

Runs one policy-controlled environment step:

1. If an episode is done, reset environment and shield.
2. Convert observations to a tensor.
3. Actor samples actions.
4. Safety shield filters each action.
5. Environment steps with safe actions.
6. Conformal uncertainty check runs.
7. Response returns rewards, totals, traffic, disruptions, shield events, and uncertainty.

#### `POST /api/traffic`

Accepts route traffic observations from the dashboard/Google Maps. It converts duration-in-traffic versus base duration into traffic multipliers and updates the environment.

#### `POST /api/disrupt`

Injects a disruption. If no custom disruption is provided, it uses a default demo disruption.

#### `POST /api/resolve`

Runs Gemini plus Neo4j signal resolution on raw text.

If Gemini or Neo4j is not configured, the server returns a fallback result instead of crashing.

#### `POST /api/reset`

Resets the environment and safety shield.

## 13. Knowledge Graph and Gemini Flow

Files:

```text
kg/gemini_extractor.py
kg/neo4j_client.py
kg/signal_resolver.py
kg/schema.cypher
```

The purpose of this layer is to turn unstructured alert text into structured operational information.

Example input:

```text
Flood at Mumbai port, truck strike in Pune, power outage at Delhi NCR warehouse
```

Flow:

1. `SignalResolver.resolve()` receives raw text.
2. `GeminiEntityExtractor.extract()` asks Gemini to return JSON entities.
3. For each entity, `KnowledgeGraphClient` queries Neo4j.
4. The resolver attaches protocols and affected locations if the entity is grounded.
5. It returns entity results and an overall confidence score.

Gemini extraction is probabilistic but configured with temperature `0` and JSON MIME type. Neo4j grounding is deterministic.

Required environment variables include:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_TIMEOUT_SECONDS`
- `GEMINI_MAX_RETRIES`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

Only `GEMINI_API_KEY` is checked before lazy-loading the resolver in the local server. Neo4j failures are caught inside resolution.

## 14. Dashboard

Main files:

```text
dashboard/src/App.jsx
dashboard/src/main.jsx
dashboard/src/styles.css
dashboard/package.json
dashboard/vite.config.js
```

The dashboard is a React app built with Vite.

It shows:

- header metrics for revenue, CO2, step, and road route coverage.
- command buttons for step, reset, and disruption.
- Google Maps logistics overlay.
- selected route details.
- backend endpoint status.
- topology quality flags.
- shield intervention feed.
- Gemini/KG signal resolver panel.

### Frontend API Base

`App.jsx` uses:

```javascript
const API_BASE = window.location.origin.startsWith("http") ? window.location.origin : "";
```

This means if the dashboard is served from Flask at `http://localhost:5000`, API calls go to the same origin. If opened as a file, it falls back to relative paths, which may not work for all API calls.

Best local path: run the Flask server and open `http://localhost:5000`.

### Polling

The dashboard polls `/api/state` every 3 seconds.

### Map Rendering

`MapsPanel` loads Google Maps only if `/api/config` says Maps is enabled and a Maps API key exists.

It renders:

- markers for factories, warehouses, ports, and retailers.
- curved lane overlays for backbone routes.
- route colors based on mode, disruption, and congestion.
- map legends.

Only selected backbone lanes are shown by default. Retailer last-mile routes still exist in the simulation but are visually hidden unless disrupted.

### Traffic Sync

When Google Maps is available:

1. The dashboard samples up to `TRAFFIC_ROUTE_LIMIT` road edges.
2. It asks Google Directions for driving duration and duration in traffic.
3. It sends observations to `POST /api/traffic`.
4. The backend converts them into multipliers.
5. Future environment steps use those multipliers in edge metrics.

This connects real-world road congestion into the simulation loop.

### Endpoint Status

Every API request records:

- HTTP status.
- elapsed time.
- timestamp.
- optional message.

The sidebar uses this to show backend health.

## 15. Environment Variables

The current code expects `.env` at the repo root. `.env.example` should be used as the template.

Important variables:

- `MAPS_API_KEY`: enables Google Maps in the dashboard.
- `GOOGLE_MAPS_MAP_ID`: optional Google Maps styling ID.
- `GEMINI_API_KEY`: enables Gemini extraction.
- `GEMINI_MODEL`: defaults to `gemini-2.5-flash`.
- `GEMINI_TIMEOUT_SECONDS`: defaults to `20`.
- `GEMINI_MAX_RETRIES`: defaults to `2`.
- `NEO4J_URI`: defaults to `bolt://localhost:7687`.
- `NEO4J_USER`: defaults to `neo4j`.
- `NEO4J_PASSWORD`: needed for Neo4j.
- `TOTAL_TIMESTEPS`: optional training override for `train_mappo.py`.

## 16. How To Run Locally

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install dashboard dependencies:

```bash
cd dashboard
npm install
```

Run the Flask demo server:

```bash
python serving/local_server.py
```

Open:

```text
http://localhost:5000
```

For frontend development:

```bash
cd dashboard
npm run dev
```

If using Vite separately, ensure API requests can reach the Flask backend. The current dashboard is simplest when served by Flask from the same origin.

## 17. How To Train

Train protagonist policy:

```bash
python training/train_mappo.py
```

Train adversary:

```bash
python training/train_adversary.py
```

View TensorBoard logs:

```bash
tensorboard --logdir runs
```

After training, actor checkpoints appear in `models/`. The local server will load the newest `*_actor.pt` automatically if it is compatible with the current observation and action shapes.

If the topology or observation structure changes, old checkpoints may no longer load. The server catches incompatible actor shape errors and falls back to the random policy.

## 18. Important Data Contracts

### Disruption Object

Typical disruption shape:

```json
{
  "disabled_edges": [["P01", "W01"]],
  "capacity_multipliers": {"F01": 0.5},
  "demand_multipliers": {"R03": 2.0},
  "lead_time_multipliers": {"W01->R01": 1.5},
  "traffic_multipliers": {"F01->W01": 1.3}
}
```

Not all keys are required.

### Traffic Observation

Dashboard sends:

```json
{
  "observations": [
    {
      "edge_key": "F01->W01",
      "base_duration_seconds": 3600,
      "traffic_duration_seconds": 4500
    }
  ]
}
```

Backend converts this to:

```text
traffic_multiplier = traffic_duration_seconds / base_duration_seconds
```

### Signal Resolve Request

```json
{
  "text": "Flood near Mumbai warehouse, road transport delayed"
}
```

### Signal Resolve Response

The response contains:

- raw text.
- extracted entities.
- grounding flag.
- protocols.
- affected locations.
- overall confidence.

## 19. What Is Real Versus Prototype

Real implementation pieces:

- PettingZoo environment.
- route/inventory/cost/CO2 simulation.
- traffic multiplier integration.
- safety shield intervention.
- Flask API.
- React dashboard.
- Gemini JSON extraction path.
- Neo4j grounding path.
- PPO-style training script.
- adversary training script.

Prototype or simplified pieces:

- The local inference actor uses flat observations, not full graph message passing.
- The conformal escalation object is not calibrated in the server.
- The dashboard map overlay is schematic, not exact road geometry for every route.
- If no trained actor checkpoint is available, behavior is from a random untrained policy.
- Gemini/Neo4j are optional at local runtime and have fallback behavior.

## 20. Common Future Changes

### Add a New Node or Route

Edit `env/topology_configs/india_50_nodes.yaml`.

Then check:

- node ID uniqueness.
- node type is one of the supported types.
- required fields exist for that type.
- edges reference valid node IDs.
- metadata node/edge counts match actual counts.

If new topology changes max inbound/outbound connections, old actor checkpoints may become incompatible.

### Change Agent Behavior

Start with:

```text
agents/gnn_mappo.py
training/train_mappo.py
env/supply_chain_env.py
```

Observation or action shape changes require coordinated updates in environment, model, training, and possibly checkpoint handling.

### Add a Safety Rule

Update:

```text
agents/shield.py
```

Then make sure the server response still serializes shield events cleanly for the dashboard.

### Add a Dashboard Panel

Update:

```text
dashboard/src/App.jsx
dashboard/src/styles.css
serving/local_server.py
```

If the panel needs new backend data, add a new endpoint or extend an existing response carefully.

### Add More KG Protocols

Update:

```text
kg/schema.cypher
```

Then load it into Neo4j and verify `KnowledgeGraphClient` queries return the expected records.

## 21. Debugging Guide

### Dashboard Loads But Map Says API Key Missing

Check `.env`:

```text
MAPS_API_KEY=...
```

Restart Flask after editing `.env`.

### Gemini Resolve Returns Fallback

Check:

```text
GEMINI_API_KEY
GEMINI_MODEL
```

Also check that network/API access is available and the key has permission.

### Gemini Extracts But No Protocols Appear

The entity may not exist in Neo4j with the same canonical name. Check `kg/schema.cypher` and Neo4j data.

### Model Does Not Load

The server prints a warning when actor weights are incompatible. This usually means the observation or action dimension changed after training.

Retrain with:

```bash
python training/train_mappo.py
```

### Quality Flags Show Topology Mismatch

Compare YAML metadata with actual counts:

- `metadata.description`
- `metadata.name`
- number of `nodes`
- number of `edges`

Fix either metadata or topology data.

### Step Button Works But Results Look Random

That usually means no trained actor was loaded. Check `models/` for a compatible `*_actor.pt`.

## 22. Mental Model For The Whole System

Think of the project as four layers:

1. **World model**: YAML topology plus `SupplyChainEnv`.
2. **Decision layer**: actor policy, adversary, conformal uncertainty.
3. **Safety and grounding layer**: shield, Gemini, Neo4j.
4. **Operator layer**: Flask APIs and React dashboard.

The most important runtime loop is:

```text
dashboard Step button
-> POST /api/step
-> actor proposes actions
-> shield filters actions
-> environment updates inventory/cost/CO2/revenue
-> conformal uncertainty check runs
-> API returns new state
-> dashboard refreshes metrics/map/sidebar
```

The most important data loop for traffic is:

```text
Google Maps route sample
-> dashboard computes base duration and traffic duration
-> POST /api/traffic
-> backend stores traffic multipliers
-> environment applies multipliers to road edge metrics
-> future steps reflect congestion
```

The most important data loop for text alerts is:

```text
operator text alert
-> POST /api/resolve
-> Gemini extracts structured entities
-> Neo4j grounds entities to protocols/locations
-> dashboard displays operational interpretation
```

## 23. Practical Notes

- The backend is stateful. API calls mutate the global `env`, `current_obs`, and shield state.
- This is fine for a local demo but would need session isolation for multi-user deployment.
- The frontend assumes backend responses are JSON.
- The trained policy is loaded once at server startup.
- Traffic updates rebuild current observations without advancing the environment.
- Reset clears environment state and shield logs but does not reload model weights.
- Disruption injection does not automatically step the environment; it changes future state transitions.
- The shield can intervene frequently if the policy is random or if caps are strict.

## 24. Recommended Next Improvements

1. Calibrate `ConformalEscalation` with held-out rollout data and expose real uncertainty metrics in the dashboard.
2. Align the local actor implementation with the full graph/GNN design, or update naming/comments to make the flat actor explicit.
3. Add tests for environment stepping, traffic multiplier effects, disruption merging, and shield filtering.
4. Add an API endpoint for clearing disruptions without a full reset.
5. Add checkpoint metadata so the server can verify observation/action dimensions before attempting load.
6. Make topology quality checks data-driven instead of inferring expected counts from metadata name.
7. Add session IDs if multiple dashboard users will use the server at the same time.

