# ChainGuard AI - Mermaid Diagrams

This file contains comprehensive diagrams for the ChainGuard AI system.

> You can render these diagrams in:
> - VS Code with Mermaid extension
> - GitHub (native support)
> - Mermaid Live Editor: https://mermaid.live/
> - Notion, Obsidian, etc.

---

## 1. High-Level Architecture Diagram

```mermaid
flowchart TB
    subgraph Client["Frontend - Dashboard"]
        UI["React App<br/>App.jsx"]
        Maps["Google Maps<br/>MapsPanel"]
        State["State Management"]
    end
    
    subgraph Backend["Backend - Flask Server"]
        API["REST API<br/>local_server.py"]
        Env["SupplyChainEnv<br/>supply_chain_env.py"]
        Shield["SafetyShield<br/>shield.py"]
        Actor["GNNActor<br/>gnn_mappo.py"]
        KG["Knowledge Graph<br/>Gemini + Neo4j"]
    end
    
    subgraph Data["Data Layer"]
        YAML["Topology Config<br/>india_50_nodes.yaml"]
        Models["Trained Models<br/>models/"]
        DB["Neo4j Database"]
        Cloud["Google Cloud<br/>Gemini/Vertex AI"]
    end
    
    UI -->|HTTP| API
    Maps -->|Directions API| Cloud
    API --> Env
    API --> Actor
    API --> Shield
    API --> KG
    Env --> YAML
    Actor --> Models
    KG --> DB
    KG --> Cloud
    
    style Client fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style Backend fill:#1a2f1e,stroke:#10b981,color:#fff
    style Data fill:#2d1f1f,stroke:#ef4444,color:#fff
```

---

## 2. Runtime Process Flow

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant API
    participant Env
    participant Actor
    participant Shield
    
    Dashboard->>Dashboard: User clicks "Step"
    
    Dashboard->>API: POST /api/step
    
    API->>Env: Get current observations
    Env-->>API: Return obs for all agents
    
    API->>Actor: Forward pass
    Actor-->>API: Return actions
    
    API->>Shield: Filter actions
    Shield-->>API: Return safe_actions + events
    
    API->>Env: Step(actions)
    Env-->>API: Return rewards, new state, done
    
    API-->>Dashboard: JSON {rewards, state, flows}
    
    Dashboard->>Dashboard: Update UI + animate trucks
    
    Note over User,Dashboard: Repeat until episode done
```

---

## 3. Disruption Flow

```mermaid
flowchart TB
    A[User clicks Disrupt] --> B[POST /api/disrupt]
    B --> C[Backend inject_disruption]
    C --> D{Valid disruption?}
    D -->|Yes| E[Merge into env.disruptions]
    D -->|No| F[Return error]
    E --> G[Return disruption to frontend]
    G --> H[Update state.disruptions]
    H --> I[Map re-renders]
    I --> J[Check disabled_edges]
    J --> K{Edge in topology?}
    K -->|Yes| L[Draw red dashed line]
    K -->|No| M[Skip - edge not found]
    L --> N[Add red circle markers]
    N --> O[Show disruption banner]
    
    style A fill:#7f1d1d,stroke:#ef4444,color:#fff
    style L fill:#991b1b,stroke:#ef4444,color:#fff
    style N fill:#991b1b,stroke:#ef4444,color:#fff
```

---

## 4. Agent Decision Flow

```mermaid
flowchart LR
    subgraph Input["Observation"]
        Inv[Inventory]
        Cost[Cost]
        CO2[CO2 Emissions]
        Step[Current Step]
        Supply[Supply Status]
        Traffic[Traffic Pressure]
    end
    
    subgraph Model["GNNActor Policy"]
        MLP[MLP Encoder]
        Head[Policy Head]
        Sample[Sample Action]
    end
    
    subgraph Shield["Safety Shield"]
        CheckCO2[Check CO2 Cap]
        CheckBudget[Check Budget]
        CheckHazmat[Check Hazmat]
        Filter[Modify Action]
    end
    
    subgraph Output["Environment"]
        StepEnv[Environment Step]
        Reward[Calculate Reward]
        NewState[New State]
    end
    
    Input --> MLP
    MLP --> Head
    Head --> Sample
    Sample --> CheckCO2
    CheckCO2 --> CheckBudget
    CheckBudget --> CheckHazmat
    CheckHazmat --> Filter
    Filter --> StepEnv
    StepEnv --> Reward
    Reward --> NewState
    
    style Input fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style Model fill:#1a2f1e,stroke:#10b981,color:#fff
    style Shield fill:#3f1f1f,stroke:#f59e0b,color:#fff
    style Output fill:#2d1f1f,stroke:#ef4444,color:#fff
```

---

## 5. Complete System Architecture

```mermaid
flowchart TB
    subgraph External["External Services"]
        GMaps[Google Maps API]
        Gemini[Gemini API]
        Neo4j[Neo4j Knowledge Graph]
        Vertex[Vertex AI]
    end
    
    subgraph Frontend["Dashboard (React)"]
        Header[Header Metrics]
        Controls[Control Buttons]
        MapPanel[Map Panel]
        Sidebar[Sidebar]
    end
    
    subgraph Flask["Flask Backend"]
        Routes[API Routes]
        StateMan[State Management]
    end
    
    subgraph AgentsML["ML Agents"]
        Actor[GNNActor]
        Critic[GNNCritic]
        Adversary[Adversary]
        Conformal[Conformal Predictor]
    end
    
    subgraph Safety["Safety Layer"]
        Shield[Safety Shield]
        Monitor[Constraint Monitor]
    end
    
    subgraph EnvSim["Environment"]
        SCEnv[SupplyChainEnv]
        Graph[Graph Builder]
        Topo[Topology YAML]
    end
    
    %% Connections
    Header -.->|Metrics| Routes
    Controls -.->|Actions| Routes
    MapPanel --> GMaps
    GMaps -->|Traffic Data| Routes
    Routes --> Frontend
    
    Routes --> StateMan
    StateMan --> Actor
    StateMan --> Critic
    StateMan --> Shield
    StateMan --> Conformal
    
    Actor -->|/api/step| Routes
    Shield -->|/api/step| Routes
    
    Routes --> SCEnv
    SCEnv --> Graph
    Graph --> Topo
    
    Shield -.->|interventions| Routes
    
    Routes --> Gemini
    Routes --> Neo4j
    
    %% Styling
    style Frontend fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style Flask fill:#1a2f1e,stroke:#10b981,color:#fff
    style AgentsML fill:#2d1f3f,stroke:#a855f7,color:#fff
    style Safety fill:#3f1f1f,stroke:#f59e0b,color:#fff
    style EnvSim fill:#1f2d1f,stroke:#22c55e,color:#fff
    style External fill:#2f1f2f,stroke:#6b7280,color:#fff
```

---

## 6. Training Pipeline

```mermaid
flowchart TB
    subgraph Init["Initialization"]
        Topo[Load Topology]
        InitEnv[Create Environment]
        InitActor[Init Actor/Critic]
    end
    
    subgraph Rollout["Rollout Collection"]
        Collect[Collect Episodes]
        ComputeGAE[Compute GAE]
        Store[Store in Buffer]
    end
    
    subgraph Update["Policy Update"]
        Sample[Sample Mini-batches]
        PPO[PPO Update]
        Clip[Clip Policy Loss]
        Value[Value Loss]
        Entropy[Entropy Bonus]
    end
    
    subgraph Output["Checkpoints"]
        SaveActor[Save Actor]
        SaveCritic[Save Critic]
        TensorBoard[TensorBoard Logs]
    end
    
    Init --> InitEnv
    Init --> InitActor
    InitEnv --> Collect
    Collect --> ComputeGAE
    ComputeGAE --> Store
    Store --> Sample
    Sample --> PPO
    PPO --> Clip
    PPO --> Value
    PPO --> Entropy
    Clip --> SaveActor
    Value --> SaveCritic
    Entropy --> TensorBoard
    
    style Init fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style Rollout fill:#1a2f1e,stroke:#10b981,color:#fff
    style Update fill:#3f1f1f,stroke:#f59e0b,color:#fff
    style Output fill:#2d1f1f,stroke:#ef4444,color:#fff
```

---

## 7. Signal Resolution Flow (KG + Gemini)

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Gemini
    participant Neo4j
    participant Resolver
    
    User->>API: POST /api/resolve<br/>text: "Flood at Mumbai port"
    
    API->>Gemini: Extract entities<br/>prompt with text
    
    Gemini-->>API: JSON entities<br/>{location: "Mumbai port", event: "flood"}
    
    API->>Resolver: Resolve entities
    
    loop For each entity
        Resolver->>Neo4j: Query location
        Neo4j-->>Resolver: Protocol: flood_response
        Resolver->>Neo4j: Query affected routes
        Neo4j-->>Resolver: [P01->W01, P01->W02]
    end
    
    Resolver-->>API: Combined result<br/>protocols, locations, confidence
    
    API-->>User: Response with<br/>grounded actions
```

---

## 8. Infrastructure Diagram

```mermaid
flowchart TB
    subgraph Client["Browser"]
        Browser[Chrome/Firefox<br/>React Dashboard]
    end
    
    subgraph Server["Python Server (localhost:5000)"]
        Flask[Flask App]
        PythonEnv[Python 3.10+]
    end
    
    subgraph DataFiles["Files"]
        Yaml[Topology YAML]
        Models[PyTorch Models]
        Dist[Production Build]
    end
    
    subgraph Optional["Optional Services"]
        Neo4jDB[Neo4j Database]
        GCP[Google Cloud<br/>Gemini/Vertex]
    end
    
    Browser -->|HTTP GET/POST| Flask
    Flask -->|Serve| Dist
    Flask -->|Load| Yaml
    Flask -->|Load| Models
    Flask -->|Query| Neo4jDB
    Flask -->|API Calls| GCP
    
    style Client fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style Server fill:#1a2f1e,stroke:#10b981,color:#fff
    style DataFiles fill:#2d1f1f,stroke:#ef4444,color:#fff
    style Optional fill:#3f1f1f,stroke:#6b7280,color:#fff
```

---

## 9. Data Contracts

```mermaid
classDiagram
    class Disruption {
        +list~list~ disabled_edges
        +dict capacity_multipliers
        +dict demand_multipliers
        +dict lead_time_multipliers
        +dict traffic_multipliers
    }
    
    class State {
        +int step
        +int max_steps
        +dict inventory
        +dict revenue
        +dict co2
        +dict cost
        +Disruption disruptions
        +dict traffic
    }
    
    class Topology {
        +list~Node~ nodes
        +list~Edge~ edges
        +Metadata metadata
    }
    
    class Action {
        +list~float~ inbound_orders
        +list~float~ outbound_allocations
    }
    
    class ShieldEvent {
        +str agent_id
        +str constraint_type
        +str message
        +list~float~ original_action
        +list~float~ modified_action
    }
    
    State --> Disruption: contains
    State --> Topology: uses
    Action --> ShieldEvent: filtered by
    
    style Disruption fill:#7f1d1d,stroke:#ef4444,color:#fff
    style State fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style Topology fill:#1a2f1e,stroke:#10b981,color:#fff
    style Action fill:#3f1f1f,stroke:#f59e0b,color:#fff
    style ShieldEvent fill:#7f1d1d,stroke:#f59e0b,color:#fff
```

---

## 10. Use Case Diagram

```mermaid
flowchart TB
    subgraph Actors["Actors"]
        Operator[Supply Chain Operator]
        System[ChainGuard AI System]
        Adversary[Adversary Agent]
        KG[Knowledge Graph]
    end
    
    subgraph UseCases["Use Cases"]
        Step[Execute Step]
        Disrupt[Inject Disruption]
        Reset[Reset Environment]
        Traffic[Update Traffic]
        Resolve[Resolve Signal]
        Train[Train Policy]
        Shield[Safety Check]
    end
    
    Operator --> Step
    Operator --> Disrupt
    Operator --> Reset
    Operator --> Traffic
    Operator --> Resolve
    
    System --> Train
    System --> Shield
    
    Adversary --> Disrupt
    
    KG --> Resolve
    
    Step --> System
    Disrupt --> System
    Reset --> System
    Traffic --> System
    
    style Actors fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style UseCases fill:#1a2f1e,stroke:#10b981,color:#fff
```

---

## Rendering Notes

1. **VS Code**: Install "Mermaid Preview" extension, create `.mmd` file, preview with `Ctrl+Shift+V`

2. **GitHub**: Native support - include in markdown file, view directly in repo

3. **Mermaid Live**: Copy-paste each diagram at https://mermaid.live/

4. **Export**: Can export as PNG/SVG from Mermaid Live Editor

---

*Generated for ChainGuard AI - Google Solution Challenge 2026*