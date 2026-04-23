# ChainGuard AI — Resilient Supply Chain Intelligence

**Google Solution Challenge 2026** | **SDGs 9, 12, 13**
*Built with: Vertex AI, Gemini, Firebase, Google Maps, and PyTorch Geometric*

## The Problem
Global supply chains are increasingly vulnerable to black-swan events and climate shocks. Existing AI solutions often fail when the physical network topology changes, oscillate under safety constraints, or lack the ability to explain "why" a disruption is occurring.

## The Solution: ChainGuard AI
ChainGuard AI is a multi-agent reinforcement learning (MARL) system designed for maximum resilience and safety.

### Core Pillars
1.  **GNN-MAPPO**: A Graph Neural Network-based RL agent that adapts to dynamic network shapes without retraining.
2.  **Safety Shield**: Mathematical action masking that guarantees 0 violations of sustainability (CO2) and budget caps.
3.  **Adversarial Robustness**: A minimax adversary that proactively "attacks" the system to find vulnerabilities.
4.  **Gemini & KG Grounding**: Deterministic grounding of unstructured alerts (e.g., "Flood at Mumbai Port") into operational protocols.
5.  **Conformal Prediction**: Single-pass uncertainty quantification for intelligent human-in-the-loop escalation.

## Getting Started

### Prerequisites
- Python 3.10+
- Neo4j Database
- Google Cloud Project with Gemini API & Vertex AI enabled
- Firebase Project for the dashboard

### Installation
```bash
git clone https://github.com/your-repo/chainguard-ai.git
cd chainguard-ai
pip install -r requirements.txt
```

### Environment Setup
1.  Copy `.env.example` to `.env` and fill in your API keys.
2.  Import the Neo4j schema:
    ```bash
    cat kg/schema.cypher | cypher-shell -u neo4j -p your-password
    ```

### Run the Demo
1.  Start the local inference server:
    ```bash
    python serving/local_server.py
    ```
2.  Open `dashboard/index.html` in your browser (use a local web server like `npx serve dashboard`).

## Repository Structure
- `env/`: Supply chain simulation and topology configurations.
- `agents/`: GNN-MAPPO, Safety Shield, Adversary, and Conformal Prediction.
- `kg/`: Gemini entity extraction and Knowledge Graph client.
- `dashboard/`: Frontend UI with Google Maps and Firebase integration.
- `training/`: Vertex AI job configurations and training scripts.

## Demo Video
[Link to YouTube Video]

## License
This project is licensed under the MIT License.
