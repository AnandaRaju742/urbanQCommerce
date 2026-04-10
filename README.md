---
title: Urban Q-Commerce Environment
emoji: 🚚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
base_path: /web
---

# 🏙️ Urban Q-Commerce: The AI Logistics Agent Environment 🚚

Welcome to **Urban Q-Commerce**, an innovative AI logistics environment designed to simulate and optimize rapid delivery operations within a dynamic urban landscape. This project offers a challenging reinforcement learning playground where an AI agent's mission is to efficiently manage a delivery fleet, preventing stockouts at various demand nodes, thereby minimizing SLA (Service Level Agreement) breaches and maximizing operational efficiency.

Built upon the robust **OpenEnv** framework, this environment exposes its core logic as a flexible HTTP API, enabling seamless interaction with diverse AI agents.

## ✨ Key Features

*   **Dynamic Demand Nodes:** Multiple delivery points across a grid, each with fluctuating stock levels and consumption rates.
*   **Fleet Management:** Control a single delivery vehicle with a finite cargo capacity.
*   **Strategic Actions:** Agents can `DISPATCH` cargo to critical demand nodes or `REFILL` their fleet at the central Dark Store.
*   **SLA-Driven Performance:** Stockouts at demand nodes result in SLA breaches, directly impacting the agent's overall score.
*   **Reward System:** Final scores are determined by the number of SLA breaches, incentivizing proactive stock management.
*   **Configurable Difficulty:** Three `task_level` settings (`easy`, `medium`, `hard`) dynamically adjust grid size, node count, and consumption rates.
*   **HTTP API Interface:** Effortlessly connect your AI agent via the provided `UrbanQCommerceEnv` client, which communicates over HTTP/WebSockets.

## 📦 Data Models

Interaction with the environment is facilitated by clearly defined Pydantic models:

### `UrbanQCommerceAction`
Represents an action the agent can take:
```python
class UrbanQCommerceAction(BaseModel):
    action_type: str = Field(..., description="'REFILL' or 'DISPATCH'")
    target_node_id: int = Field(default=0, description="0 for Dark Store, >0 for Demand Nodes")
```

### `DemandNodeStatus`
Provides real-time information about a specific demand node:
```python
class DemandNodeStatus(BaseModel):
    node_id: int
    position: Tuple[int, int]
    stock_remaining: int
    consumption_rate: int
    is_sla_at_risk: bool
```

### `UrbanQCommerceObservation`
The comprehensive observation an agent receives after each step:
```python
class UrbanQCommerceObservation(BaseModel):
    done: bool
    reward: float
    agent_node_id: int
    fleet_cargo: int
    active_nodes: List[DemandNodeStatus]
    steps_remaining: int
    message: str
```

### `UrbanQCommerceState`
The internal state of the environment.

## 🎯 Tasks and Difficulty Levels

The `openenv.yaml` configures three distinct tasks, each offering varying levels of complexity:

*   **`easy-static-routing`**
    *   `task_level: "easy"`
    *   `max_steps: 20`
    *   _Characterized by a smaller grid, fewer nodes, and slower consumption rates._

*   **`medium-dynamic-attrition`**
    *   `task_level: "medium"`
    *   `max_steps: 50`
    *   _Features a medium-sized grid, more nodes, and moderate consumption._

*   **`hard-surge-mitigation`**
    *   `task_level: "hard"`
    *   `max_steps: 100`
    *   _Presents a larger grid, increased nodes, and faster/variable consumption challenges._

## 🚀 Local Setup and Execution

To get the Urban Q-Commerce environment running on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://huggingface.co/spaces/AnandarajuG/urbanQCommerce
    cd urbanQCommerce
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start the Server:**
    ```bash
    python urban_q_commerce/server/app.py
    # Alternatively, use uvicorn directly:
    # uvicorn urban_q_commerce.server.app:app --host 0.0.0.0 --port 8000
    ```
    The server will be accessible at `http://0.0.0.0:8000`.

4.  **Verify Health:**
    ```bash
    curl http://localhost:8000/health
    ```
    A successful response will look like: `{"status":"healthy","message":"Urban Q-Commerce is live!"}`.

## 🤝 Client Interaction Example

Interact with your local environment using the `UrbanQCommerceEnv` client:

```python
import asyncio
from urban_q_commerce.client import UrbanQCommerceEnv
from urban_q_commerce.models import UrbanQCommerceAction

async def main():
    env = UrbanQCommerceEnv("http://localhost:8000") # Connect to your local server
    try:
        print("--- Resetting Environment ---")
        result = await env.reset(task_id="easy-static-routing")
        obs = result.observation
        print(f"Initial Observation: {obs.message} | Cargo: {obs.fleet_cargo} | Steps Left: {obs.steps_remaining}")

        print("\n--- Taking REFILL Action ---")
        action = UrbanQCommerceAction(action_type="REFILL", target_node_id=0)
        step_result = await env.step(action)
        obs = step_result.observation
        print(f"After REFILL: {obs.message} | Cargo: {obs.fleet_cargo} | Steps Left: {obs.steps_remaining}")

        print("\n--- Taking DISPATCH Action ---")
        if obs.active_nodes:
            target_node = obs.active_nodes[0].node_id
            action = UrbanQCommerceAction(action_type="DISPATCH", target_node_id=target_node)
            step_result = await env.step(action)
            obs = step_result.observation
            print(f"After DISPATCH to Node {target_node}: {obs.message} | Cargo: {obs.fleet_cargo} | Steps Left: {obs.steps_remaining}")
        else:
            print("No active nodes to dispatch to.")

    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🤖 AI Agent (inference.py)

The `inference.py` script showcases how an AI agent, leveraging an OpenAI-compatible LLM, can interact with the environment. It's crucial for evaluating your agent's performance in a structured manner.

To run the agent locally against your environment:

1.  **Set Environment Variables:**
    ```bash
    export HF_TOKEN="hf_YOUR_HUGGINGFACE_API_TOKEN" # Your Hugging Face API token with inference access
    export API_BASE_URL="https://router.huggingface.co/v1" # e.g., Hugging Face Inference API or other LLM provider
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" # Specify your chosen LLM model
    ```

2.  **Execute the Inference Script:**
    ```bash
    python inference.py
    ```
    The script will output structured logs (`[START]`, `[STEP]`, `[END]`) essential for validator compatibility.

## ☁️ Deployment to Hugging Face Spaces

This environment is optimized for seamless deployment on Hugging Face Spaces using the Docker SDK. The `openenv push` command handles the entire deployment process, including Docker image building and pushing.

1.  **Ensure Configuration is Synchronized:** Verify that `Dockerfile`, `requirements.txt`, and `openenv.yaml` are correctly configured. Note that `app_port` in `README.md` and `port` in `openenv.yaml` should align with the `CMD` port in `Dockerfile` (typically `8000` for local development and `7860` for Hugging Face Spaces by convention).

2.  **Login to Hugging Face CLI:**
    ```bash
    huggingface-cli login
    ```

3.  **Deploy with OpenEnv CLI:**
    ```bash
    openenv push --repo-id <your-username>/urban-q-commerce
    ```
    This command will build and deploy your Docker image to your specified Hugging Face Space.

## 🌐 API Endpoints

When deployed, the environment exposes a comprehensive set of API endpoints:

*   **`/`**: Root health check, returning `{"status": "ok", "message": "Hugging Face Health Check Passed!"}`.
*   **`/health`**: Standard OpenEnv health check, providing detailed server status.
*   **`/schema`**: Delivers the OpenAPI schema for the environment, detailing available actions, observations, and tasks.
*   **`/reset`**: Initializes a new episode. Accepts a `task_id` in the request body.
*   **`/step`**: Advances the environment by one step. Accepts an `UrbanQCommerceAction` in the request body.

## 📜 License

This project is released under the MIT License. For full details, please refer to the `LICENSE` file within the repository.