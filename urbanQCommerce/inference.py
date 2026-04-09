import os
import json
import asyncio
from openai import OpenAI
try:
    from client import UrbanQCommerceEnv
    from models import UrbanQCommerceAction
except ImportError:
    from urban_q_commerce.client import UrbanQCommerceEnv
    from urban_q_commerce.models import UrbanQCommerceAction

# 1. OpenEnv Mandatory Configuration (Injected by Judges)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize the mandatory OpenAI client
llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

async def run_task(task_id: str, server_url: str = "http://localhost:8000"):
    env = UrbanQCommerceEnv(server_url)
    
    try:
        # [START] tag is mandatory
        print(f"[START] task={task_id} env=urban_q_commerce model={MODEL_NAME}", flush=True)
        
        result = await env.reset()
        obs = result.observation
        done = False
        step_count = 0
        total_reward = 0.0

        while not done and step_count < 50:
            step_count += 1
            
            prompt = f"Target Node ID 0 is Dark Store. Nodes: {json.dumps([n.model_dump() for n in obs.active_nodes])}. Cargo: {obs.fleet_cargo}. Choose DISPATCH or REFILL."
            
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            action_data = json.loads(response.choices[0].message.content)
            action = UrbanQCommerceAction(**action_data)

            step_result = await env.step(action)
            obs = step_result.observation
            done = step_result.done
            reward = step_result.reward
            total_reward = reward # Grader gives final score on last step

            # [STEP] tag is mandatory
            print(f"[STEP] step={step_count} action={action.action_type}({action.target_node_id}) reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

        # [END] tag is mandatory
        print(f"[END] task={task_id} score={total_reward:.2f} steps={step_count} success=true", flush=True)
        return total_reward

    except Exception as e:
        print(f"[END] task={task_id} score=0.0 steps=0 success=false error={str(e)}", flush=True)
        return 0.0
    finally:
        await env.close()

async def main():
    # The validator usually looks for these specific task IDs from your openenv.yaml
    tasks = ["easy-static-routing", "medium-dynamic-attrition", "hard-surge-mitigation"]
    for task in tasks:
        await run_task(task, "http://127.0.0.1:8000")

if __name__ == "__main__":
    asyncio.run(main())