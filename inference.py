import os
import sys
import json
import asyncio
import importlib
from openai import AsyncOpenAI

# --- THE BULLETPROOF IMPORT FIX ---
# This forces Python to treat your folder as a proper package, 
# fixing the "relative import" error regardless of what the folder is named!
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
folder_name = os.path.basename(current_dir)

sys.path.insert(0, parent_dir)

client_mod = importlib.import_module(f"{folder_name}.client")
models_mod = importlib.import_module(f"{folder_name}.models")

UrbanQCommerceEnv = client_mod.UrbanQCommerceEnv
UrbanQCommerceAction = models_mod.UrbanQCommerceAction
# ----------------------------------

# Hackathon Mandatory Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Environment Variables
TASK_NAME = os.getenv("TASK_NAME", "easy-static-routing")
BENCHMARK = "urban_q_commerce"
LIVE_SPACE_URL = "https://anandarajug-urbanQCommerce.hf.space" # Your live Space!

async def run_baseline():
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    # Connect to your live environment
    env = UrbanQCommerceEnv(LIVE_SPACE_URL)
    
    try:
        result = await env.reset()
        obs = result.observation
        done = result.done
        step_count = 0
        rewards = []
        
        # [START] Mandatory Log
        print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
        
        while not done and step_count < 50:
            step_count += 1
            error_msg = "null"
            reward = 0.0
            
            try:
                # Baseline LLM Agent Logic
                prompt = f"You are a logistics AI. Fleet cargo: {obs.fleet_cargo}. Active Nodes: {obs.active_nodes}. Choose action: REFILL or DISPATCH (to node ID). Reply in pure JSON: {{\"action_type\": \"DISPATCH\", \"target_node_id\": 1}}"
                
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                # Parse LLM decision
                decision = json.loads(response.choices[0].message.content)
                action = UrbanQCommerceAction(**decision)
                
                # Take step in environment
                result = await env.step(action)
                obs = result.observation
                done = result.done
                reward = result.reward
                rewards.append(reward)
                
                action_str = f"{action.action_type}({action.target_node_id})"
                
            except Exception as e:
                error_msg = f"\"{str(e)}\""
                action_str = "ERROR"
                done = True
            
            # [STEP] Mandatory Log (Formatted to 2 decimal places)
            print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")

        # [END] Mandatory Log
        final_score = rewards[-1] if rewards else 0.0
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(final_score > 0.5).lower()} steps={step_count} score={final_score:.2f} rewards={rewards_str}")

    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(run_baseline())