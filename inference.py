import os
import json
import asyncio
from openai import AsyncOpenAI

# THE FIX: Match your EXACT folder name and models
from client import UrbanQCommerceEnv
from models import UrbanQCommerceAction

# Hackathon Mandatory Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Environment Variables
TASK_NAME = "easy-static-routing"
BENCHMARK = "urban_q_commerce"
# THE FIX: Use the direct "Live" URL of your space
LIVE_SPACE_URL = "https://anandarajug-urban-q-commerce.hf.space" 

async def run_baseline():
    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN environment variable is not set!")
        return

    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    # Connect directly to the running server
    print(f"🔄 Connecting to {LIVE_SPACE_URL}...")
    env = UrbanQCommerceEnv(base_url=LIVE_SPACE_URL)
    
    try:
        # Reset the environment
        result = await env.reset()
        obs = result.observation
        print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
        
        step_count = 0
        rewards = []
        done = False
        
        while not done and step_count < 10: # Keep it short for testing
            step_count += 1
            
            # Logic for the Q-Commerce Agent
            prompt = (
                f"Context: Logistics AI. Nodes: {obs.active_nodes}. Cargo: {obs.fleet_cargo}. "
                "Action must be REFILL or DISPATCH. "
                "Output ONLY JSON: {\"action_type\": \"DISPATCH\", \"target_node_id\": 1}"
            )
            
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            decision = json.loads(response.choices[0].message.content)
            action = UrbanQCommerceAction(**decision)
            
            # Step the environment
            result = await env.step(action)
            obs = result.observation
            done = result.done
            reward = result.reward
            rewards.append(reward)
            
            print(f"[STEP] step={step_count} action={action.action_type}({action.target_node_id}) "
                  f"reward={reward:.2f} done={str(done).lower()} error=null")

        final_score = rewards[-1] if rewards else 0.0
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(final_score > 0.5).lower()} steps={step_count} "
              f"score={final_score:.2f} rewards={rewards_str}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(run_baseline())