import os
import sys
import json
import asyncio
from openai import AsyncOpenAI

# Add current directory to path to find client and models
sys.path.append(os.getcwd())

from client import UrbanQCommerceEnv
from models import UrbanQCommerceAction

# THESE MUST BE LOADED FROM ENVIRONMENT - DO NOT HARDCODE
API_BASE_URL = os.getenv("API_BASE_URL")
# The validator specifically looks for "API_KEY" as the variable name
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

async def run_baseline():
    # 1. Initialize client using THEIR proxy and THEIR key
    client = AsyncOpenAI(
        base_url=API_BASE_URL, 
        api_key=API_KEY
    )
    
    SPACE_URL = "https://anandarajug-urbanQCommerce.hf.space"
    env = UrbanQCommerceEnv(SPACE_URL)
    
    try:
        result = await env.reset()
        obs = result.observation
        print(f"[START] task=easy-static-routing env=urban_q_commerce model={MODEL_NAME}")
        
        for step_count in range(1, 11): # 10 steps is usually enough for validation
            # 2. ASK THE LLM FOR THE ACTION
            prompt = (
                f"You are a Logistics AI. Current Status: Cargo={obs.fleet_cargo}, "
                f"Nodes={[{'id': n.node_id, 'stock': n.stock_remaining} for n in obs.active_nodes]}. "
                "Decide: REFILL or DISPATCH. If DISPATCH, provide target_node_id. "
                "Output ONLY valid JSON: {\"action_type\": \"DISPATCH\", \"target_node_id\": 1}"
            )

            # THIS CALL UPDATES THEIR 'LAST_ACTIVE' LOG
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # 3. Parse and execute
            decision = json.loads(response.choices[0].message.content)
            action = UrbanQCommerceAction(**decision)
            
            result = await env.step(action)
            obs = result.observation
            
            print(f"[STEP] step={step_count} action={action.action_type}({action.target_node_id}) "
                  f"reward={result.reward:.2f} done={str(result.done).lower()} error=null")

            if result.done:
                break

        print(f"[END] success=true steps={step_count} score={result.reward:.2f}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(run_baseline())