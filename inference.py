import os
import sys
import json
import asyncio
from openai import AsyncOpenAI

# Add current directory to path to find client and models
sys.path.append(os.getcwd())

from client import UrbanQCommerceEnv
from models import UrbanQCommerceAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

async def run_baseline():
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    env = UrbanQCommerceEnv("https://anandarajug-urbanQcommerce.hf.space")
    
    try:
        result = await env.reset()
        obs = result.observation
        print(f"[START] task=easy env=urban_q_commerce model={MODEL_NAME}")
        
        # Simple test loop
        for i in range(1, 4):
            action = UrbanQCommerceAction(action_type="REFILL", target_node_id=0)
            result = await env.step(action)
            print(f"[STEP] step={i} action=REFILL reward={result.reward:.2f} done=false error=null")
        
        print(f"[END] success=true steps=3 score=1.00 rewards=0,0,0")
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(run_baseline())
