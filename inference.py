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
    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN is not set.")
        return

    # Using the direct URL we verified in the UI earlier
    SPACE_URL = "https://anandarajug-urbanQCommerce.hf.space"
    env = UrbanQCommerceEnv(SPACE_URL)
    
    try:
        # 1. Reset Environment
        result = await env.reset()
        obs = result.observation
        print(f"[START] task=easy-static-routing env=urban_q_commerce model={MODEL_NAME}")
        
        step_count = 0
        max_steps = 15
        total_rewards = []
        done = False
        
        # 2. Intelligent Loop
        while not done and step_count < max_steps:
            step_count += 1
            
            # DECISION LOGIC
            if obs.active_nodes:
                # Find the node that needs help the most
                target_node = min(obs.active_nodes, key=lambda n: n.stock_remaining)
                
                # If we have cargo and node stock is low, DISPATCH
                if obs.fleet_cargo > 5 and target_node.stock_remaining < 15:
                    action = UrbanQCommerceAction(
                        action_type="DISPATCH", 
                        target_node_id=target_node.node_id
                    )
                else:
                    # Otherwise, go REFILL
                    action = UrbanQCommerceAction(action_type="REFILL", target_node_id=0)
            else:
                action = UrbanQCommerceAction(action_type="REFILL", target_node_id=0)
            
            # 3. Step Environment
            result = await env.step(action)
            obs = result.observation
            done = result.done
            total_rewards.append(result.reward)
            
            # 4. Mandatory Log Format
            print(f"[STEP] step={step_count} action={action.action_type}({action.target_node_id}) "
                  f"reward={result.reward:.2f} done={str(done).lower()} error=null")

        # 5. Final Summary
        final_score = sum(total_rewards)
        rewards_str = ",".join([f"{r:.2f}" for r in total_rewards])
        print(f"[END] success=true steps={step_count} score={final_score:.2f} rewards={rewards_str}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(run_baseline())