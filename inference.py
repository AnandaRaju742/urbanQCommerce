import os
import sys
import json
import asyncio
from openai import AsyncOpenAI

sys.path.append(os.getcwd())

from client import UrbanQCommerceEnv
from models import UrbanQCommerceAction

# ✅ REQUIRED DEFAULTS
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

SPACE_URL = "https://anandarajug-urbanQCommerce.hf.space"


async def run_task(task_id):
    env = UrbanQCommerceEnv(SPACE_URL)

    rewards = []
    step_count = 0
    success = True

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        print(f"[START] task={task_id} env=urban_q_commerce model={MODEL_NAME}", flush=True)

        while True:
            step_count += 1

            prompt = (
                f"You are a logistics agent.\n"
                f"Cargo: {obs.fleet_cargo}\n"
                f"Nodes: {[{'id': n.node_id, 'stock': n.stock_remaining} for n in obs.active_nodes]}\n"
                f"Choose action: REFILL or DISPATCH.\n"
                f"Return JSON: {{\"action_type\":\"DISPATCH\",\"target_node_id\":1}}"
            )

            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )

                decision = json.loads(response.choices[0].message.content)
                action = UrbanQCommerceAction(**decision)

                result = await env.step(action)
                obs = result.observation

                reward_str = f"{result.reward:.2f}"
                rewards.append(reward_str)

                print(
                    f"[STEP] step={step_count} action={action.action_type}({action.target_node_id}) "
                    f"reward={reward_str} done={str(result.done).lower()} error=null",
                    flush=True
                )

                if result.done:
                    break

            except Exception as e:
                success = False
                print(
                    f"[STEP] step={step_count} action=ERROR reward=0.00 done=true error={str(e)}",
                    flush=True
                )
                break

        print(
            f"[END] success={str(success).lower()} steps={step_count} rewards={','.join(rewards)}",
            flush=True
        )

    except Exception as e:
        print(
            f"[END] success=false steps=0 rewards= error={str(e)}",
            flush=True
        )

    finally:
        await env.close()


async def main():
    # ✅ MUST MATCH YAML EXACTLY
    tasks = [
        "easy-static-routing",
        "medium-dynamic-attrition",
        "hard-surge-mitigation"
    ]

    for task in tasks:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())