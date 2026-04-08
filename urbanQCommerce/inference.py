import os
import sys
import json
import asyncio
from openai import AsyncOpenAI

sys.path.append(os.getcwd())

from client import UrbanQCommerceEnv
from models import UrbanQCommerceAction

# ✅ SAFE ENV LOADING
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

# ✅ THIS IS THE IMPORTANT LINE
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

if API_KEY is None:
    raise ValueError("API_KEY or HF_TOKEN must be set")

client = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

SPACE_URL = "https://anandarajug-urbanQCommerce.hf.space"


def safe_parse_action(content: str):
    try:
        decision = json.loads(content)
    except Exception:
        return {"action_type": "REFILL", "target_node_id": 1}

    if decision.get("action_type") not in ["REFILL", "DISPATCH"]:
        decision["action_type"] = "REFILL"

    if (
        "target_node_id" not in decision
        or decision["target_node_id"] is None
        or not isinstance(decision["target_node_id"], int)
        or decision["target_node_id"] not in [1, 2, 3]
    ):
        decision["target_node_id"] = 1

    return decision


async def run_task(task_id):
    env = UrbanQCommerceEnv(SPACE_URL)

    rewards = []
    step_count = 0
    success = True

    print(f"[START] task={task_id} env=urban_q_commerce model={MODEL_NAME}", flush=True)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        while True:
            step_count += 1

            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": "Return JSON: {\"action_type\":\"DISPATCH\",\"target_node_id\":1}"
                    }],
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                decision = safe_parse_action(content)

                try:
                    action = UrbanQCommerceAction(**decision)
                except Exception:
                    action = UrbanQCommerceAction(action_type="REFILL", target_node_id=1)

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
                rewards.append("0.00")

                print(
                    f"[STEP] step={step_count} action=REFILL(1) reward=0.00 done=false error={str(e)}",
                    flush=True
                )

                if step_count > 5:  # safety exit
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
    tasks = [
        "easy-static-routing",
        "medium-dynamic-attrition",
        "hard-surge-mitigation"
    ]

    for task in tasks:
        await run_task(task)


# ✅ SAFE ENTRY POINT
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        # LAST RESORT: always print something
        print(f"[END] success=false steps=0 rewards= error={str(e)}", flush=True)