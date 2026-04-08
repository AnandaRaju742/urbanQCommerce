import os
import sys
import json
import asyncio
from openai import AsyncOpenAI

# Ensure local imports work
sys.path.append(os.getcwd())

from client import UrbanQCommerceEnv
from models import UrbanQCommerceAction

# ✅ REQUIRED ENV VARIABLES (WITH DEFAULTS)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize client
client = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# Your deployed space
SPACE_URL = "https://anandarajug-urbanQCommerce.hf.space"


def safe_parse_action(content: str):
    """Robust parsing for LLM output"""
    try:
        decision = json.loads(content)
    except Exception:
        return {"action_type": "REFILL", "target_node_id": 1}

    # Validate action_type
    if decision.get("action_type") not in ["REFILL", "DISPATCH"]:
        decision["action_type"] = "REFILL"

    # Validate target_node_id
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

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        print(f"[START] task={task_id} env=urban_q_commerce model={MODEL_NAME}", flush=True)

        while True:
            step_count += 1

            prompt = (
                "You are a logistics AI.\n"
                "Return ONLY valid JSON.\n"
                "Format strictly:\n"
                "{\"action_type\": \"DISPATCH\", \"target_node_id\": 1}\n"
                "Rules:\n"
                "- action_type must be REFILL or DISPATCH\n"
                "- target_node_id must be 1, 2, or 3\n"
                "No extra text.\n"
            )

            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                decision = safe_parse_action(content)

                try:
                    action = UrbanQCommerceAction(**decision)
                except Exception:
                    action = UrbanQCommerceAction(
                        action_type="REFILL",
                        target_node_id=1
                    )

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
                # ✅ NEVER CRASH — continue safely
                success = False
                reward_str = "0.00"
                rewards.append(reward_str)

                print(
                    f"[STEP] step={step_count} action=REFILL(1) reward={reward_str} done=false error={str(e)}",
                    flush=True
                )

                continue

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
    # ✅ MUST EXACTLY MATCH openenv.yaml
    tasks = [
        "easy-static-routing",
        "medium-dynamic-attrition",
        "hard-surge-mitigation"
    ]

    for task in tasks:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())