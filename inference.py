import os
import json
import asyncio
from openai import AsyncOpenAI

from client import UrbanQCommerceEnv
from models import UrbanQCommerceAction


# =========================
# ENV VARIABLES
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY is required")


client = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


# =========================
# SAFE SCORE (STRICT)
# =========================
def safe_score(x):
    try:
        x = float(x)
        # 🔥 HARD clamp BEFORE formatting
        if x <= 0.0:
            return 0.01
        if x >= 1.0:
            return 0.99
        return max(0.01, min(x, 0.99))
    except:
        return 0.01


# =========================
# SAFE FORMAT (CRITICAL FIX)
# =========================
def safe_format(x):
    x = safe_score(x)

    # 🔥 CRITICAL: prevent rounding to 0.00 or 1.00
    if x <= 0.01:
        return "0.01"
    if x >= 0.99:
        return "0.99"

    return f"{x:.2f}"


# =========================
# LLM ACTION
# =========================
async def get_action(obs):
    try:
        prompt = (f"""
You are controlling a delivery fleet.

Current state:
- Cargo: {obs.fleet_cargo}
- Nodes: {[{'id': n.node_id, 'stock': n.stock_remaining} for n in obs.active_nodes]}

Available actions:
1. DISPATCH(node_id) → send stock to node
2. REFILL() → refill cargo at warehouse

Rules:
- Choose node with LOW stock
- Avoid sending to FULL nodes
- Use REFILL if cargo is low

Return ONLY JSON:
{{"action_type": "DISPATCH", "target_node_id": <id>}}
OR
{{"action_type": "REFILL"}}
"""
        )

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)

        return UrbanQCommerceAction(
            action_type=data.get("action_type", "REFILL"),
            target_node_id=data.get("target_node_id", 1)
        ), None

    except Exception as e:
        return None, str(e)


# =========================
# RUN TASK
# =========================
async def run_task(task_id):
    SPACE_URL = os.getenv(
        "SPACE_URL",
        "https://anandarajug-urbanQCommerce.hf.space"
    )

    env = UrbanQCommerceEnv(SPACE_URL)

    rewards = []
    steps = 0
    success = True

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        print(
            f"[START] task={task_id} env=urban_q_commerce model={MODEL_NAME}",
            flush=True
        )

        for step in range(1, 51):
            steps = step

            action, error = await get_action(obs)

            if error:
                reward = 0.01
                done = True
                success = False
                error_msg = error
            else:
                try:
                    result = await env.step(action)
                    obs = result.observation
                    reward = safe_score(result.reward)
                    done = result.done
                    error_msg = "null"
                except Exception as e:
                    reward = 0.01
                    done = True
                    success = False
                    error_msg = str(e)

            rewards.append(safe_score(reward))

            action_str = (
                f"{action.action_type}({action.target_node_id})"
                if action else "ERROR"
            )

            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={safe_format(reward)} "
                f"done={str(done).lower()} error={error_msg}",
                flush=True
            )

            if done:
                break

    except Exception:
        success = False

    finally:
        try:
            await env.close()
        except:
            pass

        # =========================
        # FINAL SCORE (STRICT SAFE)
        # =========================
        score = sum(rewards) / len(rewards) if rewards else 0.01
        score = safe_score(score)

        SUCCESS_SCORE_THRESHOLD = 0.5
        success = success and (score >= SUCCESS_SCORE_THRESHOLD)

        rewards_str = ",".join(safe_format(r) for r in rewards)

        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps} rewards={rewards_str}",
            flush=True
        )


# =========================
# MAIN
# =========================
async def main():
    tasks = [
        "easy-static-routing",
        "medium-dynamic-attrition",
        "hard-surge-mitigation"
    ]

    for t in tasks:
        await run_task(t)


if __name__ == "__main__":
    asyncio.run(main())