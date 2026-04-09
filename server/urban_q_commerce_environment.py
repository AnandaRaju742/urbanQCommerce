import random
import uuid
from openenv.core.env_server import Environment

try:
    from ..models import (
        UrbanQCommerceAction,
        UrbanQCommerceObservation,
        UrbanQCommerceState,
        DemandNodeStatus
    )
except ImportError:
    from models import (
        UrbanQCommerceAction,
        UrbanQCommerceObservation,
        UrbanQCommerceState,
        DemandNodeStatus
    )


# =========================
# NEW GRADERS
# =========================

def grade_balance_efficiency(state):
    try:
        stocks = [n["stock"] for n in state.nodes.values()]
        if not stocks:
            return 0.01

        avg = sum(stocks) / len(stocks)
        imbalance = sum(abs(s - avg) for s in stocks) / len(stocks)

        score = 1 / (1 + imbalance)

        return round(max(0.01, min(0.99, score)), 3)
    except Exception:
        return 0.01


def grade_resource_utilization(state, cargo):
    try:
        max_cargo = 50
        utilization = 1 - (cargo / max_cargo)

        score = 0.5 + 0.5 * utilization

        return round(max(0.01, min(0.99, score)), 3)
    except Exception:
        return 0.01


# =========================
# ENVIRONMENT
# =========================

class UrbanQCommerceEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_level: str = "medium"):
        self._state = UrbanQCommerceState()
        self._cargo = 50
        self._pos_id = 0
        self._max_steps = 50
        self.task_level = task_level

    def reset(self, seed=None, episode_id=None, **kwargs) -> UrbanQCommerceObservation:
        self._state = UrbanQCommerceState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            nodes={
                i: {
                    "id": i,
                    "pos": (random.randint(0, 10), random.randint(0, 10)),
                    "stock": 20,
                    "max": 30,
                    "rate": 2
                } for i in range(1, 4)
            },
            total_sla_breaches=0
        )
        self._cargo = 50
        return self._obs("System reset.")

    def step(self, action: UrbanQCommerceAction, **kwargs) -> UrbanQCommerceObservation:
        self._state.step_count += 1
        step_reward = 0.0

        multiplier = kwargs.get("rate_multiplier", 1)

        # =========================
        # ACTION LOGIC
        # =========================

        if action.action_type == "REFILL":
            self._cargo = 50
            self._pos_id = 0

        elif action.action_type == "DISPATCH":
            node = self._state.nodes.get(action.target_node_id)

            if node:
                self._pos_id = action.target_node_id

                transfer = min(self._cargo, node["max"] - node["stock"])

                if transfer > 0:
                    step_reward += 0.1

                node["stock"] += transfer
                self._cargo -= transfer

        # =========================
        # CONSUMPTION
        # =========================

        for n in self._state.nodes.values():
            n["stock"] -= (n["rate"] * multiplier)

            if n["stock"] <= 0:
                n["stock"] = 0
                self._state.total_sla_breaches += 1
                step_reward -= 0.05

        # =========================
        # DONE
        # =========================

        done = self._state.step_count >= self._max_steps

        # =========================
        # REWARD CALCULATION (FIXED)
        # =========================

        total_possible_stock = 3 * 30
        current_stock = sum(n["stock"] for n in self._state.nodes.values())
        stock_efficiency = current_stock / total_possible_stock

        balance_score = grade_balance_efficiency(self._state)
        resource_score = grade_resource_utilization(self._state, self._cargo)

        sla_penalty = 1 - (self._state.total_sla_breaches / 50)

        raw_score = (
            0.4 * stock_efficiency +
            0.2 * balance_score +
            0.2 * resource_score +
            0.2 * sla_penalty +
            step_reward
        )

        # 🔥 STRICT RANGE FIX (CRITICAL)
        reward = round(max(0.01, min(0.99, raw_score)), 2)

        obs = self._obs("Step complete")
        obs.reward = reward
        obs.done = done

        return obs

    # =========================
    # OBSERVATION
    # =========================

    def _obs(self, message: str) -> UrbanQCommerceObservation:
        nodes = [
            DemandNodeStatus(
                node_id=n["id"],
                position=n["pos"],
                stock_remaining=n["stock"],
                consumption_rate=n["rate"],
                is_sla_at_risk=n["stock"] <= 5
            )
            for n in self._state.nodes.values()
        ]

        return UrbanQCommerceObservation(
            done=False,
            reward=0.0,
            agent_node_id=self._pos_id,
            fleet_cargo=self._cargo,
            active_nodes=nodes,
            steps_remaining=50 - self._state.step_count,
            message=message
        )

    @property
    def state(self) -> UrbanQCommerceState:
        return self._state


# =========================
# FINAL TASK SCORE (USED BY YAML)
# =========================

def compute_score(trajectory, **kwargs):
    try:
        if not trajectory:
            return 0.01

        rewards = [step.reward for step in trajectory if step.reward is not None]

        if not rewards:
            return 0.01

        score = sum(rewards) / len(rewards)

        # 🔥 STRICT FIX (THIS WAS FAILING YOU)
        score = max(1e-6, min(score, 1 - 1e-6))

        return round(score, 3)

    except Exception:
        return 0.01