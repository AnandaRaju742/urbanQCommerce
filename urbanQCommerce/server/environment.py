import random
import uuid
from openenv.core.env_server import Environment
from ..models import (
    UrbanQCommerceAction, UrbanQCommerceObservation, 
    UrbanQCommerceState, DemandNodeStatus
)

class UrbanQCommerceEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_level: str = "medium"):
        self._state = UrbanQCommerceState()
        self._cargo, self._pos_id, self._max_steps = 0, 0, 50
        self.task_level = task_level

    def reset(self, seed=None, episode_id=None, **kwargs) -> UrbanQCommerceObservation:
        self._state = UrbanQCommerceState(
            episode_id=episode_id or str(uuid.uuid4()), step_count=0,
            nodes={i: {"id": i, "pos": (random.randint(0, 10), random.randint(0, 10)), 
                       "stock": 20, "max": 30, "rate": 2} for i in range(1, 4)}
        )
        self._cargo = 50
        return self._obs("System reset.")

    def step(self, action: UrbanQCommerceAction, **kwargs) -> UrbanQCommerceObservation:
        self._state.step_count += 1
        if action.action_type == "REFILL": 
            self._cargo, self._pos_id = 50, 0
        elif action.action_type == "DISPATCH":
            node = self._state.nodes.get(action.target_node_id)
            if node:
                self._pos_id = action.target_node_id
                transfer = min(self._cargo, node["max"] - node["stock"])
                node["stock"] += transfer
                self._cargo -= transfer
        
        for n in self._state.nodes.values():
            n["stock"] -= n["rate"]
            if n["stock"] < 0: 
                n["stock"] = 0
                self._state.total_sla_breaches += 1

        done = self._state.step_count >= self._max_steps
        obs = self._obs("Step complete")
        obs.done = done
        obs.reward = max(0.0, 1.0 - (self._state.total_sla_breaches * 0.1)) if done else 0.0
        return obs

    def _obs(self, message: str) -> UrbanQCommerceObservation:
        nodes = [DemandNodeStatus(**n, position=n["pos"], stock_remaining=n["stock"],
                                 consumption_rate=n["rate"], is_sla_at_risk=n["stock"] <= 5) 
                for n in self._state.nodes.values()]
        return UrbanQCommerceObservation(
            done=False, reward=0.0, agent_node_id=self._pos_id, fleet_cargo=self._cargo,
            active_nodes=nodes, steps_remaining=50 - self._state.step_count, message=message
        )

    @property
    def state(self) -> UrbanQCommerceState: return self._state