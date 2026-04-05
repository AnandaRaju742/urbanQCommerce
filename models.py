from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any

class UrbanQCommerceAction(BaseModel):
    action_type: str = Field(..., description="REFILL or DISPATCH")
    target_node_id: int = Field(default=0)

class DemandNodeStatus(BaseModel):
    node_id: int
    position: Tuple[int, int]
    stock_remaining: int
    consumption_rate: int
    is_sla_at_risk: bool

class UrbanQCommerceObservation(BaseModel):
    done: bool
    reward: float
    agent_node_id: int
    fleet_cargo: int
    active_nodes: List[DemandNodeStatus]
    steps_remaining: int
    message: str

class UrbanQCommerceState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    total_sla_breaches: int = 0
    nodes: Dict[int, Any] = {}