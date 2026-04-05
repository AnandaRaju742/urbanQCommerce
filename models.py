from typing import List, Dict, Tuple
from openenv.core.env_server import Action, Observation, State
from pydantic import Field

class UrbanQCommerceAction(Action):
    action_type: str = Field(..., description="'DISPATCH' or 'REFILL'")
    target_node_id: int = Field(..., description="0 for Dark Store, >0 for Nodes")

class DemandNodeStatus(State): 
    node_id: int
    position: Tuple[int, int]
    stock_remaining: int
    consumption_rate: int
    is_sla_at_risk: bool

class UrbanQCommerceObservation(Observation):
    agent_node_id: int
    fleet_cargo: int
    active_nodes: List[DemandNodeStatus]
    steps_remaining: int
    message: str

class UrbanQCommerceState(State):
    grid_size: int = 10
    dark_store_pos: Tuple[int, int] = (5, 5)
    max_cargo: int = 50
    nodes: Dict[int, dict] = {}
    total_sla_breaches: int = 0