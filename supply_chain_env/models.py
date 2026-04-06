"""
Typed models for the Supply Chain Disruption Navigator OpenEnv.
All step() / reset() / state() inputs and outputs are validated here.
"""
from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SupplierTier(int, Enum):
    RAW_MATERIAL = 4
    PROCESSOR    = 3
    COMPONENT    = 2
    ASSEMBLER    = 1
    OEM          = 0


class DisruptionType(str, Enum):
    PORT_STRIKE       = "port_strike"
    WEATHER_EVENT     = "weather_event"
    GEOPOLITICAL_BLOCK= "geopolitical_block"
    FACTORY_FIRE      = "factory_fire"
    QUALITY_FAILURE   = "quality_failure"
    LOGISTICS_DELAY   = "logistics_delay"
    PANDEMIC_CLOSURE  = "pandemic_closure"


class DisruptionSeverity(str, Enum):
    MINOR    = "minor"    # capacity -20%
    MODERATE = "moderate" # capacity -50%
    SEVERE   = "severe"   # capacity -80%
    CRITICAL = "critical" # capacity -100% (total shutdown)


class ActionType(str, Enum):
    REROUTE_ORDER      = "reroute_order"
    ADJUST_SAFETY_STOCK= "adjust_safety_stock"
    EXPEDITE_SHIPMENT  = "expedite_shipment"
    ACCEPT_SUBSTITUTE  = "accept_substitute"
    NEGOTIATE_LEAD_TIME= "negotiate_lead_time"
    DO_NOTHING         = "do_nothing"


# ---------------------------------------------------------------------------
# Supplier models
# ---------------------------------------------------------------------------

class SupplierState(BaseModel):
    supplier_id: str
    name: str
    tier: SupplierTier
    country: str
    capacity_pct: float = Field(ge=0.0, le=1.0, description="Current capacity 0–1")
    lead_time_days: int = Field(ge=1, description="Current lead time in days")
    base_lead_time_days: int = Field(ge=1)
    unit_cost: float = Field(ge=0)
    reliability_score: float = Field(ge=0.0, le=1.0)
    esg_score: float = Field(ge=0.0, le=1.0, description="Environmental/social score")
    inventory_held: float = Field(ge=0.0, description="Units in inventory at this node")
    active_disruptions: List[str] = Field(default_factory=list, description="IDs of active disruptions")
    is_available: bool = True


class ActiveDisruption(BaseModel):
    disruption_id: str
    disruption_type: DisruptionType
    severity: DisruptionSeverity
    affected_supplier_id: str
    duration_remaining_days: int
    capacity_impact: float = Field(ge=0.0, le=1.0, description="Fraction of capacity lost")
    cost_multiplier: float = Field(ge=1.0, description="Cost increase factor")
    is_cascadable: bool = False
    cascade_probability: float = Field(ge=0.0, le=1.0, default=0.0)
    description: str = ""


# ---------------------------------------------------------------------------
# Order & flow models
# ---------------------------------------------------------------------------

class OrderRoute(BaseModel):
    from_supplier_id: str
    to_supplier_id: str
    material_type: str
    quantity: float
    lead_time_days: int
    unit_cost: float
    is_expedited: bool = False
    is_substitute: bool = False


class ProductionLine(BaseModel):
    line_id: str
    required_materials: Dict[str, float]  # material_type -> units_per_day
    current_throughput_pct: float = Field(ge=0.0, le=1.0)
    daily_demand: float
    units_produced_today: float = 0.0


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    A single agent action at one timestep.
    Multiple sub-actions can be bundled in one step.
    """
    action_type: ActionType
    # For REROUTE_ORDER
    from_supplier_id: Optional[str] = None
    to_supplier_id: Optional[str] = None
    material_type: Optional[str] = None
    quantity: Optional[float] = None
    # For ADJUST_SAFETY_STOCK
    supplier_id: Optional[str] = None
    target_stock_days: Optional[float] = None   # days of demand coverage
    # For EXPEDITE_SHIPMENT
    order_id: Optional[str] = None
    expedite_factor: Optional[float] = None      # speed multiplier, cost multiplier
    # For ACCEPT_SUBSTITUTE
    substitute_supplier_id: Optional[str] = None
    quality_penalty_pct: Optional[float] = None
    # For NEGOTIATE_LEAD_TIME
    target_lead_time_days: Optional[int] = None
    cost_premium_pct: Optional[float] = None     # % extra cost accepted


class MultiAction(BaseModel):
    """Bundle of actions the agent takes in one step."""
    actions: List[Action] = Field(default_factory=list)
    timestep: int = 0


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Full observable state returned after each step."""
    timestep: int
    day: int
    episode_length: int

    # Network state
    suppliers: Dict[str, SupplierState]
    active_disruptions: Dict[str, ActiveDisruption]
    current_routes: List[OrderRoute]
    production_lines: List[ProductionLine]

    # Aggregate KPIs (partial signals for reward shaping)
    fill_rate: float = Field(ge=0.0, le=1.0, description="% demand satisfied this step")
    inventory_coverage_days: float = Field(ge=0.0)
    total_cost_today: float
    cost_vs_baseline: float = Field(description="Ratio: actual/baseline cost")
    on_time_delivery_rate: float = Field(ge=0.0, le=1.0)
    esg_score_weighted: float = Field(ge=0.0, le=1.0)

    # Disruption intelligence (partial info in harder tasks)
    disruption_forecast: List[ActiveDisruption] = Field(
        default_factory=list,
        description="Visible upcoming disruptions (may be empty in hard mode)"
    )

    # Reward decomposition hint
    reward_breakdown: Optional["RewardBreakdown"] = None

    # Terminal flags
    is_done: bool = False
    is_truncated: bool = False
    info: Dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Decomposed reward — partial signals so agents can learn faster."""
    total: float = Field(ge=0.0, le=1.2)
    production_continuity: float = Field(ge=0.0, le=1.0,
        description="Fraction of demand met without line stoppage")
    cost_efficiency: float = Field(ge=0.0, le=1.0,
        description="1 - cost_overrun, penalises overspend vs baseline")
    esg_compliance: float = Field(ge=0.0, le=1.0,
        description="Weighted avg ESG score of active suppliers")
    lead_time_adherence: float = Field(ge=0.0, le=1.0,
        description="On-time delivery fraction")
    resilience_bonus: float = Field(ge=0.0, le=0.2,
        description="Bonus for proactive stock-building before disruptions")


# Needed for forward reference in Observation
Observation.model_rebuild()


# ---------------------------------------------------------------------------
# Reset / State API models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(description="'easy' | 'medium' | 'hard'")
    seed: Optional[int] = None
    config_overrides: Dict = Field(default_factory=dict)


class ResetResponse(BaseModel):
    observation: Observation
    task_description: str
    action_space_description: str
    episode_length: int
    seed: int


class StateResponse(BaseModel):
    """Snapshot of full internal state (for debugging / graders)."""
    observation: Observation
    internal_config: Dict
    cumulative_reward: float
    step_rewards: List[float]
