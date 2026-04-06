"""
SupplyChainEnv — OpenEnv-compliant environment.

Public API:
    env.reset(task_id, seed) -> ResetResponse
    env.step(multi_action)   -> Observation
    env.state()              -> StateResponse

Each task (easy/medium/hard) loads its own disruption schedule,
episode length, and reward weights.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional

from .models import (
    Action, ActionType, MultiAction, Observation, ResetRequest,
    ResetResponse, RewardBreakdown, StateResponse,
)
from .network import SupplyChainNetwork
from .disruptions import DisruptionEngine
from .reward import compute_reward, episode_score


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict] = {
    "easy": dict(
        description=(
            "Single-tier crisis: one T2 component supplier goes down. "
            "Agent must reroute orders and maintain production. "
            "Full disruption visibility. 30-day episode."
        ),
        episode_length=30,
        base_fire_probability=0.0,   # no random fires — only forced
        max_concurrent=1,
        cascade_enabled=False,
        partial_info=False,
        adversarial=False,
        forced_disruptions=[
            dict(
                supplier_id="S201",
                fire_on_day=5,
                disruption_type="factory_fire",
                severity="severe",
                duration=12,
                capacity_impact=0.80,
                cost_multiplier=1.5,
                description="Factory fire at Battery Mfg Eta — T2 supplier down",
            )
        ],
        reward_weights=dict(
            production_continuity=0.60,
            cost_efficiency=0.20,
            esg_compliance=0.10,
            lead_time_adherence=0.10,
        ),
        cost_tolerance=0.40,
    ),

    "medium": dict(
        description=(
            "Multi-tier crisis with cascading failures and partial information. "
            "Geopolitical block hits a T4 supplier, cascades to T3. "
            "Cost budget enforced. 60-day episode."
        ),
        episode_length=60,
        base_fire_probability=0.015,
        max_concurrent=3,
            cascade_enabled=True,
        partial_info=True,
        adversarial=False,
        forced_disruptions=[
            dict(
                supplier_id="S401",
                fire_on_day=7,
                disruption_type="geopolitical_block",
                severity="critical",
                duration=25,
                capacity_impact=1.0,
                cost_multiplier=2.0,
                description="Export ban on lithium ore from Chile",
            ),
            dict(
                supplier_id="S301",
                fire_on_day=12,          # cascade consequence
                disruption_type="logistics_delay",
                severity="moderate",
                duration=10,
                capacity_impact=0.50,
                cost_multiplier=1.25,
                description="Smelter input shortage from Chile export ban",
            ),
        ],
        reward_weights=dict(
            production_continuity=0.50,
            cost_efficiency=0.25,
            esg_compliance=0.15,
            lead_time_adherence=0.10,
        ),
        cost_tolerance=0.30,
    ),

    "hard": dict(
        description=(
            "Full adversarial mode: high-frequency stochastic disruptions, "
            "cascades enabled, no forecast, ESG scoring + cost optimisation, "
            "simultaneous multi-tier crises. 90-day episode."
        ),
        episode_length=90,
        base_fire_probability=0.04,
        max_concurrent=5,
        cascade_enabled=True,
        partial_info=True,
        adversarial=True,
        forced_disruptions=[
            dict(
                supplier_id="S301",
                fire_on_day=3,
                disruption_type="pandemic_closure",
                severity="severe",
                duration=20,
                capacity_impact=0.80,
                cost_multiplier=1.5,
                description="Government-mandated pandemic closure",
            ),
            dict(
                supplier_id="S202",
                fire_on_day=10,
                disruption_type="quality_failure",
                severity="moderate",
                duration=8,
                capacity_impact=0.50,
                cost_multiplier=1.25,
                description="PCB quality failure triggers recall",
            ),
        ],
        reward_weights=dict(
            production_continuity=0.40,
            cost_efficiency=0.25,
            esg_compliance=0.20,
            lead_time_adherence=0.15,
        ),
        cost_tolerance=0.20,   # tighter cost constraint
    ),
}


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class SupplyChainEnv:
    """
    OpenEnv-compliant Supply Chain Disruption Navigator.

    Usage:
        env = SupplyChainEnv()
        resp = env.reset(ResetRequest(task_id="easy", seed=42))
        obs = env.step(MultiAction(actions=[Action(action_type="do_nothing")]))
        state = env.state()
    """

    def __init__(self):
        self._network: Optional[SupplyChainNetwork] = None
        self._disruption_engine: Optional[DisruptionEngine] = None
        self._config: Dict = {}
        self._task_id: str = ""
        self._seed: int = 0
        self._timestep: int = 0
        self._episode_length: int = 0
        self._step_rewards: List[float] = []
        self._prev_inventory_days: float = 0.0
        self._is_done: bool = False
        self._start_time: float = 0.0

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(self, request: ResetRequest) -> ResetResponse:
        """
        Reset the environment to the start of an episode.
        Returns initial observation + task metadata.
        """
        task_id = request.task_id
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Choose from: {list(TASK_CONFIGS)}"
            )
        cfg = dict(TASK_CONFIGS[task_id])
        cfg.update(request.config_overrides)

        seed = request.seed if request.seed is not None else int(time.time()) % 100000
        self._task_id = task_id
        self._seed = seed
        self._config = cfg
        self._timestep = 0
        self._episode_length = cfg["episode_length"]
        self._step_rewards = []
        self._is_done = False
        self._start_time = time.time()

        # Build network and disruption engine
        self._network = SupplyChainNetwork(seed=seed)
        self._disruption_engine = DisruptionEngine(
            seed=seed,
            base_fire_probability=cfg["base_fire_probability"],
            max_concurrent=cfg["max_concurrent"],
            cascade_enabled=cfg["cascade_enabled"],
            partial_info=cfg["partial_info"],
            adversarial=cfg["adversarial"],
            forced_disruptions=cfg["forced_disruptions"],
        )

        obs = self._build_observation(
            fill_rates={l.line_id: 1.0 for l in self._network.production_lines},
            reward=None,
            extra_cost=0.0,
        )
        self._prev_inventory_days = obs.inventory_coverage_days

        return ResetResponse(
            observation=obs,
            task_description=cfg["description"],
            action_space_description=self._action_space_description(),
            episode_length=self._episode_length,
            seed=seed,
        )

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(self, multi_action: MultiAction) -> Observation:
        """
        Execute one day of simulation:
        1. Apply agent actions
        2. Fire disruptions
        3. Simulate material flow + production
        4. Compute reward
        5. Return observation
        """
        if self._is_done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._network is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        self._timestep += 1
        extra_cost = 0.0

        # 1. Apply actions
        for action in multi_action.actions:
            cost = self._apply_action(action)
            extra_cost += cost

        # 2. Tick disruption engine
        active_disruptions = self._disruption_engine.tick(
            suppliers=self._network.suppliers,
            current_day=self._timestep,
        )

        # 3. Tick network (material flow + production)
        fill_rates = self._network.tick(disruptions=active_disruptions)

        # 4. Compute reward
        daily_cost = self._network.get_daily_cost() + extra_cost
        esg_score = self._network.get_esg_weighted_score()
        on_time_rate = self._compute_on_time_rate()
        inv_days = self._compute_inventory_days()

        reward_breakdown = compute_reward(
            fill_rates=fill_rates,
            daily_cost=daily_cost,
            baseline_daily_cost=self._network.baseline_daily_cost,
            esg_score=esg_score,
            on_time_rate=on_time_rate,
            inventory_days=inv_days,
            previous_inventory_days=self._prev_inventory_days,
            active_disruption_count=len(active_disruptions),
            weights=self._config.get("reward_weights"),
            cost_tolerance=self._config.get("cost_tolerance", 0.30),
        )
        self._step_rewards.append(reward_breakdown.total)
        self._prev_inventory_days = inv_days

        # 5. Terminal condition
        done = self._timestep >= self._episode_length
        # Critical failure: production line fully stopped for 3+ consecutive days
        truncated = self._check_critical_failure()
        self._is_done = done or truncated

        obs = self._build_observation(
            fill_rates=fill_rates,
            reward=reward_breakdown,
            extra_cost=extra_cost,
            is_done=done,
            is_truncated=truncated,
        )
        return obs

    # -----------------------------------------------------------------------
    # state()
    # -----------------------------------------------------------------------

    def state(self) -> StateResponse:
        """Return full internal state snapshot (for graders and debugging)."""
        if self._network is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        obs = self._build_observation(
            fill_rates={l.line_id: l.current_throughput_pct
                        for l in self._network.production_lines},
            reward=None,
            extra_cost=0.0,
        )
        return StateResponse(
            observation=obs,
            internal_config={
                "task_id": self._task_id,
                "seed": self._seed,
                "episode_length": self._episode_length,
                "timestep": self._timestep,
                "cascade_enabled": self._config.get("cascade_enabled"),
                "adversarial": self._config.get("adversarial"),
            },
            cumulative_reward=sum(self._step_rewards),
            step_rewards=list(self._step_rewards),
        )

    # -----------------------------------------------------------------------
    # Action dispatch
    # -----------------------------------------------------------------------

    def _apply_action(self, action: Action) -> float:
        """Apply a single action. Returns extra cost incurred."""
        t = action.action_type
        net = self._network
        extra_cost = 0.0

        if t == ActionType.DO_NOTHING:
            pass

        elif t == ActionType.REROUTE_ORDER:
            net.reroute_order(
                from_id=action.from_supplier_id,
                to_id=action.to_supplier_id,
                material=action.material_type,
                quantity=action.quantity or 50.0,
            )

        elif t == ActionType.ADJUST_SAFETY_STOCK:
            extra_cost = net.adjust_safety_stock(
                supplier_id=action.supplier_id,
                target_days=action.target_stock_days or 7.0,
            )

        elif t == ActionType.EXPEDITE_SHIPMENT:
            extra_cost = net.expedite_shipment(
                order_id_or_material=action.material_type or action.order_id or "",
                expedite_factor=action.expedite_factor or 2.0,
            )

        elif t == ActionType.ACCEPT_SUBSTITUTE:
            net.accept_substitute(
                supplier_id=action.substitute_supplier_id or "",
                quality_penalty=action.quality_penalty_pct or 0.05,
            )

        elif t == ActionType.NEGOTIATE_LEAD_TIME:
            extra_cost = net.negotiate_lead_time(
                supplier_id=action.supplier_id or "",
                target_days=action.target_lead_time_days or 7,
                cost_premium_pct=action.cost_premium_pct or 10.0,
            )

        return extra_cost

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_observation(
        self,
        fill_rates: Dict[str, float],
        reward: Optional[RewardBreakdown],
        extra_cost: float,
        is_done: bool = False,
        is_truncated: bool = False,
    ) -> Observation:
        net = self._network
        de = self._disruption_engine
        active_disruptions = de.active if de else {}
        forecast = de.get_forecast() if de else []

        fill_rate_mean = (
            sum(fill_rates.values()) / len(fill_rates) if fill_rates else 1.0
        )
        daily_cost = net.get_daily_cost() + extra_cost
        cost_vs_baseline = daily_cost / max(1.0, net.baseline_daily_cost)
        inv_days = self._compute_inventory_days()
        on_time = self._compute_on_time_rate()
        esg = net.get_esg_weighted_score()

        return Observation(
            timestep=self._timestep,
            day=self._timestep,
            episode_length=self._episode_length,
            suppliers=net.snapshot(),
            active_disruptions=dict(active_disruptions),
            current_routes=list(net.active_routes),
            production_lines=list(net.production_lines),
            fill_rate=round(fill_rate_mean, 4),
            inventory_coverage_days=round(inv_days, 2),
            total_cost_today=round(daily_cost, 2),
            cost_vs_baseline=round(cost_vs_baseline, 4),
            on_time_delivery_rate=round(on_time, 4),
            esg_score_weighted=round(esg, 4),
            disruption_forecast=forecast,
            reward_breakdown=reward,
            is_done=is_done,
            is_truncated=is_truncated,
            info={
                "task_id": self._task_id,
                "seed": self._seed,
                "cumulative_reward": round(sum(self._step_rewards), 4),
                "episode_score": round(
                    episode_score(self._step_rewards, self._episode_length), 4
                ),
            },
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _compute_inventory_days(self) -> float:
        """Estimate OEM buffer in days of demand."""
        if not self._network:
            return 0.0
        oem = self._network.suppliers.get("OEM0")
        if not oem:
            return 0.0
        total_daily_demand = sum(
            l.daily_demand for l in self._network.production_lines
        )
        if total_daily_demand <= 0:
            return 0.0
        # Proxy: sum of T1 assembler inventories / daily demand
        t1_inv = sum(
            s.inventory_held
            for s in self._network.suppliers.values()
            if s.tier.value == 1
        )
        return t1_inv / total_daily_demand

    def _compute_on_time_rate(self) -> float:
        """
        Fraction of routes delivering within their nominal lead time.
        Proxy: routes where current lead_time <= base_lead_time.
        """
        if not self._network:
            return 1.0
        routes = self._network.active_routes
        if not routes:
            return 1.0
        on_time = sum(
            1 for r in routes
            if r.lead_time_days <= self._network.suppliers[
                r.from_supplier_id
            ].base_lead_time_days
        )
        return on_time / len(routes)

    def _check_critical_failure(self) -> bool:
        """
        Truncate episode if every production line has been at <5% for
        3+ consecutive steps.
        """
        if not self._network:
            return False
        all_lines_stopped = all(
            l.current_throughput_pct < 0.05
            for l in self._network.production_lines
        )
        # Use last 3 rewards as proxy (all near-zero = lines stopped)
        if len(self._step_rewards) >= 3:
            recent = self._step_rewards[-3:]
            if all_lines_stopped and all(r < 0.05 for r in recent):
                return True
        return False

    def _action_space_description(self) -> str:
        return """
Actions (MultiAction bundles any combination):
  - do_nothing: pass
  - reroute_order: {from_supplier_id, to_supplier_id, material_type, quantity}
  - adjust_safety_stock: {supplier_id, target_stock_days}
  - expedite_shipment: {material_type, expedite_factor (1.5–4.0)}
  - accept_substitute: {substitute_supplier_id, quality_penalty_pct}
  - negotiate_lead_time: {supplier_id, target_lead_time_days, cost_premium_pct}

Supplier IDs: S401,S402,S403 (T4) | S301,S302,S303 (T3)
              S201,S202,S203 (T2) | S101,S102 (T1) | OEM0

Materials: lithium_ore, bio_resin_raw, solvents, polymer_base,
           refined_lithium, battery_electrolyte, pcb_chemicals, molded_resin,
           battery_cells, circuit_boards, housings,
           sub_assembly_A, sub_assembly_B
"""
