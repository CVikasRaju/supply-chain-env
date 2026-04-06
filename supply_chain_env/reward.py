"""
Reward function for the Supply Chain Disruption Navigator.

Reward = production_continuity × cost_efficiency × esg_compliance
         + resilience_bonus

All components are in [0, 1]. Total reward in [0, 1.0 + 0.2 bonus].
Partial signals on every dimension so agents get dense learning signal.
"""
from __future__ import annotations
from typing import Dict, List

from .models import RewardBreakdown


# ---------------------------------------------------------------------------
# Weights (tunable per task config)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = dict(
    production_continuity=0.50,
    cost_efficiency=0.25,
    esg_compliance=0.15,
    lead_time_adherence=0.10,
)


def compute_reward(
    fill_rates: Dict[str, float],       # line_id -> 0..1
    daily_cost: float,
    baseline_daily_cost: float,
    esg_score: float,                   # 0..1
    on_time_rate: float,                # 0..1
    inventory_days: float,              # days of coverage at OEM
    previous_inventory_days: float,
    active_disruption_count: int,
    weights: Dict[str, float] = None,
    cost_tolerance: float = 0.30,       # allow up to 30% over baseline
) -> RewardBreakdown:
    """
    Compute one-step reward with full decomposition.

    Args:
        fill_rates: per production-line fill rate (0=line stopped, 1=full)
        daily_cost: total cost of all active routes today
        baseline_daily_cost: cost with no disruptions and default routes
        esg_score: weighted ESG score of active supplier network
        on_time_rate: fraction of shipments arriving on time
        inventory_days: current OEM buffer in days of demand
        previous_inventory_days: buffer at previous step (for resilience delta)
        active_disruption_count: number of active disruptions this step
        weights: override default component weights
        cost_tolerance: fraction above baseline that is still "efficient"

    Returns:
        RewardBreakdown with total and per-component scores
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # 1. Production continuity
    #    Mean fill rate across all lines, penalised harder for line stoppages
    if fill_rates:
        avg_fill = sum(fill_rates.values()) / len(fill_rates)
        # Extra penalty: any line fully stopped → -0.1 per stopped line
        n_stopped = sum(1 for v in fill_rates.values() if v < 0.05)
        stoppage_penalty = n_stopped * 0.10
        production_continuity = max(0.0, avg_fill - stoppage_penalty)
    else:
        production_continuity = 0.0

    # 2. Cost efficiency
    #    Full score at baseline cost, zero score at (1 + cost_tolerance) × baseline
    cost_ratio = daily_cost / max(1.0, baseline_daily_cost)
    if cost_ratio <= 1.0:
        cost_efficiency = 1.0
    elif cost_ratio <= (1.0 + cost_tolerance):
        # Linear decay from 1.0 to 0.0 across the tolerance band
        cost_efficiency = 1.0 - (cost_ratio - 1.0) / cost_tolerance
    else:
        cost_efficiency = 0.0

    # 3. ESG compliance
    esg_compliance = float(esg_score)

    # 4. Lead time adherence
    lead_time_adherence = float(on_time_rate)

    # 5. Resilience bonus
    #    Agent is rewarded for proactively building buffer BEFORE disruptions
    resilience_bonus = _compute_resilience_bonus(
        inventory_days=inventory_days,
        previous_inventory_days=previous_inventory_days,
        active_disruption_count=active_disruption_count,
    )

    # Weighted total (multiplicative rule as per spec: three dimensions)
    total = (production_continuity * cost_efficiency * esg_compliance) + resilience_bonus
    total = min(1.0, total)

    return RewardBreakdown(
        total=round(total, 4),
        production_continuity=round(production_continuity, 4),
        cost_efficiency=round(cost_efficiency, 4),
        esg_compliance=round(esg_compliance, 4),
        lead_time_adherence=round(lead_time_adherence, 4),
        resilience_bonus=round(resilience_bonus, 4),
    )


def _compute_resilience_bonus(
    inventory_days: float,
    previous_inventory_days: float,
    active_disruption_count: int,
) -> float:
    """
    Up to +0.20 bonus for:
    - Building safety stock proactively (when disruptions active)
    - Maintaining > 7 days of buffer
    """
    bonus = 0.0
    # Bonus for holding buffer during active disruptions
    if active_disruption_count > 0 and inventory_days >= 7.0:
        bonus += 0.10
    # Bonus for increasing inventory when under threat
    if active_disruption_count > 0 and inventory_days > previous_inventory_days:
        bonus += 0.05
    # Buffer level bonus (independent of disruptions)
    if inventory_days >= 14.0:
        bonus += 0.05
    return min(0.20, bonus)


def episode_score(step_rewards: List[float], episode_length: int) -> float:
    """
    Final episode score for graders: mean step reward, normalised.
    Penalises short episodes (agent crashed a line early).
    """
    if not step_rewards:
        return 0.0
    mean_reward = sum(step_rewards) / max(1, len(step_rewards))
    # Completeness factor: if episode ended early due to critical failure
    completeness = len(step_rewards) / max(1, episode_length)
    return round(mean_reward * completeness, 4)
