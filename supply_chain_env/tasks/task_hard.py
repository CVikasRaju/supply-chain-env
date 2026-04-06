"""
Task 3 — Hard: Adversarial Multi-Tier Disruptions

Scenario:
  Day 3 : Pandemic closure hits Chem Plant Epsilon (S301, T3) — SEVERE, 20 days.
  Day 10: Quality failure at PCB Fab Theta (S202, T2) — MODERATE, 8 days.
  Background: high-frequency stochastic disruptions (p=0.04/supplier/day),
  adversarially clustered on T2/T1 suppliers (2.5× base rate).
  No forecast. Cost tolerance 20%. ESG weight 20%. 90-day episode.

  The agent must:
    1. Navigate two simultaneous scripted disruptions early.
    2. Manage 5+ concurrent stochastic disruptions across the episode.
    3. Maintain portfolio diversification (no single-source dependencies).
    4. Keep ESG score ≥ 0.65 while staying within 20% of baseline cost.
    5. Hold ≥ 7 days of buffer inventory across all 90 days.
    6. Avoid critical failures (all lines stopped for 3 consecutive days).

Learning objectives:
  - Simultaneous multi-tier crisis management.
  - ESG-cost optimisation under adversarial conditions.
  - Long-horizon supply chain resilience strategy.
  - Portfolio diversification vs concentration risk.

Reward structure:
  production_continuity: 0.40  (reduced weight — adversarial pressure is expected)
  cost_efficiency:       0.25
  esg_compliance:        0.20  (highest of all tasks)
  lead_time_adherence:   0.15
  cost_tolerance:        20%   (tightest)

Baseline heuristic score: ~0.41
Random agent score:        ~0.18
State-of-the-art target:   ~0.75
"""

TASK_ID = "hard"

# -----------------------------------------------------------------------
# Supplier decision matrix (for planning-based agents)
# -----------------------------------------------------------------------

# For each material, ranked list of (supplier_id, esg_score, unit_cost)
# Agents should prefer high-ESG, then low-cost
SUPPLIER_DECISION_MATRIX = {
    "lithium_ore": [
        ("S401", 0.55, 8.0),   # primary (Chile)
        ("S402", 0.70, 3.0),   # backup (Brazil) — higher ESG, lower cost!
    ],
    "refined_lithium": [
        ("S301", 0.50, 22.0),  # primary (China)
        ("S302", 0.80, 35.0),  # backup (Germany) — high ESG, higher cost
    ],
    "pcb_chemicals": [
        ("S302", 0.80, 35.0),  # primary (Germany, high ESG)
        ("S303", 0.65, 18.0),  # backup (Korea)
    ],
    "battery_cells": [
        ("S201", 0.60, 80.0),  # primary (China)
        # No direct substitute — build safety stock or accept delay
    ],
    "circuit_boards": [
        ("S202", 0.72, 45.0),  # primary (Taiwan)
        # S202 substitute: use S203 housing + manual assembly (quality penalty)
    ],
    "sub_assembly_A": [
        ("S101", 0.65, 60.0),  # primary (Vietnam)
        ("S102", 0.70, 50.0),  # can cross-supply
    ],
    "sub_assembly_B": [
        ("S102", 0.70, 50.0),  # primary (India)
        ("S101", 0.65, 60.0),  # can cross-supply
    ],
}

# -----------------------------------------------------------------------
# ESG-optimal rerouting recommendations
# -----------------------------------------------------------------------

ESG_REROUTE_MAP = {
    # If S301 (ESG=0.50) disrupted → prefer S302 (ESG=0.80) even at cost premium
    "S301": ("S302", "refined_lithium", "Higher cost but ESG score lifts significantly"),
    # If S201 (ESG=0.60) disrupted → no high-ESG alternative; hold stock
    "S201": (None, "battery_cells", "No high-ESG alternative. Build safety stock."),
    # If S403 (ESG=0.40) disrupted → use S402 (ESG=0.70) — also cheaper!
    "S403": ("S402", "bio_resin_raw", "S402 is both cheaper AND higher ESG"),
}


def esg_advice(obs) -> str:
    """Return ESG optimisation advice given current routing."""
    advice = []
    active_low_esg = [
        r.from_supplier_id for r in obs.current_routes
        if r.from_supplier_id in {"S403", "S301", "S201"}
    ]
    for sid in set(active_low_esg):
        info = ESG_REROUTE_MAP.get(sid)
        if info:
            alt, mat, reason = info
            if alt:
                advice.append(
                    f"ESG: {sid} has low ESG score. "
                    f"Consider rerouting {mat} to {alt}. Reason: {reason}"
                )
            else:
                advice.append(
                    f"ESG: {sid} has low ESG score. {reason}"
                )
    if obs.esg_score_weighted < 0.60:
        advice.append(
            "ESG ALERT: Weighted ESG score has dropped below 0.60. "
            "Priority reroutes: S403→S402, S301→S302."
        )
    return "\n".join(advice) if advice else "ESG status: within target."


# -----------------------------------------------------------------------
# Observation guide
# -----------------------------------------------------------------------

OBSERVATION_GUIDE = """
KEY FIELDS TO MONITOR (hard task):

obs.active_disruptions       — check EVERY step, no forecast available
obs.esg_score_weighted        — target ≥ 0.65; below 0.55 = ESG failure
obs.cost_vs_baseline          — hard limit: ≤ 1.20; breaches compound over 90 days
obs.inventory_coverage_days   — target ≥ 7.0; below 3.0 is critical
obs.fill_rate                 — adversarial target ≥ 0.70 on average

Adversarial clustering:
  T2/T1 suppliers (S201, S202, S203, S101, S102) face 2.5× disruption rate.
  Never rely on a single T2 or T1 supplier for any material.
  Maintain parallel active routes for sub_assembly_A and sub_assembly_B.

Anti-patterns to avoid:
  × Expediting every disruption → cost explodes within 20 days
  × Ignoring ESG in favour of cheapest suppliers → ESG dimension tanks
  × Only reacting, never proactively stocking → inventory crashes under adversarial bursts
  × Letting S301 or S201 stay as sole source for a tier → single point of failure

Recommended strategy:
  ✓ Keep 2 active routes for every critical material
  ✓ Negotiate lead times (one-time cost) rather than expediting (recurring)
  ✓ Check esg_score_weighted every step; if < 0.65, switch to S302/S402
  ✓ Adjust safety stock every 15 days regardless of disruption state
  ✓ When 3+ disruptions active simultaneously, focus on T1 assemblers first
"""

GRADING_RUBRIC = {
    "score_0.75_to_1.0": (
        "Maintained ≥0.80 fill rate, ESG ≥0.70, cost ≤1.15×, "
        "inventory ≥7 days throughout 90-day adversarial episode. "
        "Expert-level supply chain management."
    ),
    "score_0.55_to_0.75": (
        "Survived without critical failure. Fill rate ≥0.65 on average. "
        "Occasional ESG or cost breaches. No extended line stoppages."
    ),
    "score_0.35_to_0.55": (
        "Significant disruptions caused production drops. "
        "Either ESG, cost, or inventory management was consistently poor."
    ),
    "score_below_0.35": (
        "Critical failure (all lines stopped 3+ days) or "
        "severe and persistent underperformance on all dimensions."
    ),
}
