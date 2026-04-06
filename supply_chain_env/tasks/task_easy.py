"""
Task 1 — Easy: Single-Tier Crisis Response

Scenario:
  A factory fire at Battery Manufacturing Eta (S201, Tier-2) causes an
  SEVERE disruption on day 5 that lasts 12 days. The agent has full
  visibility of all disruptions and a simple 30-day horizon.

  The agent's job:
    1. Detect the disruption on day 5.
    2. Reroute battery_cell supply away from S201.
    3. Manage safety stock at Tier-1 assemblers.
    4. Restore normal routing after S201 recovers (day 17).

Learning objectives:
  - Basic disruption detection (observe active_disruptions).
  - Reactive rerouting (reroute_order action).
  - Simple safety-stock management.

Reward structure:
  production_continuity: 0.60  (dominant — keep the line running)
  cost_efficiency:       0.20
  esg_compliance:        0.10
  lead_time_adherence:   0.10
  cost_tolerance:        40%   (lenient — learning task)

Baseline heuristic score: ~0.72
Random agent score:        ~0.35
"""

TASK_ID = "easy"

# -----------------------------------------------------------------------
# Optimal action hints (used by oracle agent / ablation studies)
# -----------------------------------------------------------------------

def optimal_action_hint(obs) -> str:
    """Returns a natural-language description of the optimal action."""
    disrupted = {d.affected_supplier_id for d in obs.active_disruptions.values()}
    if "S201" in disrupted and obs.timestep <= 8:
        return (
            "S201 is disrupted. Immediately reroute battery_cells away from S201 "
            "to S202 (PCB Fab, same tier, can absorb demand). "
            "Also adjust safety stock at S101 and S102 to 10 days."
        )
    if obs.inventory_coverage_days < 4.0 and not disrupted:
        return (
            "Inventory buffer is low. Call adjust_safety_stock on S101 and S102 "
            "to build 7–10 days coverage before a disruption can happen."
        )
    if "S201" not in disrupted and obs.timestep > 17:
        return (
            "S201 has recovered. Reroute battery_cells back to S201 "
            "to restore primary supply and reduce cost."
        )
    return "Situation stable. Do nothing or build safety stock proactively."


# -----------------------------------------------------------------------
# Observation field guide (for LLM agents)
# -----------------------------------------------------------------------

OBSERVATION_GUIDE = """
KEY FIELDS TO MONITOR (easy task):

obs.active_disruptions
    Dict of currently active disruptions. Check every step.
    Key fields per disruption:
      .affected_supplier_id  — which supplier is hit
      .severity              — minor/moderate/severe/critical
      .capacity_impact       — fraction of capacity lost (0.0–1.0)
      .duration_remaining_days — how many more days it lasts

obs.suppliers["S201"]
    .capacity_pct          — current capacity (will drop to 0.2 on day 5)
    .inventory_held        — units in stock at this node
    .is_available          — False if capacity_pct == 0

obs.fill_rate              — fraction of daily demand being met (target: >= 0.90)
obs.inventory_coverage_days — days of buffer at T1 assemblers (target: >= 7)
obs.cost_vs_baseline       — keep <= 1.40

CRITICAL SUPPLIER PAIRS (rerouting options):
  S201 (Battery Mfg, China)  →  backup: S202 (PCB Fab, Taiwan)
  S202 (PCB Fab, Taiwan)     →  backup: S203 (Housing Mold, Mexico)
  S101 (Assembler, Vietnam)  →  S102 (Assembler, India) and vice-versa

MATERIALS NEEDED BY OEM:
  sub_assembly_A  from S101
  sub_assembly_B  from S102
  Both needed for full production.
"""

# -----------------------------------------------------------------------
# Grading rubric
# -----------------------------------------------------------------------

GRADING_RUBRIC = {
    "score_0.9_to_1.0": (
        "Detected disruption on day 5, rerouted within 1 day, "
        "maintained >90% fill rate throughout, restored S201 routing after recovery."
    ),
    "score_0.7_to_0.9": (
        "Rerouted within 3 days, fill rate dipped but stayed above 70%, "
        "cost within 40% of baseline."
    ),
    "score_0.5_to_0.7": (
        "Rerouted but with 4+ day delay, fill rate dropped below 50% "
        "for multiple days, or significant cost overrun."
    ),
    "score_below_0.5": (
        "Did not reroute, or rerouted incorrectly, causing extended line stoppage "
        "and/or critical cost overrun."
    ),
}
