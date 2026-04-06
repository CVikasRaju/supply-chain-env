"""
Task 2 — Medium: Multi-Tier Cascade with Partial Information

Scenario:
  A geopolitical export ban hits Mining Co Alpha (S401, Tier-4) on day 7.
  The lithium ore supply from Chile is completely blocked (CRITICAL severity,
  25-day duration). Five days later (day 12), this cascades downstream to
  Smelter Delta (S301, Tier-3) which runs out of raw material and enters a
  MODERATE 10-day disruption. Random background disruptions also fire
  (p=0.015 per supplier per day). Partial information: agent gets no forecast.

  The agent must:
    1. Detect the T4 block on day 7.
    2. Find an alternative lithium ore source (S402 is closest substitute).
    3. Anticipate the T3 cascade and pre-position stock at T2 before day 12.
    4. Handle background disruptions with no advance warning.
    5. Maintain cost and ESG targets across 60 days.

Learning objectives:
  - Multi-tier awareness (T4 disruption → T3 consequence).
  - Proactive vs reactive decision-making.
  - Managing without forecast (partial information).
  - Long-horizon cost control.

Reward structure:
  production_continuity: 0.50
  cost_efficiency:       0.25
  esg_compliance:        0.15
  lead_time_adherence:   0.10
  cost_tolerance:        30%

Baseline heuristic score: ~0.58
Random agent score:        ~0.28
"""

TASK_ID = "medium"

# -----------------------------------------------------------------------
# Cascade map (which T4 disruption implies which T3 consequence)
# -----------------------------------------------------------------------

CASCADE_MAP = {
    "S401": {
        "downstream": "S301",
        "delay_days": 5,
        "description": "S401 export block starves S301 of lithium ore input",
        "mitigating_action": "reroute_order(from='S401', to='S402', material='lithium_ore')"
    },
    "S403": {
        "downstream": "S303",
        "delay_days": 4,
        "description": "S403 polymer disruption starves S303 of polymer_base",
        "mitigating_action": "adjust_safety_stock(supplier_id='S303', target_stock_days=12)"
    },
}


def cascade_warning(obs) -> str:
    """
    Given current observation, return a cascade warning if a T4 supplier
    is disrupted and cascade hasn't hit yet.
    """
    warnings = []
    for d in obs.active_disruptions.values():
        s = obs.suppliers.get(d.affected_supplier_id)
        if s and s.tier.value == 4:
            cascade_info = CASCADE_MAP.get(d.affected_supplier_id)
            if cascade_info:
                warnings.append(
                    f"WARNING: {d.affected_supplier_id} (T4) is disrupted. "
                    f"Expect cascade to {cascade_info['downstream']} (T3) "
                    f"in ~{cascade_info['delay_days']} days. "
                    f"Recommended action: {cascade_info['mitigating_action']}"
                )
    return "\n".join(warnings) if warnings else ""


# -----------------------------------------------------------------------
# Observation guide
# -----------------------------------------------------------------------

OBSERVATION_GUIDE = """
KEY FIELDS TO MONITOR (medium task):

obs.active_disruptions
    No forecast is given. Check suppliers across ALL tiers, not just T1/T2.
    A T4 disruption today → T3 disruption in ~5 days (cascade).

Cascade logic:
  S401 disrupted → S301 will run short of lithium_ore input in 5 days
  S403 disrupted → S303 will run short of polymer_base in 4 days

Pre-emptive actions on detecting T4 disruption:
  1. reroute_order(from='S401', to='S402', ...) — swap ore source
  2. adjust_safety_stock(supplier_id='S301', target_stock_days=14) — buy time
  3. negotiate_lead_time(supplier_id='S302', ...) — ensure T3 alt is fast

Background disruptions (no warning):
  Monitor obs.suppliers[*].capacity_pct each step.
  Any supplier dropping below 0.6 warrants an immediate reroute.

Cost management across 60 days:
  obs.cost_vs_baseline  — keep ≤ 1.30 on average
  Expediting for >3 consecutive days will blow the budget.
  Prefer: negotiate_lead_time (one-time cost premium) over expedite_shipment.
"""

GRADING_RUBRIC = {
    "score_0.85_to_1.0": (
        "Rerouted S401→S402 within 2 days of block. Pre-stocked S301 before cascade. "
        "Handled all background disruptions reactively. Cost ≤ 1.25× throughout."
    ),
    "score_0.65_to_0.85": (
        "Managed T4 block and cascade but with some delay. Cost within 1.30×. "
        "Fill rate > 75% for ≥ 50 of 60 days."
    ),
    "score_0.45_to_0.65": (
        "Reacted to cascade after it hit (not before). Partial recovery. "
        "Occasional cost overruns."
    ),
    "score_below_0.45": (
        "Did not address T4 block or cascade. Extended line stoppages. "
        "Budget significantly exceeded."
    ),
}
