"""
MediumGrader — scores Task 2 (medium).

Scenario recap:
  Day 7 : Geopolitical block on S401 (T4, lithium ore) — CRITICAL, 25 days.
  Day 12: Cascade to S301 (T3, smelter) — MODERATE, 10 days.
  Random background disruptions (p=0.015). Partial info. 60-day episode.

Extra checks:
  - Did agent identify and respond to the T4→T3 cascade?
  - Did agent find an alternative lithium source (S402) before day 12?
  - Was cost within 30% of baseline across the long episode?
  - ESG maintained despite needing to use backup suppliers?

Scoring is balanced across all six dimensions (harder task → stiffer weights).
"""
from __future__ import annotations
from typing import Dict, List

from .base_grader import BaseGrader, AgentFn, GradeResult
from supply_chain_env.models import Observation, ActionType, MultiAction


class MediumGrader(BaseGrader):

    task_id = "medium"
    episode_length = 60
    BLOCK_DAY = 7
    CASCADE_DAY = 12
    BLOCK_SUPPLIER = "S401"
    CASCADE_SUPPLIER = "S301"
    ALT_ORE_SUPPLIER = "S402"   # bio-based alternative for reroute
    RESPONSE_WINDOW_DAYS = 2    # stricter than easy
    COST_TOLERANCE = 0.30

    dimension_weights = dict(
        production_continuity = 0.30,
        crisis_response_speed = 0.20,
        cost_discipline       = 0.20,
        esg_management        = 0.10,
        resilience            = 0.10,
        recovery_completeness = 0.10,
    )

    _last_obs_log: List[Observation] = []
    _last_action_log: List[MultiAction] = []

    def grade(self, agent_fn: AgentFn, seed: int = 42) -> GradeResult:
        result = super().grade(agent_fn, seed)
        extra = self._task_specific_checks(self._last_obs_log, self._last_action_log)

        result.breakdown["cascade_handling"]  = extra["cascade_handling"]
        result.breakdown["alt_source_found"]  = extra["alt_source_found"]

        ext_weights = {k: v * 0.90 for k, v in self.dimension_weights.items()}
        ext_weights["cascade_handling"] = 0.06
        ext_weights["alt_source_found"] = 0.04

        total = sum(result.breakdown.get(d, 0.0) * w for d, w in ext_weights.items())
        result.score = round(min(1.0, max(0.0, total)), 4)
        return result

    def _compute_breakdown(self, obs_log, action_log, step_rewards):
        self._last_obs_log = obs_log
        self._last_action_log = action_log
        return super()._compute_breakdown(obs_log, action_log, step_rewards)

    # -----------------------------------------------------------------------
    # Task-specific checks
    # -----------------------------------------------------------------------

    def _task_specific_checks(
        self,
        obs_log: List[Observation],
        action_log: List[MultiAction],
    ) -> Dict[str, float]:

        # Check 1: Did agent handle the cascade?
        # Proxy: did agent take any action targeting S301 between day 12–16?
        cascade_action_found = False
        for obs, act in zip(obs_log, action_log):
            if self.CASCADE_DAY <= obs.timestep <= self.CASCADE_DAY + 4:
                for a in act.actions:
                    if (a.action_type in (
                            ActionType.REROUTE_ORDER,
                            ActionType.ADJUST_SAFETY_STOCK,
                            ActionType.EXPEDITE_SHIPMENT,
                            ActionType.NEGOTIATE_LEAD_TIME,
                    ) and (
                            getattr(a, "from_supplier_id", "") == self.CASCADE_SUPPLIER
                            or getattr(a, "supplier_id", "") == self.CASCADE_SUPPLIER
                    )):
                        cascade_action_found = True

        # Partial credit: even if exact supplier not targeted, if agent
        # reacted to any disruption around cascade window, give 0.5
        if not cascade_action_found:
            for obs, act in zip(obs_log, action_log):
                if self.CASCADE_DAY <= obs.timestep <= self.CASCADE_DAY + 5:
                    if any(a.action_type.value != "do_nothing" for a in act.actions):
                        cascade_action_found = None   # partial (None → 0.5)
                        break

        if cascade_action_found is True:
            cascade_handling = 1.0
        elif cascade_action_found is None:
            cascade_handling = 0.5
        else:
            cascade_handling = 0.0

        # Check 2: Did agent find an alt lithium source before cascade hit?
        alt_source_found = 0.0
        for obs, act in zip(obs_log, action_log):
            if obs.timestep > self.CASCADE_DAY:
                break
            for a in act.actions:
                if (a.action_type == ActionType.REROUTE_ORDER and
                        a.to_supplier_id == self.ALT_ORE_SUPPLIER):
                    # Earlier is better
                    latency = max(0, obs.timestep - self.BLOCK_DAY)
                    alt_source_found = max(alt_source_found,
                                           1.0 - latency / 5)

        return dict(
            cascade_handling=round(cascade_handling, 4),
            alt_source_found=round(alt_source_found, 4),
        )

    def _generate_feedback(self, obs_log, action_log, breakdown):
        fb = super()._generate_feedback(obs_log, action_log, breakdown)

        ch = breakdown.get("cascade_handling", 0.0)
        af = breakdown.get("alt_source_found", 0.0)

        if ch >= 0.90:
            fb.append("[Task-specific] Excellent cascade handling: S301 was addressed quickly "
                      "after the T4→T3 cascade.")
        elif ch >= 0.50:
            fb.append("[Task-specific] Partial cascade response: some reaction, but S301 "
                      "wasn't explicitly targeted. The cascade created a smelter shortage — "
                      "expedite or reroute refined_lithium immediately.")
        else:
            fb.append("[Task-specific] Cascade missed: no action on S301 after cascade from "
                      "S401. In multi-tier environments, monitor ALL downstream suppliers "
                      "when a T4 disruption fires — cascades arrive ~5 days later.")

        if af >= 0.80:
            fb.append("[Task-specific] Great alt-sourcing: found S402 as lithium backup "
                      "before the cascade materialised.")
        elif af > 0:
            fb.append("[Task-specific] Late alt-sourcing: rerouted to S402 but after cascade "
                      "had already hit. Aim to reroute within 2 days of geopolitical block.")
        else:
            fb.append("[Task-specific] Alt source not found: S402 (Agri Corp Beta) can supply "
                      "bio_resin_raw as a lithium-ore bridge. "
                      "Action: reroute_order(from='S401', to='S402', material='lithium_ore').")

        return fb
