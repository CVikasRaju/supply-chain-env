"""
EasyGrader — scores Task 1 (easy).

Scenario recap:
  Day 5: Battery Mfg Eta (S201, T2) — factory fire, severity=SEVERE, 12 days.
  Full disruption visibility. 30-day episode.

Extra checks specific to easy task:
  - Did agent reroute away from S201 before day 8?
  - Did agent avoid using S201 during the disruption window?
  - Did production fully recover by day 20?

Scoring weights are more lenient (beginner task).
"""
from __future__ import annotations
from typing import Dict, List

from .base_grader import BaseGrader, AgentFn, GradeResult
from supply_chain_env.models import Observation, ActionType, MultiAction


class EasyGrader(BaseGrader):

    task_id = "easy"
    episode_length = 30
    DISRUPTION_DAY = 5
    DISRUPTION_END_DAY = 17    # day 5 + 12 days duration
    DISRUPTION_SUPPLIER = "S201"
    COST_TOLERANCE = 0.40      # more forgiving on cost for easy task

    dimension_weights = dict(
        production_continuity = 0.35,
        crisis_response_speed = 0.25,
        cost_discipline       = 0.15,
        esg_management        = 0.05,
        resilience            = 0.10,
        recovery_completeness = 0.10,
    )

    # -----------------------------------------------------------------------
    # Override grade to inject task-specific checks
    # -----------------------------------------------------------------------

    def grade(self, agent_fn: AgentFn, seed: int = 42) -> GradeResult:
        result = super().grade(agent_fn, seed)

        # Add task-specific dimension
        obs_log = self._last_obs_log
        action_log = self._last_action_log
        extra = self._task_specific_checks(obs_log, action_log)

        # Blend in: replace resilience dimension with task-specific score
        result.breakdown["s201_avoidance"] = extra.get("s201_avoidance", 0.0)
        result.breakdown["early_reroute"]  = extra.get("early_reroute", 0.0)

        # Recompute score with extended dimensions
        ext_weights = dict(self.dimension_weights)
        ext_weights["s201_avoidance"] = 0.05
        ext_weights["early_reroute"]  = 0.05
        # Rescale original weights to sum to 0.90
        for k in self.dimension_weights:
            ext_weights[k] = self.dimension_weights[k] * 0.90

        total = sum(result.breakdown.get(d, 0.0) * w for d, w in ext_weights.items())
        result.score = round(min(1.0, max(0.0, total)), 4)
        return result

    # -----------------------------------------------------------------------
    # Intercept obs/action logs from parent run loop
    # -----------------------------------------------------------------------

    _last_obs_log: List[Observation] = []
    _last_action_log: List[MultiAction] = []

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

        # Check 1: Did agent reroute away from S201 within 3 days of disruption?
        early_reroute = 0.0
        for obs, act in zip(obs_log, action_log):
            if self.DISRUPTION_DAY <= obs.timestep <= self.DISRUPTION_DAY + 3:
                for a in act.actions:
                    if (a.action_type == ActionType.REROUTE_ORDER and
                            a.from_supplier_id == self.DISRUPTION_SUPPLIER):
                        delay = obs.timestep - self.DISRUPTION_DAY
                        early_reroute = max(0.0, 1.0 - delay / 3)
                        break

        # Check 2: Did agent avoid routing through S201 during blackout?
        # (i.e. no active routes through S201 between day 5–17)
        s201_routes_during_disruption = 0
        total_disruption_steps = 0
        for obs in obs_log:
            if self.DISRUPTION_DAY <= obs.timestep <= self.DISRUPTION_END_DAY:
                total_disruption_steps += 1
                for route in obs.current_routes:
                    if route.from_supplier_id == self.DISRUPTION_SUPPLIER:
                        s201_routes_during_disruption += 1
                        break

        if total_disruption_steps > 0:
            s201_avoidance = 1.0 - (s201_routes_during_disruption / total_disruption_steps)
        else:
            s201_avoidance = 1.0

        return dict(
            early_reroute=round(early_reroute, 4),
            s201_avoidance=round(s201_avoidance, 4),
        )

    def _generate_feedback(self, obs_log, action_log, breakdown):
        fb = super()._generate_feedback(obs_log, action_log, breakdown)

        # Task-specific feedback
        er = breakdown.get("early_reroute", 0.0)
        av = breakdown.get("s201_avoidance", 0.0)

        if er >= 0.80:
            fb.append("[Task-specific] Excellent: rerouted away from S201 quickly after fire.")
        elif er >= 0.40:
            fb.append("[Task-specific] Moderate: rerouted from S201 but with some delay.")
        else:
            fb.append("[Task-specific] Missed: did not reroute away from S201 after factory fire. "
                      "Action: reroute_order(from='S201', to='S202', material='battery_cells').")

        if av >= 0.80:
            fb.append("[Task-specific] Good avoidance: S201 was not used during the blackout window.")
        else:
            fb.append("[Task-specific] S201 was still routing material during its disruption — "
                      "this dragged down production. Remove routes through disrupted suppliers.")

        return fb
