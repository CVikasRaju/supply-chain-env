"""
HardGrader — scores Task 3 (hard).

Scenario recap:
  Day 3 : Pandemic closure on S301 (T3) — SEVERE, 20 days.
  Day 10: Quality failure on S202 (T2) — MODERATE, 8 days.
  High background disruption rate (p=0.04). Adversarial clustering.
  No forecast. ESG and cost both tightly constrained. 90-day episode.

Extra checks specific to hard task:
  - Portfolio diversification: did agent spread risk across multiple T2/T1 suppliers?
  - ESG-cost balance: did agent find low-cost AND high-ESG alternatives?
  - Adversarial survival: what fraction of adversarial disruptions were mitigated within 2 days?
  - Long-horizon planning: did average inventory coverage stay >= 7 days across 90 days?

Weights are the strictest of all three tasks.
"""
from __future__ import annotations
import statistics
from typing import Dict, List, Optional

from .base_grader import BaseGrader, AgentFn, GradeResult
from supply_chain_env.models import Observation, ActionType, MultiAction, SupplierTier


# High-ESG suppliers (score >= 0.70)
HIGH_ESG_SUPPLIERS = {"S302", "S402", "S202", "S102", "OEM0"}
# Low-ESG suppliers (score < 0.55) — should be de-prioritised
LOW_ESG_SUPPLIERS  = {"S403", "S301", "S201"}


class HardGrader(BaseGrader):

    task_id = "hard"
    episode_length = 90
    RESPONSE_WINDOW_DAYS = 1    # must respond within 1 day in hard mode
    COST_TOLERANCE = 0.20       # tightest cost constraint
    RECOVERY_THRESHOLD = 0.80   # slightly lower bar given adversarial pressure

    dimension_weights = dict(
        production_continuity = 0.25,
        crisis_response_speed = 0.20,
        cost_discipline       = 0.20,
        esg_management        = 0.15,
        resilience            = 0.10,
        recovery_completeness = 0.10,
    )

    _last_obs_log: List[Observation] = []
    _last_action_log: List[MultiAction] = []

    def grade(self, agent_fn: AgentFn, seed: int = 42) -> GradeResult:
        result = super().grade(agent_fn, seed)
        extra = self._task_specific_checks(self._last_obs_log, self._last_action_log)

        result.breakdown["portfolio_diversification"] = extra["portfolio_diversification"]
        result.breakdown["esg_cost_balance"]          = extra["esg_cost_balance"]
        result.breakdown["adversarial_survival"]      = extra["adversarial_survival"]
        result.breakdown["long_horizon_inventory"]    = extra["long_horizon_inventory"]

        # Rescale base weights to 0.75, give 0.25 to task-specific
        ext_weights = {k: v * 0.75 for k, v in self.dimension_weights.items()}
        ext_weights["portfolio_diversification"] = 0.08
        ext_weights["esg_cost_balance"]          = 0.07
        ext_weights["adversarial_survival"]      = 0.06
        ext_weights["long_horizon_inventory"]    = 0.04

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

        # 1. Portfolio diversification
        #    Count distinct active suppliers used across routes per step.
        #    Score = mean(distinct_active_suppliers) / total_possible (11 non-OEM)
        diversification_scores = []
        for obs in obs_log:
            if obs.timestep == 0:
                continue
            active_sids = set()
            for route in obs.current_routes:
                s = obs.suppliers.get(route.from_supplier_id)
                if s and s.is_available:
                    active_sids.add(route.from_supplier_id)
            diversification_scores.append(len(active_sids) / 11.0)
        portfolio_diversification = (
            statistics.mean(diversification_scores) if diversification_scores else 0.0
        )

        # 2. ESG-cost balance
        #    Steps where agent used high-ESG supplier AND cost < 1.2×baseline
        esg_cost_ok = []
        for obs in obs_log:
            if obs.timestep == 0:
                continue
            high_esg_active = any(
                r.from_supplier_id in HIGH_ESG_SUPPLIERS
                for r in obs.current_routes
            )
            low_esg_avoided = not any(
                r.from_supplier_id in LOW_ESG_SUPPLIERS
                for r in obs.current_routes
            )
            cost_ok = obs.cost_vs_baseline <= 1.20
            # Partial: ESG or cost gets 0.5 each
            score = (0.5 * int(high_esg_active and low_esg_avoided) +
                     0.5 * int(cost_ok))
            esg_cost_ok.append(score)
        esg_cost_balance = statistics.mean(esg_cost_ok) if esg_cost_ok else 0.0

        # 3. Adversarial survival
        #    For each disruption that fired, check if agent responded in ≤2 days.
        #    Track disruption IDs seen, measure first real-action after each new one.
        seen_disruptions = set()
        disruption_response_scores = []
        for i, (obs, act) in enumerate(zip(obs_log, action_log)):
            new_disrupt = set(obs.active_disruptions.keys()) - seen_disruptions
            for did in new_disrupt:
                seen_disruptions.add(did)
                # Look for response in next 3 steps
                responded = False
                for j in range(i, min(i + 4, len(action_log))):
                    if any(a.action_type.value != "do_nothing"
                           for a in action_log[j].actions):
                        delay = j - i
                        disruption_response_scores.append(
                            max(0.0, 1.0 - delay / 3)
                        )
                        responded = True
                        break
                if not responded:
                    disruption_response_scores.append(0.0)

        adversarial_survival = (
            statistics.mean(disruption_response_scores)
            if disruption_response_scores else 1.0
        )

        # 4. Long-horizon inventory
        #    Mean inventory_coverage_days over the 90-day episode.
        #    Score = min(1.0, mean_coverage / 10.0)
        coverages = [o.inventory_coverage_days for o in obs_log if o.timestep > 0]
        mean_coverage = statistics.mean(coverages) if coverages else 0.0
        long_horizon_inventory = min(1.0, mean_coverage / 10.0)

        return dict(
            portfolio_diversification = round(portfolio_diversification, 4),
            esg_cost_balance          = round(esg_cost_balance, 4),
            adversarial_survival      = round(adversarial_survival, 4),
            long_horizon_inventory    = round(long_horizon_inventory, 4),
        )

    def _generate_feedback(self, obs_log, action_log, breakdown):
        fb = super()._generate_feedback(obs_log, action_log, breakdown)

        pd_  = breakdown.get("portfolio_diversification", 0.0)
        ecb  = breakdown.get("esg_cost_balance", 0.0)
        adv  = breakdown.get("adversarial_survival", 0.0)
        lhi  = breakdown.get("long_horizon_inventory", 0.0)

        # Portfolio
        if pd_ >= 0.70:
            fb.append("[Hard] Strong portfolio diversification — risk well-spread across tiers.")
        else:
            fb.append(f"[Hard] Low portfolio diversification ({pd_:.0%}). "
                      "In adversarial mode, single-source dependencies are fatal. "
                      "Maintain active routes through at least 7 of 11 non-OEM suppliers.")

        # ESG-cost balance
        if ecb >= 0.75:
            fb.append("[Hard] Excellent ESG-cost balance — managed sustainability without "
                      "significant cost premiums.")
        elif ecb >= 0.50:
            fb.append("[Hard] Moderate ESG-cost balance. Increase use of S302 (Germany, ESG=0.80) "
                      "and S402 (Brazil, ESG=0.70) to lift the ESG score without large cost hits.")
        else:
            fb.append("[Hard] Poor ESG-cost balance. Either ESG was neglected "
                      "or cost overruns from excessive expediting degraded this score. "
                      "Prioritise: S302 > S303 > S301 on ESG grounds; negotiate lead times "
                      "instead of expediting to keep costs in check.")

        # Adversarial survival
        if adv >= 0.80:
            fb.append("[Hard] Outstanding adversarial survival — responded to most disruptions "
                      "within 1–2 days despite no forecast available.")
        elif adv >= 0.50:
            fb.append("[Hard] Acceptable adversarial survival, but some disruptions went "
                      "unaddressed for 3+ days. In hard mode, check active_disruptions "
                      "every single step and have pre-planned reroute pairs ready.")
        else:
            fb.append("[Hard] Low adversarial survival. Build a reactive decision table: "
                      "map each supplier_id to its backup and trigger reroute immediately "
                      "when that supplier appears in active_disruptions.")

        # Long-horizon inventory
        if lhi >= 0.80:
            fb.append("[Hard] Excellent long-horizon inventory management — buffer "
                      "stayed healthy across the full 90-day episode.")
        else:
            fb.append(f"[Hard] Insufficient long-horizon buffer (avg coverage = "
                      f"{lhi*10:.1f} days, target ≥10). "
                      "Use adjust_safety_stock proactively every 10–15 days, "
                      "not just after disruptions fire.")

        return fb
