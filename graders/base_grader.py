"""
BaseGrader — shared scoring infrastructure for all task graders.

Each grader runs a complete episode with the agent under test, collects
per-step observations, then scores across six dimensions:

  1. production_continuity  — did the OEM factory keep running?
  2. crisis_response_speed  — how quickly did agent react to disruption?
  3. cost_discipline        — did spend stay within tolerance?
  4. esg_management         — were low-ESG suppliers deprioritised?
  5. resilience             — did agent proactively build buffers?
  6. recovery_completeness  — did production fully recover post-disruption?

Final score = weighted sum, normalised to [0.0, 1.0].

Usage:
    grader = EasyGrader()
    result = grader.grade(agent_fn, seed=42)
    print(result.score)          # 0.0 – 1.0
    print(result.breakdown)      # per-dimension dict
    print(result.feedback)       # human-readable commentary
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import statistics

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from supply_chain_env import (
    SupplyChainEnv, MultiAction, ResetRequest,
)
from supply_chain_env.models import Observation
from supply_chain_env.reward import episode_score


# ---------------------------------------------------------------------------
# Grade result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    task_id: str
    seed: int
    score: float                        # final 0.0–1.0
    breakdown: Dict[str, float]         # per-dimension scores
    feedback: List[str]                 # human-readable commentary
    step_rewards: List[float]
    episode_length: int
    steps_completed: int
    observations: List[Dict] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        lines = [
            f"Task: {self.task_id}  |  Score: {self.score:.4f}  |  "
            f"Steps: {self.steps_completed}/{self.episode_length}",
            "",
            "Dimension breakdown:",
        ]
        for dim, val in self.breakdown.items():
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            lines.append(f"  {dim:<28s} {bar}  {val:.3f}")
        lines.append("")
        lines.append("Feedback:")
        for fb in self.feedback:
            lines.append(f"  • {fb}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AgentFn type alias
# ---------------------------------------------------------------------------
# Any callable: (Observation) -> MultiAction

AgentFn = Callable[[Observation], MultiAction]


# ---------------------------------------------------------------------------
# BaseGrader
# ---------------------------------------------------------------------------

class BaseGrader(ABC):

    task_id: str = "base"
    episode_length: int = 30

    # Override per task
    dimension_weights: Dict[str, float] = dict(
        production_continuity = 0.30,
        crisis_response_speed = 0.20,
        cost_discipline       = 0.20,
        esg_management        = 0.10,
        resilience            = 0.10,
        recovery_completeness = 0.10,
    )

    # Thresholds
    RESPONSE_WINDOW_DAYS: int = 3    # agent should react within N days of disruption
    RECOVERY_THRESHOLD: float = 0.85 # fill rate considered "recovered"
    COST_TOLERANCE: float = 0.30

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def grade(self, agent_fn: AgentFn, seed: int = 42) -> GradeResult:
        """
        Run a full episode and score the agent.
        agent_fn: callable(Observation) -> MultiAction
        """
        env = SupplyChainEnv()
        resp = env.reset(ResetRequest(task_id=self.task_id, seed=seed))
        obs = resp.observation

        step_rewards: List[float] = []
        obs_log: List[Observation] = [obs]
        action_log: List[MultiAction] = []

        for _ in range(resp.episode_length):
            action = agent_fn(obs)
            action_log.append(action)
            obs = env.step(action)
            obs_log.append(obs)
            if obs.reward_breakdown:
                step_rewards.append(obs.reward_breakdown.total)
            if obs.is_done or obs.is_truncated:
                break

        breakdown = self._compute_breakdown(obs_log, action_log, step_rewards)
        score = self._weighted_score(breakdown)
        feedback = self._generate_feedback(obs_log, action_log, breakdown)

        return GradeResult(
            task_id=self.task_id,
            seed=seed,
            score=round(score, 4),
            breakdown={k: round(v, 4) for k, v in breakdown.items()},
            feedback=feedback,
            step_rewards=step_rewards,
            episode_length=resp.episode_length,
            steps_completed=len(step_rewards),
        )

    # -----------------------------------------------------------------------
    # Dimension scorers — shared implementations, override where needed
    # -----------------------------------------------------------------------

    def _score_production_continuity(
        self,
        obs_log: List[Observation],
    ) -> float:
        """Mean fill rate across all non-zero steps, with stoppage penalty."""
        fill_rates = [o.fill_rate for o in obs_log if o.timestep > 0]
        if not fill_rates:
            return 0.0
        mean_fill = statistics.mean(fill_rates)
        n_stopped = sum(1 for f in fill_rates if f < 0.05)
        stoppage_penalty = min(0.50, n_stopped * 0.05)
        return max(0.0, mean_fill - stoppage_penalty)

    def _score_crisis_response_speed(
        self,
        obs_log: List[Observation],
        action_log: List[MultiAction],
    ) -> float:
        """
        Detects when first disruption fires, then checks whether agent took
        a non-trivial action within RESPONSE_WINDOW_DAYS.
        Score: 1.0 if responded in 1 day, decays to 0.0 at RESPONSE_WINDOW+3.
        """
        disruption_day = None
        for obs in obs_log:
            if obs.active_disruptions and disruption_day is None:
                disruption_day = obs.timestep

        if disruption_day is None:
            return 1.0  # no disruption → full score (nothing to respond to)

        for i, (obs, act) in enumerate(zip(obs_log, action_log)):
            if obs.timestep < disruption_day:
                continue
            has_real_action = any(
                a.action_type.value != "do_nothing"
                for a in act.actions
            )
            if has_real_action:
                delay = obs.timestep - disruption_day
                # Score: 1.0 at delay=0, linear decay, 0.0 at delay=6
                return max(0.0, 1.0 - delay / (self.RESPONSE_WINDOW_DAYS + 3))

        return 0.0  # never responded

    def _score_cost_discipline(
        self,
        obs_log: List[Observation],
    ) -> float:
        """Fraction of steps where cost_vs_baseline <= 1 + COST_TOLERANCE."""
        steps = [o for o in obs_log if o.timestep > 0]
        if not steps:
            return 1.0
        within_budget = sum(
            1 for o in steps
            if o.cost_vs_baseline <= 1.0 + self.COST_TOLERANCE
        )
        return within_budget / len(steps)

    def _score_esg_management(
        self,
        obs_log: List[Observation],
    ) -> float:
        """Mean weighted ESG score across episode."""
        scores = [o.esg_score_weighted for o in obs_log if o.timestep > 0]
        return statistics.mean(scores) if scores else 0.0

    def _score_resilience(
        self,
        obs_log: List[Observation],
        action_log: List[MultiAction],
    ) -> float:
        """
        Agent score for proactive safety-stock building.
        Looks for ADJUST_SAFETY_STOCK actions before the first disruption.
        Also rewards maintaining >7 days of inventory coverage.
        """
        disruption_day = None
        for obs in obs_log:
            if obs.active_disruptions:
                disruption_day = obs.timestep
                break

        proactive_actions = 0
        for obs, act in zip(obs_log, action_log):
            if disruption_day and obs.timestep >= disruption_day:
                break
            for a in act.actions:
                if a.action_type.value == "adjust_safety_stock":
                    proactive_actions += 1

        proactive_score = min(1.0, proactive_actions / 2)

        # Inventory coverage bonus
        coverage = [o.inventory_coverage_days for o in obs_log if o.timestep > 0]
        mean_coverage = statistics.mean(coverage) if coverage else 0.0
        coverage_score = min(1.0, mean_coverage / 14.0)

        return 0.5 * proactive_score + 0.5 * coverage_score

    def _score_recovery_completeness(
        self,
        obs_log: List[Observation],
    ) -> float:
        """
        After the last disruption ends, did fill rate return to >= RECOVERY_THRESHOLD?
        Score: 1.0 if fully recovered, partial if partially recovered.
        """
        last_disruption_end = None
        for obs in reversed(obs_log):
            if obs.active_disruptions:
                last_disruption_end = obs.timestep
                break

        if last_disruption_end is None:
            return 1.0  # no disruption

        post_disruption = [
            o.fill_rate for o in obs_log
            if o.timestep > last_disruption_end
        ]
        if not post_disruption:
            return 0.5  # disruption at very end, can't assess

        max_recovery = max(post_disruption)
        if max_recovery >= self.RECOVERY_THRESHOLD:
            return 1.0
        return max_recovery / self.RECOVERY_THRESHOLD

    # -----------------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------------

    def _compute_breakdown(
        self,
        obs_log: List[Observation],
        action_log: List[MultiAction],
        step_rewards: List[float],
    ) -> Dict[str, float]:
        return dict(
            production_continuity = self._score_production_continuity(obs_log),
            crisis_response_speed = self._score_crisis_response_speed(obs_log, action_log),
            cost_discipline       = self._score_cost_discipline(obs_log),
            esg_management        = self._score_esg_management(obs_log),
            resilience            = self._score_resilience(obs_log, action_log),
            recovery_completeness = self._score_recovery_completeness(obs_log),
        )

    def _weighted_score(self, breakdown: Dict[str, float]) -> float:
        total = sum(
            breakdown.get(dim, 0.0) * w
            for dim, w in self.dimension_weights.items()
        )
        return min(1.0, max(0.0, total))

    # -----------------------------------------------------------------------
    # Feedback generator
    # -----------------------------------------------------------------------

    def _generate_feedback(
        self,
        obs_log: List[Observation],
        action_log: List[MultiAction],
        breakdown: Dict[str, float],
    ) -> List[str]:
        fb = []
        pc = breakdown["production_continuity"]
        cr = breakdown["crisis_response_speed"]
        cd = breakdown["cost_discipline"]
        esg = breakdown["esg_management"]
        res = breakdown["resilience"]
        rec = breakdown["recovery_completeness"]

        # Production continuity
        if pc >= 0.90:
            fb.append(f"Excellent production continuity ({pc:.0%}) — OEM factory ran near capacity.")
        elif pc >= 0.70:
            fb.append(f"Good production continuity ({pc:.0%}) — minor stoppages occurred.")
        else:
            fb.append(f"Poor production continuity ({pc:.0%}) — significant line stoppages. "
                      f"Consider earlier rerouting or safety stock increases.")

        # Crisis response
        if cr >= 0.80:
            fb.append("Fast crisis response — agent reacted within 1–2 days of disruption.")
        elif cr >= 0.50:
            fb.append("Moderate crisis response — some lag after disruption detected.")
        else:
            fb.append("Slow crisis response — agent delayed action after disruption. "
                      "Monitor active_disruptions each step and act immediately.")

        # Cost
        if cd >= 0.90:
            fb.append(f"Strong cost discipline ({cd:.0%} of steps within budget).")
        elif cd >= 0.70:
            fb.append(f"Acceptable cost control ({cd:.0%} of steps within budget).")
        else:
            fb.append(f"Cost overruns detected ({cd:.0%} within budget). "
                      "Avoid over-expediting — use negotiate_lead_time instead.")

        # ESG
        if esg >= 0.70:
            fb.append(f"Good ESG management (avg {esg:.2f}). Supplier mix is responsible.")
        elif esg >= 0.55:
            fb.append(f"Moderate ESG score ({esg:.2f}). Review low-ESG supplier reliance.")
        else:
            fb.append(f"Low ESG score ({esg:.2f}). Rerouting to high-ESG alternatives "
                      "like S302 (Germany, 0.80) would improve this dimension.")

        # Resilience
        if res >= 0.70:
            fb.append("Good resilience — proactive safety stock building observed.")
        else:
            fb.append("Low resilience score. Consider calling adjust_safety_stock on T1 "
                      "assemblers (S101, S102) before disruptions materialise.")

        # Recovery
        if rec >= 0.90:
            fb.append("Full recovery post-disruption — production returned to normal.")
        elif rec >= 0.60:
            fb.append("Partial recovery — production improved but didn't fully normalise.")
        else:
            fb.append("Incomplete recovery. After disruption ends, reroute back to primary "
                      "suppliers and build inventory to restore throughput.")

        return fb

    # -----------------------------------------------------------------------
    # Abstract: task-specific overrides
    # -----------------------------------------------------------------------

    @abstractmethod
    def _task_specific_checks(
        self,
        obs_log: List[Observation],
        action_log: List[MultiAction],
    ) -> Dict[str, float]:
        """Return any extra dimension scores specific to this task."""
        ...
