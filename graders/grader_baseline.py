"""
Grader for the 'baseline' task:
  Standard operations with no forced disruptions. 20 days.
  Pass threshold: score >= 0.70
"""
from __future__ import annotations
from typing import Callable, Any, Optional

AgentFn = Callable[[Any], Any]


def grade(agent_fn: Optional[AgentFn] = None, seed: int = 42, **kwargs) -> dict:
    """
    Top-level grade function for the 'baseline' task.
    Called by the OpenEnv validator via: graders.grader_baseline:grade
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from supply_chain_env import SupplyChainEnv, ResetRequest, MultiAction, Action, ActionType
        from supply_chain_env.reward import episode_score
    except ImportError as e:
        return {"task_id": "baseline", "score": 0.0, "passed": False, "details": f"Import error: {e}"}

    task_id = "baseline"
    try:
        env = SupplyChainEnv()
        resp = env.reset(ResetRequest(task_id=task_id, seed=seed))
        obs = resp.observation
        rewards = []

        for _ in range(resp.episode_length):
            if obs.is_done or obs.is_truncated:
                break
            if agent_fn is not None:
                action = agent_fn(obs)
            else:
                action = MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)])
            obs = env.step(action)
            rewards.append(obs.reward_breakdown.total if obs.reward_breakdown else 0.0)

        score = float(min(max(episode_score(rewards, resp.episode_length), 0.0), 1.0))
        passed = score >= 0.70
        return {
            "task_id": task_id,
            "score": score,
            "passed": passed,
            "details": f"Baseline task: {len(rewards)} steps. Score: {score:.4f}",
        }
    except Exception as e:
        return {"task_id": task_id, "score": 0.0, "passed": False, "details": f"Error: {e}"}


class BaselineGrader:
    task_id = "baseline"

    def grade(self, agent_fn: Optional[AgentFn] = None, seed: int = 42, env=None, **kwargs) -> dict:
        return grade(agent_fn=agent_fn, seed=seed)
