from __future__ import annotations
from .base_grader import BaseGrader, GradeResult, AgentFn
from supply_chain_env.reward import episode_score


class HardGrader(BaseGrader):
    """
    Grader for the 'hard' task:
    Adversarial multi-tier disruptions. No forecast. ESG + cost. 90 days.
    Pass threshold: score >= 0.50
    """
    task_id: str = "hard"

    def grade(self, agent_fn: AgentFn = None, seed: int = 42, env=None) -> GradeResult:
        from supply_chain_env import SupplyChainEnv, ResetRequest

        try:
            _env = env if env is not None else SupplyChainEnv()
            resp = _env.reset(ResetRequest(task_id=self.task_id, seed=seed))
            obs = resp.observation
            rewards = []

            for _ in range(resp.episode_length):
                if obs.is_done or obs.is_truncated:
                    break
                action = agent_fn(obs) if agent_fn else None
                if action is None:
                    from supply_chain_env import MultiAction, Action, ActionType
                    action = MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)])
                obs = _env.step(action)
                reward = obs.reward_breakdown.total if obs.reward_breakdown else 0.0
                rewards.append(reward)

            score = episode_score(rewards, resp.episode_length)
            score = min(max(score, 0.0), 1.0)
            passed = score >= 0.50
            return GradeResult(
                task_id=self.task_id,
                score=score,
                passed=passed,
                details=f"Hard task: {len(rewards)} steps completed. Episode score: {score:.4f}"
            )
        except Exception as e:
            return GradeResult(
                task_id=self.task_id,
                score=0.0,
                passed=False,
                details=f"Error during grading: {e}"
            )