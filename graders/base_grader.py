from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any

# AgentFn: any callable that takes an Observation and returns a MultiAction
AgentFn = Callable[[Any], Any]


@dataclass
class GradeResult:
    task_id: str
    score: float
    passed: bool
    details: str = ""

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"Task: {self.task_id}  |  Score: {self.score:.4f}  |  {status}\n"
            f"{self.details}"
        )


class BaseGrader:
    task_id: str = "base"

    def grade(self, agent_fn: AgentFn = None, seed: int = 42) -> "GradeResult":
        from supply_chain_env import SupplyChainEnv, ResetRequest, Action, ActionType, MultiAction
        from supply_chain_env.reward import episode_score

        score = 0.0
        detail = ""

        try:
            env = SupplyChainEnv()
            resp = env.reset(ResetRequest(task_id=self.task_id, seed=seed))
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
                reward = obs.reward_breakdown.total if obs.reward_breakdown else 0.0
                rewards.append(reward)

            score = obs.info.get("episode_score", episode_score(rewards, resp.episode_length))
            score = float(min(max(score, 0.0), 1.0))
            detail = f"Completed {len(rewards)} steps. Episode score: {score:.4f}"

        except Exception as exc:
            detail = f"Error during grading: {exc}"
            score = 0.0

        return GradeResult(
            task_id=self.task_id,
            score=score,
            passed=score > 0.5,
            details=detail,
        )