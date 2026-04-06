"""
Graders package for the Supply Chain Disruption Navigator.

Convenience usage:
    from graders import grade_all, EasyGrader, MediumGrader, HardGrader

    def my_agent(obs): ...

    results = grade_all(my_agent, seed=42)
    for task_id, result in results.items():
        print(result.summary())
"""
from .base_grader import BaseGrader, GradeResult, AgentFn
from .grader_easy import EasyGrader
from .grader_medium import MediumGrader
from .grader_hard import HardGrader


def grade_all(agent_fn: AgentFn, seed: int = 42) -> dict:
    """Run all three graders and return {task_id: GradeResult}."""
    results = {}
    for GraderCls in [EasyGrader, MediumGrader, HardGrader]:
        grader = GraderCls()
        result = grader.grade(agent_fn, seed=seed)
        results[result.task_id] = result
    return results


__all__ = [
    "BaseGrader", "GradeResult", "AgentFn",
    "EasyGrader", "MediumGrader", "HardGrader",
    "grade_all",
]
