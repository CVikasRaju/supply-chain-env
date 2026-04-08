from supply_chain_env.reward import episode_score

class BaseGrader:
    @classmethod
    def grade(cls, env=None, task_id=None, seed=42) -> float:
        if env is not None:
            if hasattr(env, "_step_rewards"):
                return episode_score(env._step_rewards, env._episode_length)
            # fallback if info has episode_score
            try:
                state = env.state()
                return state.observation.info.get("episode_score", 0.0)
            except Exception:
                pass
        return 0.0
