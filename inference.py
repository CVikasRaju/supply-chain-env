import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from supply_chain_env import SupplyChainEnv, ResetRequest, MultiAction, Action, ActionType
from supply_chain_env.reward import episode_score

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = os.getenv("BENCHMARK", "supply-chain-env")
TEMPERATURE = 0.2
MAX_TOKENS = 512

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI managing a multi-tier supply chain disruption.
    Your goal is to maximize the total reward by maintaining high fill rates, low costs, and high ESG scores.
    Each step you will receive an observation containing active disruptions, inventory, and supplier states.
    You must output a single well-formatted JSON object representing your action.
    The JSON must follow this exact schema:
    {
      "actions": [
        {
          "action_type": "do_nothing" | "reroute_order" | "adjust_safety_stock" | "expedite_shipment" | "accept_substitute" | "negotiate_lead_time",
          "from_supplier_id": "S...",
          "to_supplier_id": "S...",
          "material_type": "...",
          "quantity": 10.0,
          "supplier_id": "S...",
          "target_stock_days": 10.0,
          "expedite_factor": 2.0,
          "substitute_supplier_id": "S...",
          "quality_penalty_pct": 0.05,
          "target_lead_time_days": 5,
          "cost_premium_pct": 10.0
        }
      ]
    }
    Reply with ONLY the raw JSON object. Do not include markdown blocks like ```json.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs_json: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation (JSON): {obs_json}
        Last reward: {last_reward:.2f}
        Previous actions/rewards:
        {history_block}
        Send your next action as JSON.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, obs_json: str, last_reward: float, history: List[str]) -> tuple[MultiAction, str, Optional[str]]:
    user_prompt = build_user_prompt(step, obs_json, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Clean text if wrapped in markdown
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        text = text.strip()
        
        parsed = json.loads(text)
        action_obj = MultiAction(**parsed)
        return action_obj, text, None
    except Exception as exc:
        print(f"[DEBUG] Model mapping failed: {exc}", flush=True)
        fallback = MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)])
        return fallback, '{"actions":[{"action_type":"do_nothing"}]}', str(exc)

async def main() -> None:
    # Initialize the synchronous OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Instantiate the synchronous environment directly 
    env = SupplyChainEnv()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(ResetRequest(task_id=TASK_NAME, seed=42))
        obs = result.observation
        last_reward = 0.0

        for step in range(1, result.episode_length + 1):
            if obs.is_done or obs.is_truncated:
                break

            # The full observation dict dumped to json string
            obs_json = obs.model_dump_json()

            multi_action, raw_action_str, error_msg = get_model_action(client, step, obs_json, last_reward, history)

            obs = env.step(multi_action)

            reward = obs.reward_breakdown.total if obs.reward_breakdown else 0.0
            done = obs.is_done or obs.is_truncated

            # Formatting action text cleanly for stdout on a single line
            action_log_str = raw_action_str.replace('\n', '').replace('\r', '').replace(' ', '')
            
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_log_str, reward=reward, done=done, error=error_msg)

            history.append(f"Step {step}: {action_log_str} -> reward {reward:+.2f}")

            if done:
                break

        score = episode_score(rewards, result.episode_length)
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= 0.50 # Assuming normalized threshold out of 1.0

    except Exception as e:
        print(f"[DEBUG] env error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
