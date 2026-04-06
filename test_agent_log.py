import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supply_chain_env import SupplyChainEnv, ResetRequest
from baseline.run_baseline import heuristic_agent

env = SupplyChainEnv()
resp = env.reset(ResetRequest(task_id="easy", seed=42))
obs = resp.observation
for i in range(10):
    action = heuristic_agent(obs)
    print(f"Step {i}: actions={[a.action_type for a in action.actions]}, cost_ratio={obs.cost_vs_baseline:.2f}, fill={obs.fill_rate:.2f}")
    obs = env.step(action)
