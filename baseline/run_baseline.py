"""
Baseline inference script.
Runs a simple heuristic agent on all 3 tasks and prints reproducible scores.

Heuristic strategy:
  - If a disruption hits a T2/T1 supplier → reroute to lowest-cost available alt
  - If inventory_coverage_days < 5 → adjust safety stock to 10 days
  - If on_time_rate < 0.7 → expedite critical materials
  - Otherwise → do_nothing

Run:
    python baseline/run_baseline.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from supply_chain_env import (
    SupplyChainEnv, Action, ActionType, MultiAction,
    ResetRequest, SupplierTier,
)
from supply_chain_env.reward import episode_score


# ---------------------------------------------------------------------------
# Heuristic agent
# ---------------------------------------------------------------------------

REROUTE_ALTERNATIVES = {
    # If S201 (Battery Mfg) is disrupted → try to pull from S101 inventory buffer
    "S201": ("S202", "battery_cells"),
    # If S202 (PCB Fab) is disrupted → accept substitute from S303
    "S202": ("S303", "circuit_boards"),
    # If S301 (Smelter) is disrupted → reduce T3 demand via stock build at T2
    "S301": ("S302", "refined_lithium"),
    # If S401 is disrupted → reroute ore from S402
    "S401": ("S402", "lithium_ore"),
}

CRITICAL_MATERIALS = ["battery_cells", "circuit_boards", "sub_assembly_A", "sub_assembly_B"]


def heuristic_agent(obs) -> MultiAction:
    actions = []
    disrupted_ids = {d.affected_supplier_id for d in obs.active_disruptions.values()}
    existing_dests = {r.to_supplier_id for r in obs.current_routes if r.quantity > 0}
    
    # We use a state tracker attached to the function to remember one-off actions
    if not hasattr(heuristic_agent, "state"):
        heuristic_agent.state = {
            "rerouted": set(),
            "substituted": set(),
            "expedited": set()
        }
    state = heuristic_agent.state

    # Rule 1: reroute if key suppliers are disrupted
    for sid in disrupted_ids:
        if sid in REROUTE_ALTERNATIVES:
            alt_id, mat = REROUTE_ALTERNATIVES[sid]
            if sid not in state["rerouted"]:
                actions.append(Action(
                    action_type=ActionType.REROUTE_ORDER,
                    from_supplier_id=sid,
                    to_supplier_id=alt_id,
                    material_type=mat,
                    quantity=80.0,
                ))
                state["rerouted"].add(sid)

    # Rule 2: build safety stock if buffer is low
    if obs.inventory_coverage_days < 5.0 and not disrupted_ids:
        t1_inv = sum(s.inventory_held for s in obs.suppliers.values() if s.tier.value == 1)
        if t1_inv < 300:  # simplistic check to avoid infinite buying
            for sid in ["S101", "S102"]:
                actions.append(Action(
                    action_type=ActionType.ADJUST_SAFETY_STOCK,
                    supplier_id=sid,
                    target_stock_days=10.0,
                ))

    # Rule 3: expedite critical materials if delivery is late
    if obs.on_time_delivery_rate < 0.70:
        for mat in CRITICAL_MATERIALS[:2]:
            if mat not in state["expedited"]:
                actions.append(Action(
                    action_type=ActionType.EXPEDITE_SHIPMENT,
                    material_type=mat,
                    expedite_factor=2.0,
                ))
                state["expedited"].add(mat)

    # Rule 4: accept substitute if T2 is severely disrupted
    for did, d in obs.active_disruptions.items():
        if d.capacity_impact >= 0.80 and d.affected_supplier_id.startswith("S2"):
            if d.affected_supplier_id not in state["substituted"]:
                actions.append(Action(
                    action_type=ActionType.ACCEPT_SUBSTITUTE,
                    substitute_supplier_id=d.affected_supplier_id,
                    quality_penalty_pct=0.05,
                ))
                state["substituted"].add(d.affected_supplier_id)

    if not actions:
        actions.append(Action(action_type=ActionType.DO_NOTHING))

    return MultiAction(actions=actions, timestep=obs.timestep)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run_task(task_id: str, seed: int = 42) -> dict:
    env = SupplyChainEnv()
    resp = env.reset(ResetRequest(task_id=task_id, seed=seed))
    obs = resp.observation

    step_rewards = []
    for step in range(resp.episode_length):
        action = heuristic_agent(obs)
        obs = env.step(action)
        if obs.reward_breakdown:
            step_rewards.append(obs.reward_breakdown.total)
        if obs.is_done or obs.is_truncated:
            break

    final_state = env.state()
    score = episode_score(step_rewards, resp.episode_length)

    return dict(
        task_id=task_id,
        seed=seed,
        steps_completed=len(step_rewards),
        episode_length=resp.episode_length,
        episode_score=score,
        mean_reward=round(sum(step_rewards) / max(1, len(step_rewards)), 4),
        cumulative_reward=round(sum(step_rewards), 4),
        final_fill_rate=round(obs.fill_rate, 4),
        final_cost_vs_baseline=round(obs.cost_vs_baseline, 4),
        final_esg_score=round(obs.esg_score_weighted, 4),
        truncated=obs.is_truncated,
    )


def main():
    from typing import Dict
    print("\n" + "="*60)
    print("  Supply Chain Disruption Navigator — Baseline Scores")
    print("="*60)

    results = {}
    for task_id in ["easy", "medium", "hard"]:
        print(f"\nRunning task: {task_id} (seed=42)...")
        r = run_task(task_id, seed=42)
        results[task_id] = r
        print(f"  Episode score    : {r['episode_score']:.4f}")
        print(f"  Mean step reward : {r['mean_reward']:.4f}")
        print(f"  Steps completed  : {r['steps_completed']}/{r['episode_length']}")
        print(f"  Fill rate (final): {r['final_fill_rate']:.2%}")
        print(f"  Cost vs baseline : {r['final_cost_vs_baseline']:.2f}x")
        print(f"  ESG score        : {r['final_esg_score']:.2%}")
        print(f"  Truncated early  : {r['truncated']}")

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    for task_id, r in results.items():
        print(f"  {task_id:8s} -> {r['episode_score']:.4f}")
    print()

    # Save results
    import json
    out_path = os.path.join(os.path.dirname(__file__), "results", "baseline_scores.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    from typing import Dict
    main()
