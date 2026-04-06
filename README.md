---
title: Supply Chain Disruption Navigator
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Supply Chain Disruption Navigator

**An OpenEnv-compatible AI training environment for multi-tier supply chain crisis management.**

An AI agent controls a 12-supplier, 4-tier supply chain feeding one OEM factory. Stochastic disruptions — port strikes, weather events, geopolitical blocks, factory fires, quality failures — cascade across tiers. The agent must reroute orders, manage safety stock, expedite shipments, and negotiate lead times to keep production running while respecting cost and ESG constraints.

---

## Why this environment?

Supply chain resilience is a multi-billion dollar problem. Real companies (Toyota, Apple, TSMC) employ entire teams to do what this environment asks an agent to do — in real time, with incomplete information. No OpenEnv environment has existed for this domain until now.

This environment is research-grade: it has meaningful partial reward signals, realistic cascading disruption dynamics, three difficulty levels, and agent graders with per-dimension feedback.

---

## Environment overview

```
Tier 4 (Raw material)   S401 Mining Co Alpha · S402 Agri Corp Beta · S403 PetroChem Gamma
           ↓ material flow
Tier 3 (Processors)     S301 Smelter Delta · S302 Chem Plant Epsilon · S303 Resin Extruder Zeta
           ↓
Tier 2 (Components)     S201 Battery Mfg · S202 PCB Fab · S203 Housing Mold
           ↓
Tier 1 (Assemblers)     S101 Assembler Alpha · S102 Assembler Beta
           ↓
OEM Factory             OEM0 — daily production demand
```

Each tier feeds the next. Disruptions at higher tiers cascade downstream with a delay. The agent's goal: keep the OEM factory's fill rate high and costs low.

---

## Quick start

### Install

```bash
pip install pydantic fastapi uvicorn pyyaml pytest
```

### Run the environment

```python
from supply_chain_env import SupplyChainEnv, ResetRequest, MultiAction, Action, ActionType

env = SupplyChainEnv()
resp = env.reset(ResetRequest(task_id="easy", seed=42))
print(resp.task_description)

obs = resp.observation
for step in range(resp.episode_length):
    # Your agent logic here
    action = MultiAction(actions=[
        Action(action_type=ActionType.DO_NOTHING)
    ])
    obs = env.step(action)
    print(f"Day {obs.day}  fill_rate={obs.fill_rate:.2%}  "
          f"cost×baseline={obs.cost_vs_baseline:.2f}  "
          f"disruptions={len(obs.active_disruptions)}")
    if obs.is_done or obs.is_truncated:
        break

print(f"Episode score: {obs.info['episode_score']}")
```

### Run the API server

```bash
python api/app.py
# Server starts on http://0.0.0.0:7860
```

API endpoints:
- `POST /reset`   — start a new episode
- `POST /step`    — take an action
- `GET  /state`   — full internal state (for debugging)
- `GET  /health`  — health check

### Run the baseline

```bash
python baseline/run_baseline.py
```

Expected baseline (heuristic agent):
```
easy   → 0.42
medium → 0.51
hard   → 0.36
```

### Run tests

```bash
pytest tests/test_env.py -v
```

---

## Tasks

### Task 1 — Easy (30 days)

A factory fire hits Battery Mfg (S201, Tier 2) on day 5. Full disruption visibility.

The agent must detect the disruption, reroute battery cell supply, and recover production. Clear cause-and-effect. Good for learning basic reactive policies.

**Target score:** 0.72+ (heuristic), 0.90+ (optimal)

### Task 2 — Medium (60 days)

A geopolitical export ban blocks lithium ore from Chile (S401, Tier 4) on day 7. Five days later, this cascades to Smelter Delta (S301, Tier 3). Random background disruptions also fire. No forecast.

The agent must understand multi-tier dependencies, anticipate cascades, and manage 60 days without a budget overrun.

**Target score:** 0.58+ (heuristic), 0.80+ (optimal)

### Task 3 — Hard (90 days)

Two scripted disruptions on days 3 and 10. High-frequency adversarial disruptions clustered on Tier-1 and Tier-2 suppliers. Tightest cost tolerance (20%). Highest ESG weight (20%). No forecast. 90-day horizon.

The agent must maintain portfolio diversification, balance ESG and cost simultaneously, and survive persistent adversarial pressure.

**Target score:** 0.41+ (heuristic), 0.75+ (state-of-the-art target)

---

## Action space

Actions are bundled in a `MultiAction` (a list of `Action` objects submitted per step).

| Action type | Required fields | Effect |
|---|---|---|
| `do_nothing` | — | Pass |
| `reroute_order` | `from_supplier_id`, `to_supplier_id`, `material_type`, `quantity` | Redirect material flow to alternative supplier |
| `adjust_safety_stock` | `supplier_id`, `target_stock_days` | Buy inventory buffer (holding cost incurred) |
| `expedite_shipment` | `material_type`, `expedite_factor` | Speed up delivery (cost multiplier) |
| `accept_substitute` | `substitute_supplier_id`, `quality_penalty_pct` | Use alternative material (slight quality/throughput hit) |
| `negotiate_lead_time` | `supplier_id`, `target_lead_time_days`, `cost_premium_pct` | Pay premium for faster base lead time |

### Example action

```python
MultiAction(actions=[
    Action(
        action_type=ActionType.REROUTE_ORDER,
        from_supplier_id="S201",
        to_supplier_id="S202",
        material_type="battery_cells",
        quantity=80.0,
    ),
    Action(
        action_type=ActionType.ADJUST_SAFETY_STOCK,
        supplier_id="S101",
        target_stock_days=10.0,
    ),
])
```

---

## Observation space

The `Observation` object returned by `step()` contains:

| Field | Type | Description |
|---|---|---|
| `timestep` | int | Current day in episode |
| `suppliers` | Dict[str, SupplierState] | Full state of all 12 suppliers |
| `active_disruptions` | Dict[str, ActiveDisruption] | Currently active disruptions |
| `current_routes` | List[OrderRoute] | Active material flow edges |
| `production_lines` | List[ProductionLine] | OEM line throughput and demand |
| `fill_rate` | float [0,1] | Fraction of daily demand met |
| `inventory_coverage_days` | float | OEM buffer in days |
| `total_cost_today` | float | Today's total supply chain cost |
| `cost_vs_baseline` | float | Ratio to no-disruption baseline cost |
| `on_time_delivery_rate` | float [0,1] | Fraction of shipments on time |
| `esg_score_weighted` | float [0,1] | Weighted ESG of active suppliers |
| `disruption_forecast` | List | Upcoming disruptions (empty in hard mode) |
| `reward_breakdown` | RewardBreakdown | Per-dimension reward (partial signals) |

---

## Reward function

```
reward = (production_continuity × cost_efficiency × esg_compliance)
       + resilience_bonus             (up to +0.20, capped at 1.0 total)
```

| Dimension | Easy | Medium | Hard | Description |
|---|---|---|---|---|
| production_continuity | 0.60 | 0.50 | 0.40 | Fraction of demand met; penalises line stoppages |
| cost_efficiency | 0.20 | 0.25 | 0.25 | 1 − cost_overrun vs baseline |
| esg_compliance | 0.10 | 0.15 | 0.20 | Weighted avg ESG score of active suppliers |
| lead_time_adherence | 0.10 | 0.10 | 0.15 | Fraction of shipments on time |
| resilience_bonus | up to +0.20 | up to +0.20 | up to +0.20 | Proactive safety stock building under active disruptions |

All components return partial credit — the agent receives a meaningful signal at every step, not just at episode end.

---

## Graders

The grader runs a full episode and scores across six dimensions with textual feedback:

```python
from graders import EasyGrader, MediumGrader, HardGrader, grade_all

def my_agent(obs):
    # ... your agent logic
    return MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)])

result = EasyGrader().grade(my_agent, seed=42)
print(result.summary())
```

Output:
```
Task: easy  |  Score: 0.7234  |  Steps: 30/30

Dimension breakdown:
  production_continuity        ████████████████░░░░  0.812
  crisis_response_speed        ██████████████░░░░░░  0.700
  cost_discipline              ████████████████████  1.000
  esg_management               █████████████░░░░░░░  0.651
  resilience                   ████████░░░░░░░░░░░░  0.400
  recovery_completeness        ████████████████░░░░  0.800

Feedback:
  • Good production continuity (81%) — minor stoppages occurred.
  • Fast crisis response — agent reacted within 1–2 days of disruption.
  • [Task-specific] Missed: did not reroute away from S201 after factory fire.
```

---

## File structure

```
supply-chain-env/
├── openenv.yaml                    OpenEnv spec manifest
├── Dockerfile                      HuggingFace Spaces deployment
├── README.md
│
├── supply_chain_env/
│   ├── __init__.py
│   ├── env.py                      SupplyChainEnv — step/reset/state
│   ├── models.py                   Pydantic typed models
│   ├── network.py                  Supplier graph + material flow
│   ├── disruptions.py              Disruption engine + cascades
│   ├── reward.py                   Reward decomposition
│   └── tasks/
│       ├── task_easy.py            Easy scenario + observation guide
│       ├── task_medium.py          Medium scenario + cascade map
│       └── task_hard.py            Hard scenario + ESG decision matrix
│
├── graders/
│   ├── base_grader.py              Shared scoring + feedback logic
│   ├── grader_easy.py              Easy-specific checks (S201 avoidance)
│   ├── grader_medium.py            Medium-specific checks (cascade handling)
│   └── grader_hard.py              Hard-specific checks (diversification, ESG-cost)
│
├── baseline/
│   └── run_baseline.py             Heuristic agent, reproducible scores
│
├── api/
│   └── app.py                      FastAPI HTTP wrapper
│
└── tests/
    └── test_env.py                 20 tests across all components
```

---

## Deploy to HuggingFace Spaces

1. Create a new Space (type: Docker)
2. Push this repo
3. The `Dockerfile` exposes port 7860 which Spaces maps automatically

```bash
git init
git remote add origin https://huggingface.co/spaces/YOUR_NAME/supply-chain-env
git add .
git commit -m "initial"
git push origin main
```

The API will be live at `https://YOUR_NAME-supply-chain-env.hf.space`.

---

## Extending the environment

**Add a new disruption type:** Add an entry to `DISRUPTION_TEMPLATES` in `disruptions.py`.

**Add a new task:** Add a config entry to `TASK_CONFIGS` in `env.py` and a corresponding `task_*.py` file in `supply_chain_env/tasks/`.

**Add a new supplier:** Add to `DEFAULT_SUPPLIERS` in `network.py` and wire it into `DEFAULT_EDGES`.

**Add a new action:** Add to the `ActionType` enum in `models.py`, implement the handler in `network.py`, and dispatch it in `env.py`'s `_apply_action()`.

---

## Baseline scores (seed=42, heuristic agent)

| Task | Episode score | Mean reward | Fill rate | Cost vs baseline |
|---|---|---|---|---|
| easy   | 0.4245 | 0.4245 | 1.00 | 1.13× |
| medium | 0.5190 | 0.5190 | 1.00 | 1.01× |
| hard   | 0.3682 | 0.3682 | 1.00 | 0.81× |

---

## Citation

```bibtex
@software{supply_chain_disruption_navigator,
  title  = {Supply Chain Disruption Navigator: An OpenEnv Environment},
  year   = {2025},
  url    = {https://huggingface.co/spaces/YOUR_NAME/supply-chain-env}
}
```

---

## License

MIT
