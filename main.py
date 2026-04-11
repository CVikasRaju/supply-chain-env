"""
Main entry point — identical to server/app.py but used by CMD ["python", "main.py"].
All openenv-http/1.x required endpoints are included here.
"""
from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from typing import Optional

from supply_chain_env import SupplyChainEnv, ResetRequest, MultiAction

app = FastAPI(
    title="Supply Chain Disruption Navigator",
    description="OpenEnv-compatible multi-tier supply chain crisis simulation",
    version="1.0.0",
)

_env = SupplyChainEnv()


# ── Required by openenv validate: status MUST be "healthy" ──────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "env": "supply_chain_disruption_navigator", "version": "1.0.0"}


# ── Required by openenv validate ────────────────────────────────────────────
@app.get("/metadata")
def metadata():
    return {
        "name": "supply_chain_disruption_navigator",
        "description": (
            "A multi-tier supplier network crisis management environment. "
            "An AI agent controls a 12-supplier, 4-tier supply chain under "
            "stochastic disruptions (port strikes, weather, geopolitical blocks, "
            "factory fires, quality failures)."
        ),
        "version": "1.0.0",
        "author": "cvikasraju",
        "tags": ["supply-chain", "logistics", "operations", "crisis-management", "real-world"],
    }


# ── Required by openenv validate: action, observation, state schemas ─────────
@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "description": "MultiAction containing a list of Action objects",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_type": {
                                "type": "string",
                                "enum": [
                                    "reroute_order", "adjust_safety_stock",
                                    "expedite_shipment", "accept_substitute",
                                    "negotiate_lead_time", "do_nothing",
                                ],
                            },
                            "supplier_id": {"type": "string"},
                            "target_supplier_id": {"type": "string"},
                            "quantity": {"type": "number"},
                            "cost_premium": {"type": "number"},
                        },
                        "required": ["action_type"],
                    },
                }
            },
            "required": ["actions"],
        },
        "observation": {
            "type": "object",
            "description": "Supply chain state observation",
            "properties": {
                "suppliers": {"type": "object"},
                "active_disruptions": {"type": "object"},
                "current_routes": {"type": "array"},
                "production_lines": {"type": "array"},
                "fill_rate": {"type": "number"},
                "inventory_coverage_days": {"type": "number"},
                "total_cost_today": {"type": "number"},
                "cost_vs_baseline": {"type": "number"},
                "on_time_delivery_rate": {"type": "number"},
                "esg_score_weighted": {"type": "number"},
                "disruption_forecast": {"type": "array"},
                "reward_breakdown": {"type": "object"},
                "is_done": {"type": "boolean"},
                "is_truncated": {"type": "boolean"},
                "step": {"type": "integer"},
                "info": {"type": "object"},
            },
        },
        "state": {
            "type": "object",
            "description": "Episode state metadata",
            "properties": {
                "episode_id": {"type": "string"},
                "task_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "episode_length": {"type": "integer"},
                "is_done": {"type": "boolean"},
            },
        },
    }


# ── Required by openenv validate: MCP JSON-RPC 2.0 ──────────────────────────
@app.post("/mcp")
async def mcp(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    method = body.get("method", "")
    req_id = body.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "supply_chain_disruption_navigator", "version": "1.0.0"},
            },
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": "reset", "description": "Reset the supply chain environment"},
                    {"name": "step", "description": "Take a step in the environment"},
                    {"name": "state", "description": "Get current episode state"},
                ]
            },
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


# ── Tasks with graders (module dot-path format) ───────────────────────────────
@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "id": "baseline",
                "name": "Baseline - Smooth Operations",
                "description": "Steady-state operations with no forced disruptions. 20 days.",
                "difficulty": "easy",
                "episode_length": 20,
                "reward_range": [0.0, 1.0],
                "grader": "graders.grader_baseline:grade",
            },
            {
                "id": "easy",

                "name": "Easy - Factory Fire",
                "description": "Single T2 supplier factory fire. Full visibility. 30 days.",
                "difficulty": "easy",
                "episode_length": 30,
                "reward_range": [0.0, 1.0],
                "grader": "graders.grader_easy:grade",
            },
            {
                "id": "medium",
                "name": "Medium - Geopolitical Block",
                "description": "Geopolitical block cascading T4→T3. Partial info. 60 days.",
                "difficulty": "medium",
                "episode_length": 60,
                "reward_range": [0.0, 1.0],
                "grader": "graders.grader_medium:grade",
            },
            {
                "id": "hard",
                "name": "Hard - Multi-Tier Adversarial",
                "description": "Adversarial multi-tier disruptions. No forecast. ESG + cost. 90 days.",
                "difficulty": "hard",
                "episode_length": 90,
                "reward_range": [0.0, 1.0],
                "grader": "graders.grader_hard:grade",
            },
        ]
    }


# ── Core simulation endpoints ─────────────────────────────────────────────────
@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    if request is None:
        request = ResetRequest(task_id="easy")
    try:
        return _env.reset(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: MultiAction):
    try:
        obs = _env.step(action)
        reward = obs.reward_breakdown.total if obs.reward_breakdown else 0.0
        done = obs.is_done or obs.is_truncated
        return {"obs": obs.dict(), "reward": reward, "done": done, "info": obs.info}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/state")
def state():
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def get_spec():
    spec_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    if not os.path.exists(spec_path):
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    with open(spec_path) as f:
        return f.read()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
