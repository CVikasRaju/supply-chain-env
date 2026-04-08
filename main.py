import os, yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from typing import Optional

from supply_chain_env import SupplyChainEnv, ResetRequest, MultiAction

app = FastAPI(
    title="Supply Chain Disruption Navigator",
    description="OpenEnv-compatible multi-tier supply chain crisis simulation",
    version="1.0.0",
)

_env = SupplyChainEnv()

@app.get("/health")
def health():
    return {"status": "ok", "env": "supply_chain_disruption_navigator", "version": "1.0.0"}

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
        return {
            "obs": obs.dict(),
            "reward": reward,
            "done": done,
            "info": obs.info
        }
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
