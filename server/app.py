"""
FastAPI HTTP wrapper — exposes the OpenEnv step/reset/state API.

Endpoints:
  POST /reset        ResetRequest  → ResetResponse
  POST /step         MultiAction   → Observation
  GET  /state                      → StateResponse
  GET  /health                     → {"status": "ok"}
  GET  /openenv.yaml               → spec manifest
"""
from __future__ import annotations
import os, yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from supply_chain_env import SupplyChainEnv, ResetRequest, MultiAction

app = FastAPI(
    title="Supply Chain Disruption Navigator",
    description="OpenEnv-compatible multi-tier supply chain crisis simulation",
    version="1.0.0",
)

# One global env instance per worker (stateful)
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
        return _env.step(action)
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
    spec_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
    if not os.path.exists(spec_path):
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    with open(spec_path) as f:
        return f.read()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
