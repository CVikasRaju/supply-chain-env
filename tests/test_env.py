"""
Tests for Supply Chain Disruption Navigator.
Run: python -m pytest tests/test_env.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from supply_chain_env import (
    SupplyChainEnv, Action, ActionType, MultiAction,
    ResetRequest, SupplierTier,
)
from supply_chain_env.reward import compute_reward, episode_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return SupplyChainEnv()


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_easy(self, env):
        resp = env.reset(ResetRequest(task_id="easy", seed=42))
        assert resp.episode_length == 30
        obs = resp.observation
        assert obs.timestep == 0
        assert len(obs.suppliers) == 12
        assert len(obs.production_lines) == 2
        assert obs.fill_rate == 1.0

    def test_reset_medium(self, env):
        resp = env.reset(ResetRequest(task_id="medium", seed=42))
        assert resp.episode_length == 60

    def test_reset_hard(self, env):
        resp = env.reset(ResetRequest(task_id="hard", seed=42))
        assert resp.episode_length == 90

    def test_invalid_task(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(ResetRequest(task_id="impossible"))

    def test_deterministic_seed(self, env):
        r1 = env.reset(ResetRequest(task_id="easy", seed=99))
        env2 = SupplyChainEnv()
        r2 = env2.reset(ResetRequest(task_id="easy", seed=99))
        assert r1.seed == r2.seed
        assert r1.observation.fill_rate == r2.observation.fill_rate

    def test_reset_clears_state(self, env):
        env.reset(ResetRequest(task_id="easy", seed=1))
        env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))
        env.reset(ResetRequest(task_id="easy", seed=2))
        s = env.state()
        assert s.observation.timestep == 0
        assert s.cumulative_reward == 0.0


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_increments_timestep(self, env):
        env.reset(ResetRequest(task_id="easy", seed=42))
        obs = env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))
        assert obs.timestep == 1

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))

    def test_step_after_done_raises(self, env):
        env.reset(ResetRequest(task_id="easy", seed=42))
        # Fast-forward to end
        for _ in range(30):
            obs = env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))
        assert obs.is_done
        with pytest.raises(RuntimeError):
            env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))

    def test_reward_in_range(self, env):
        env.reset(ResetRequest(task_id="easy", seed=42))
        for _ in range(5):
            obs = env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))
            if obs.reward_breakdown:
                assert 0.0 <= obs.reward_breakdown.total <= 1.0

    def test_episode_completes(self, env):
        env.reset(ResetRequest(task_id="easy", seed=42))
        done = False
        steps = 0
        while not done and steps < 50:
            obs = env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))
            done = obs.is_done or obs.is_truncated
            steps += 1
        assert done
        assert steps <= 31  # should finish exactly at 30

    def test_multi_action_bundling(self, env):
        env.reset(ResetRequest(task_id="medium", seed=42))
        actions = [
            Action(action_type=ActionType.ADJUST_SAFETY_STOCK,
                   supplier_id="S101", target_stock_days=10.0),
            Action(action_type=ActionType.ADJUST_SAFETY_STOCK,
                   supplier_id="S102", target_stock_days=10.0),
        ]
        obs = env.step(MultiAction(actions=actions))
        assert obs.timestep == 1
        # Safety stock increase should reflect in inventory days
        assert obs.inventory_coverage_days >= 0.0


# ---------------------------------------------------------------------------
# Action tests
# ---------------------------------------------------------------------------

class TestActions:
    def test_reroute_order(self, env):
        env.reset(ResetRequest(task_id="easy", seed=42))
        obs = env.step(MultiAction(actions=[
            Action(
                action_type=ActionType.REROUTE_ORDER,
                from_supplier_id="S201",
                to_supplier_id="S202",
                material_type="battery_cells",
                quantity=80.0,
            )
        ]))
        assert obs.timestep == 1

    def test_expedite_shipment_costs_more(self, env):
        env.reset(ResetRequest(task_id="easy", seed=42))
        # Step 1: do nothing
        obs1 = env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))
        base_cost = obs1.total_cost_today
        env.reset(ResetRequest(task_id="easy", seed=42))
        # Step 1: expedite
        obs2 = env.step(MultiAction(actions=[
            Action(action_type=ActionType.EXPEDITE_SHIPMENT,
                   material_type="battery_cells", expedite_factor=2.0)
        ]))
        assert obs2.total_cost_today >= base_cost

    def test_disruption_fires_on_day5_easy(self, env):
        env.reset(ResetRequest(task_id="easy", seed=42))
        # Step through to day 6 (disruption fires on day 5)
        obs = None
        for _ in range(6):
            obs = env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))
        assert len(obs.active_disruptions) >= 1
        disrupted_ids = {d.affected_supplier_id for d in obs.active_disruptions.values()}
        assert "S201" in disrupted_ids

    def test_fill_rate_drops_after_disruption(self, env):
        env.reset(ResetRequest(task_id="easy", seed=42))
        fill_rates = []
        for _ in range(25):
            obs = env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))
            fill_rates.append(obs.fill_rate)
        # At least one step should show reduced fill rate after disruption
        assert min(fill_rates) < 1.0


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

class TestReward:
    def test_reward_decomposition_sums(self):
        rb = compute_reward(
            fill_rates={"LINE_A": 1.0, "LINE_B": 1.0},
            daily_cost=10000,
            baseline_daily_cost=10000,
            esg_score=0.7,
            on_time_rate=0.95,
            inventory_days=8.0,
            previous_inventory_days=6.0,
            active_disruption_count=1,
        )
        assert rb.production_continuity == 1.0
        assert rb.cost_efficiency == 1.0
        assert 0.0 <= rb.total <= 1.0

    def test_zero_fill_rate_gives_low_reward(self):
        rb = compute_reward(
            fill_rates={"LINE_A": 0.0, "LINE_B": 0.0},
            daily_cost=20000,
            baseline_daily_cost=10000,
            esg_score=0.4,
            on_time_rate=0.3,
            inventory_days=0.0,
            previous_inventory_days=0.0,
            active_disruption_count=3,
        )
        assert rb.production_continuity < 0.05
        assert rb.total < 0.30

    def test_episode_score_penalises_early_termination(self):
        rewards = [0.8] * 10   # only 10 of 30 steps
        score = episode_score(rewards, episode_length=30)
        assert score < 0.8 * 10 / 30 + 0.01

    def test_full_episode_score(self):
        rewards = [0.8] * 30
        score = episode_score(rewards, episode_length=30)
        assert abs(score - 0.8) < 0.01

    def test_resilience_bonus_given_proactive_stocking(self):
        rb = compute_reward(
            fill_rates={"LINE_A": 0.9},
            daily_cost=10500,
            baseline_daily_cost=10000,
            esg_score=0.7,
            on_time_rate=0.9,
            inventory_days=10.0,        # healthy buffer
            previous_inventory_days=6.0, # was building up
            active_disruption_count=2,   # under active disruption
        )
        assert rb.resilience_bonus > 0.0


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestState:
    def test_state_before_reset_raises(self):
        env = SupplyChainEnv()
        with pytest.raises(RuntimeError):
            env.state()

    def test_state_reflects_cumulative_reward(self, env):
        env.reset(ResetRequest(task_id="easy", seed=42))
        for _ in range(5):
            env.step(MultiAction(actions=[Action(action_type=ActionType.DO_NOTHING)]))
        s = env.state()
        assert s.cumulative_reward > 0.0
        assert len(s.step_rewards) == 5
