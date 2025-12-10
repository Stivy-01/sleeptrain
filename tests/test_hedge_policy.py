import json
from pathlib import Path

from scripts.rl.hedge_hippocampus import ACTIONS, HedgePolicy
from scripts.training.train_loop import HedgeConfig, HedgeTrainer


def test_hedge_policy_updates_and_saves(tmp_path: Path):
    policy = HedgePolicy(eta=0.5, reward_clip=(-1, 1))
    probs_before = policy.probs()
    assert len(probs_before) == len(ACTIONS)

    # Update with a positive reward on STORE
    store_idx = policy.action_index("STORE")
    policy.update(store_idx, reward=0.5)
    probs_after = policy.probs()
    assert probs_after[store_idx] > probs_before[store_idx]

    # Save/load round-trip
    save_path = tmp_path / "hedge.json"
    policy.save(str(save_path))
    loaded = HedgePolicy.load(str(save_path))
    assert loaded.probs() == policy.probs()


def test_hedge_trainer_step_and_routing_bias():
    calls = {"STORE": 0, "REJECT": 0, "CORRECT": 0}

    def fake_action_fn(action_name: str):
        calls[action_name] += 1
        # simulate metric delta: reward higher for STORE
        return {"delta_metric": 0.2 if action_name == "STORE" else 0.0}

    def reward_fn(metrics):
        return metrics["delta_metric"]

    cfg = HedgeConfig(routing_bias="store-heavy")
    trainer = HedgeTrainer(cfg, action_fn=fake_action_fn, reward_fn=reward_fn)

    log = trainer.step()
    assert "action" in log and "reward" in log
    assert calls["STORE"] >= 1  # bias should lean toward STORE
