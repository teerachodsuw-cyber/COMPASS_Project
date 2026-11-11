"""Per-outcome checkpoint selection (configs/training.yaml: training_loop.checkpoint_selection).

Synthetic data only. Validation losses are crafted (Trainer.evaluate replaced) so the best epoch
of each outcome, and the early-stopping epoch, are known in advance; training itself is real,
so every epoch's weights differ and checkpoints can be matched to the epoch they came from.
"""

import copy
import importlib.util
import json

import numpy as np
import pytest
import torch

from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation import pipeline
from compass.training import tuning
from compass.training.train import (
    BEST_TOTAL_CHECKPOINT,
    SELECTION_FILE,
    best_checkpoint_name,
    load_selected_models,
    outcome_checkpoints,
)

OUT = ("mortality", "malo", "cvd")
N_PATIENTS, DATA_SEED, PATIENCE = 300, 42, 3
SCHEMA = tuning.load_yaml("configs/data_schema.yaml")
TRAINING_CFG = tuning.load_yaml("configs/training.yaml")
ENTRY = {"model": "dynamic_deephit", "config_id": "x", "learning_rate": 0.001, "weight_decay": 0.0}

_spec = importlib.util.spec_from_file_location("evaluate_script_cs", "scripts/evaluate.py")
evaluate_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_script)

# (mortality, MALO, CVD) val NLL per epoch. Bests: mortality @1, MALO @3, CVD @5; nothing
# improves after epoch 5, so per-outcome early stopping (patience 3) stops after epoch 8.
# Val total (= sum here) is best @3, so total-loss early stopping stops after epoch 6.
SEQ = [
    (1.00, 1.00, 1.00),
    (0.90, 0.95, 1.10),
    (0.95, 0.90, 1.20),
    (0.96, 0.80, 0.95),
    (0.97, 0.85, 1.30),
    (0.98, 0.86, 0.90),
    (0.99, 0.87, 1.40),
    (1.00, 0.88, 1.50),
    (1.01, 0.89, 1.60),
] + [(0.50, 0.50, 0.50)] * 3  # would be an improvement -- reached only if stopping is wrong
BEST = {"mortality": 1, "malo": 3, "cvd": 5}


def _losses(mort, malo, cvd, ranking=0.0, aux=0.0):
    nll = mort + malo + cvd
    return {"nll_mortality": mort, "nll_malo": malo, "nll_cvd": cvd, "survival_nll": nll,
            "ranking_loss": ranking, "auxiliary_loss": aux, "total": nll + 0.1 * ranking + 0.1 * aux}


@pytest.fixture(scope="module")
def data():
    return generate_synthetic_cohort(n_patients=N_PATIENTS, seed=DATA_SEED)


@pytest.fixture(scope="module")
def split(data, tmp_path_factory):
    visits, outcomes = data
    return tuning.resolve_split(tmp_path_factory.mktemp("split"), visits, outcomes, TRAINING_CFG)


def _cfg(mode):
    return {**TRAINING_CFG, "training_loop": {**TRAINING_CFG["training_loop"], "checkpoint_selection": mode}}


def _trainer(data, split, mode="per_outcome"):
    visits, outcomes = data
    return tuning.make_trainer(ENTRY, visits, outcomes, SCHEMA, _cfg(mode), split)


def _craft(trainer, rows, snapshots, extras=None):
    """Replace trainer.evaluate: return the crafted losses row by row, recording the model's
    weights at each epoch (the weights a checkpoint saved that epoch must hold)."""
    rows = iter(list(rows))
    extras = iter(list(extras) if extras is not None else [])

    def fake_evaluate(patients, resample_landmarks=False):
        snapshots.append(copy.deepcopy(trainer.model.state_dict()))
        return _losses(*next(rows), *next(extras, (0.0, 0.0)))

    trainer.evaluate = fake_evaluate


def _weights(path):
    return torch.load(path, weights_only=False)["model_state_dict"]


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def per_outcome_run(data, split, tmp_path_factory):
    out = tmp_path_factory.mktemp("per_outcome") / "dynamic_deephit"
    trainer, model = _trainer(data, split)
    snapshots = []
    _craft(trainer, SEQ, snapshots)
    history = trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=50, checkpoint_dir=str(out),
                          early_stopping_patience=PATIENCE)
    return out, trainer, model, history, snapshots


def test_each_outcome_checkpoint_is_saved_at_its_own_best_epoch(per_outcome_run):
    out, _, _, history, snapshots = per_outcome_run
    assert history["checkpoint_selection"] == "per_outcome"
    assert history["best_epoch_per_outcome"] == BEST
    assert history["best_val_nll_per_outcome"] == {o: SEQ[BEST[o]][k] for k, o in enumerate(OUT)}
    for o in OUT:
        ckpt = torch.load(out / best_checkpoint_name(o), weights_only=False)
        assert ckpt["epoch"] == BEST[o]
        assert _same(ckpt["model_state_dict"], snapshots[BEST[o]])  # the weights of that epoch
    # three different epochs -> three different weight sets
    assert not _same(_weights(out / best_checkpoint_name("mortality")), _weights(out / best_checkpoint_name("cvd")))
    sel = json.loads((out / SELECTION_FILE).read_text())
    assert sel["checkpoint_selection"] == "per_outcome"
    assert sel["checkpoints"] == {o: best_checkpoint_name(o) for o in OUT}
    assert sel["best_epoch"] == BEST and sel["best_total_epoch"] == 3
    assert torch.load(out / BEST_TOTAL_CHECKPOINT, weights_only=False)["epoch"] == 3  # still written


def test_training_continues_while_any_outcome_improves(per_outcome_run):
    _, _, _, history, _ = per_outcome_run
    # mortality stopped improving at epoch 1, but MALO (3) and CVD (5) reset the counter;
    # it stops after epoch 5 + patience 3 = epoch 8, i.e. 9 epochs, never reaching epoch 9.
    assert len(history["train"]) == len(history["val"]) == BEST["cvd"] + PATIENCE + 1
    assert history["epochs_without_improvement"] == PATIENCE


def test_ranking_and_auxiliary_terms_do_not_affect_selection(data, split, per_outcome_run, tmp_path):
    out_a, _, _, history_a, _ = per_outcome_run
    # Same per-outcome NLLs, but ranking / auxiliary losses that move "total" around wildly
    # (total's best epoch becomes 8 instead of 3).
    extras = [(50.0, 50.0)] * 8 + [(-40.0, -40.0)] + [(0.0, 0.0)] * 3
    trainer, _ = _trainer(data, split)
    _craft(trainer, SEQ, [], extras)
    history_b = trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=50, checkpoint_dir=str(tmp_path),
                            early_stopping_patience=PATIENCE)
    assert history_b["best_epoch"] == 8 != history_a["best_epoch"]       # total really did change
    assert history_b["best_epoch_per_outcome"] == history_a["best_epoch_per_outcome"] == BEST
    assert len(history_b["train"]) == len(history_a["train"])            # same stopping epoch
    for o in OUT:                                                        # same selected weights
        assert _same(_weights(tmp_path / best_checkpoint_name(o)), _weights(out_a / best_checkpoint_name(o)))


def test_ddh_per_outcome_nll_excludes_ranking_and_auxiliary_terms(data, split):
    trainer, _ = _trainer(data, split)
    losses = trainer.evaluate(trainer.val_ids)
    assert sum(losses[f"nll_{o}"] for o in OUT) == pytest.approx(losses["survival_nll"], rel=1e-5)
    assert losses["total"] > losses["survival_nll"]  # total also carries ranking / auxiliary terms


def test_evaluation_uses_outcome_k_checkpoint_for_outcome_k(data, per_outcome_run):
    out, _, _, _, _ = per_outcome_run
    visits, outcomes = data
    trainer, models = evaluate_script.load_trained("dynamic_deephit", str(out.parent), visits, outcomes, SCHEMA, TRAINING_CFG)
    assert isinstance(models, dict) and set(models) == set(OUT)
    for o in OUT:
        assert _same(models[o].state_dict(), _weights(out / best_checkpoint_name(o)))

    selected = pipeline.predict_selected_cifs(models, trainer, "val", 10)
    for o in OUT:
        np.testing.assert_array_equal(selected[o], pipeline.predict_cifs(models[o], trainer, "val", 10)[o])
    other = pipeline.predict_cifs(models["mortality"], trainer, "val", 10)
    assert not np.allclose(selected["cvd"], other["cvd"])  # really a different model per outcome

    # End to end: outcome k's metrics equal those of evaluating model k alone.
    # (np.testing.assert_equal: NaN == NaN for an outcome with no val events in this small cohort)
    full = evaluate_script.evaluate_models({"dynamic_deephit": trainer}, {"dynamic_deephit": models}, "val", 10, 2, 0)
    finite = 0
    for o in OUT:
        alone = evaluate_script.evaluate_models({"dynamic_deephit": trainer}, {"dynamic_deephit": models[o]}, "val", 10, 2, 0)
        pick = lambda res: {r["metric_name"]: r["point_estimate"] for r in res["per_model"] if r["outcome"] == o}  # noqa: E731
        np.testing.assert_equal(pick(full), pick(alone))
        finite += int(np.isfinite(pick(full)["cindex"]))
    assert finite >= 2


def test_resume_restores_per_outcome_bests_and_counter(data, split, tmp_path):
    first, _ = _trainer(data, split)
    _craft(first, SEQ[:7], [])
    first.fit(first.train_ids, first.val_ids, num_epochs=7, checkpoint_dir=str(tmp_path), early_stopping_patience=PATIENCE)
    # after epoch 6: bests from epochs 1 / 3 / 5, one epoch without improvement

    second, _ = _trainer(data, split)
    second.resume(str(tmp_path / "checkpoint_latest.pt"))
    state = second._resume_selection_state
    assert state["best_epoch"] == {"total": 3, **BEST}
    assert state["best_val"]["mortality"] == 0.90 and state["best_val"]["malo"] == 0.80 and state["best_val"]["cvd"] == 0.90
    assert state["epochs_without_improvement"] == 1

    _craft(second, SEQ[7:], [])
    history = second.fit(second.train_ids, second.val_ids, num_epochs=50, checkpoint_dir=str(tmp_path),
                         early_stopping_patience=PATIENCE)
    # Counter carried over: epochs 7 and 8 bring it to 3 -> stop. Fresh bests (the old bug,
    # audit #17) would count epoch 7 as an improvement for everything and keep going.
    assert len(history["train"]) == 2 and history["epochs_without_improvement"] == PATIENCE
    assert history["best_epoch_per_outcome"] == BEST
    for o in OUT:
        assert torch.load(tmp_path / best_checkpoint_name(o), weights_only=False)["epoch"] == BEST[o]
    assert torch.load(tmp_path / BEST_TOTAL_CHECKPOINT, weights_only=False)["epoch"] == 3
    assert json.loads((tmp_path / SELECTION_FILE).read_text())["best_epoch"] == BEST


def test_resume_refuses_a_different_selection_mode(data, split, per_outcome_run):
    out, _, _, _, _ = per_outcome_run
    trainer, _ = _trainer(data, split, mode="total")
    with pytest.raises(ValueError, match="checkpoint_selection"):
        trainer.resume(str(out / "checkpoint_latest.pt"))


def test_total_selection_reproduces_previous_behaviour(data, split, tmp_path):
    trainer, model = _trainer(data, split, mode="total")
    snapshots = []
    _craft(trainer, SEQ, snapshots)
    history = trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=50, checkpoint_dir=str(tmp_path),
                          early_stopping_patience=PATIENCE)
    totals = [sum(r) for r in SEQ]
    best = int(np.argmin(totals[:7]))
    assert best == 3 and history["best_epoch"] == best and history["best_val_loss"] == pytest.approx(totals[best])
    assert len(history["train"]) == best + PATIENCE + 1                   # early stopping on total only
    assert not any((tmp_path / best_checkpoint_name(o)).exists() for o in OUT)
    assert outcome_checkpoints(tmp_path) == {o: tmp_path / BEST_TOTAL_CHECKPOINT for o in OUT}
    assert _same(model.state_dict(), snapshots[best])                     # best-total weights reloaded
    loaded = load_selected_models(copy.deepcopy(model), tmp_path)
    assert isinstance(loaded, torch.nn.Module) and _same(loaded.state_dict(), snapshots[best])


def test_runs_without_selection_file_use_checkpoint_best_for_every_outcome(tmp_path):
    # e.g. runs/full_v1, trained before per-outcome selection
    assert outcome_checkpoints(tmp_path) == {o: tmp_path / BEST_TOTAL_CHECKPOINT for o in OUT}


def test_unknown_selection_mode_is_rejected(data, split):
    with pytest.raises(ValueError, match="checkpoint_selection"):
        _trainer(data, split, mode="best_guess")
