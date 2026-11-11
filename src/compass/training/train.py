"""Architecture-agnostic training loop shared by all three COMPASS candidate architectures.

Before this file, no training loop existed anywhere in this repo -- each model's own build
session verified it could learn via a throwaway, uncommitted script (e.g. the
test_overfits_tiny_batch pattern in tests/test_dynamic_deephit.py is the one piece of that
which DID get committed, as a unit test; TraCeR and SurvLatent ODE's equivalent throwaway
verification never became reusable infrastructure). This is that infrastructure, built once,
for real.

Design: ONE shared Trainer class, not three per-architecture training scripts. The
orchestration mechanics -- epoch iteration, minibatching, optimizer step, checkpointing,
logging, early stopping -- are identical across all three architectures; duplicating that
logic three times would risk subtle divergences becoming a confound in the eventual paired
model comparison. Trainer never looks at any model's internal loss structure: it calls the
shared model.compute_loss(batch) -> dict[str, Tensor] interface (base.py): it optimises the
guaranteed "total" key, and selects checkpoints / early-stops on the per-outcome survival NLLs
("nll_mortality", "nll_malo", "nll_cvd" -- every model returns these) or, with
training_loop.checkpoint_selection: total, on "total". Other keys (survival_nll,
ranking_loss, auxiliary_loss, ...) are logged only and never used for selection.

Checkpoint selection (configs/training.yaml: training_loop.checkpoint_selection):
- per_outcome (default): each outcome's validation NLL is tracked on its own; when it beats
  that outcome's best so far, checkpoint_best_<outcome>.pt is saved. Early stopping waits
  until NONE of the three outcomes has improved for `patience` epochs (any improvement resets
  the counter). Evaluation uses outcome k's checkpoint for outcome k (load_selected_models).
- total: the previous behaviour -- one checkpoint_best.pt on validation "total" loss, early
  stopping on "total", every outcome evaluated from checkpoint_best.pt.
checkpoint_best.pt (best "total") is written in both modes. fit() writes
checkpoint_selection.json (mode, which checkpoint each outcome uses, best epochs / losses).
The per-outcome bests and the patience counter are saved in every checkpoint and restored by
resume().

Landmark resampling policy (configs/training.yaml: landmarking.sampling =
"one_random_landmark_per_subject_per_epoch"): TRAINING patients get a fresh random landmark
(via compass.data.landmarking.sample_landmark) every epoch -- this is what
compass.data.preprocessing.run_preprocessing's resample_each_epoch/epoch parameters already
document as the intended training-time behavior. EVALUATION (val/test) patients get ONE fixed
landmark, sampled once and reused for every evaluate() call across the whole training run --
otherwise validation loss/early-stopping would be comparing against a moving target epoch to
epoch, making "did the model actually improve" unanswerable.

Deliberately NOT implemented as "call run_preprocessing(resample_each_epoch=True, epoch=N)
every epoch": that would ALSO refit the feature scalers from scratch every epoch (scalers are
fit from whichever landmark-conditioned visit data happens to be in run_preprocessing's
`train_examples` for that call), introducing unwanted epoch-to-epoch drift in the input
feature SCALE itself, on top of the intended landmark variety -- and scaler refitting is not
free (it now happens every epoch instead of once at setup). Instead, Trainer fits scalers
ONCE at __init__ (via one run_preprocessing call) and reuses compass.data.preprocessing.
scale_and_mask_examples -- the exact function run_preprocessing itself calls internally, now
exposed as a public, reusable, single source of truth -- to scale/mask each epoch's freshly
resampled landmark examples with those same fixed scalers. Same underlying functions either
way (sample_landmark, construct_landmark_example, scale_and_mask_examples, collate_examples);
just orchestrated directly instead of through the full run_preprocessing wrapper, specifically
to avoid the scaler-refit side effect.

Deviation from the originally sketched Trainer.__init__(model, model_config, training_config)
signature: raw data (visits_df, outcomes_df) and schema info (continuous_columns,
medication_columns, sex_categories, smoking_status_categories, horizons,
administrative_cutoff) are ALSO required constructor arguments. Without them the class has no
way to build landmark examples at all -- train_one_epoch/evaluate take bare patient_id lists,
not raw per-patient data, so the Trainer itself must hold the data to look patients up by id.
"""

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compass.data.landmarking import construct_landmark_example, death_date_of, sample_landmark
from compass.data.preprocessing import run_preprocessing, scale_and_mask_examples
from compass.data.splitting import data_fingerprint, save_split
from compass.models.base import OUTCOME_NAMES
from compass.training.batching import collate_examples
from compass.training.targets import compute_last_visit_dates

# Fixed, epoch-independent seed offset for the ONE-TIME val/test landmark sample built at
# Trainer setup -- matches compass.data.preprocessing.run_preprocessing's own convention
# (seed + 100_000 for val, seed + 200_000 for test) exactly, since that initial
# run_preprocessing call is in fact how these fixed examples are built.
_RESAMPLE_LANDMARKS_SEED_OFFSET = 900_000  # used only when evaluate(resample_landmarks=True)

CHECKPOINT_SELECTION_MODES = ("per_outcome", "total")
SELECTION_FILE = "checkpoint_selection.json"
BEST_TOTAL_CHECKPOINT = "checkpoint_best.pt"


def best_checkpoint_name(outcome: str) -> str:
    return f"checkpoint_best_{outcome}.pt"


def _fresh_selection_state() -> dict:
    """Bests per selection key ("total" + each outcome) and the early-stopping counter."""
    keys = ("total", *OUTCOME_NAMES)
    return {"best_val": {k: math.inf for k in keys}, "best_epoch": {k: None for k in keys}, "epochs_without_improvement": 0}


def outcome_checkpoints(checkpoint_dir) -> dict:
    """{outcome: checkpoint path} as recorded by fit() in checkpoint_selection.json. A run
    without that file (trained before per-outcome selection) has one checkpoint_best.pt,
    used for every outcome."""
    d = Path(checkpoint_dir)
    f = d / SELECTION_FILE
    if not f.exists():
        return {o: d / BEST_TOTAL_CHECKPOINT for o in OUTCOME_NAMES}
    files = json.loads(f.read_text())["checkpoints"]
    return {o: d / files[o] for o in OUTCOME_NAMES}


def load_selected_models(model, checkpoint_dir):
    """The weights evaluation uses. If every outcome uses the same checkpoint (total
    selection, or older runs): `model` itself with those weights. Otherwise
    {outcome: a copy of `model` with that outcome's checkpoint weights} -- outcome k's
    predictions must then come from entry k only (compass.evaluation.pipeline)."""
    paths = outcome_checkpoints(checkpoint_dir)

    def load_into(m, path):
        m.load_state_dict(torch.load(path, weights_only=False)["model_state_dict"])
        m.eval()
        return m

    if len(set(paths.values())) == 1:
        return load_into(model, next(iter(paths.values())))
    return {o: load_into(copy.deepcopy(model), path) for o, path in paths.items()}


class Trainer:
    def __init__(
        self,
        model,
        model_config: dict,
        training_config: dict,
        visits_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        continuous_columns: list[str],
        medication_columns: list[str],
        sex_categories: tuple,
        smoking_status_categories: list,
        horizons: list[int],
        administrative_cutoff: pd.Timestamp,
        split: dict | None = None,
    ):
        """model: any of the three architectures, satisfying base.py's shared interface.
        model_config: from configs/model_*.yaml (per-architecture hyperparameters -- reads
        learning_rate, weight_decay (AdamW, default 0), batch_size, num_time_bins,
        bin_width_years; optional "scheduler" dict
        for an LR scheduler, e.g. {"type": "StepLR", "step_size": 10, "gamma": 0.5} --
        none of the current configs set one, so the default is no scheduler).
        training_config: from configs/training.yaml (reads seed, splitting.train_frac/
        val_frac/test_frac, training_loop.checkpoint_selection: per_outcome (default) | total).
        visits_df, outcomes_df, continuous_columns, medication_columns, sex_categories,
        smoking_status_categories, horizons, administrative_cutoff: everything else
        run_preprocessing() needs -- see module docstring for why these are required here
        too, beyond the model/model_config/training_config originally sketched.
        split: optional pinned split {"train", "val", "test": patient IDs} (from a run's
        split.json); None = derive it with patient_level_split from training_config's seed.
        fit() saves the split actually used, plus a data fingerprint and the seed, to its
        checkpoint_dir/split.json, so evaluation reuses it instead of re-deriving it.
        """
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.outcomes_df = outcomes_df
        self.horizons = list(horizons)
        self.administrative_cutoff = pd.Timestamp(administrative_cutoff)
        self.bin_width_years = float(model_config["bin_width_years"])
        self.num_time_bins = int(model_config["num_time_bins"])
        self.batch_size = int(model_config["batch_size"])
        self.seed = int(training_config["seed"])
        self.checkpoint_selection = (training_config.get("training_loop") or {}).get("checkpoint_selection", "per_outcome")
        if self.checkpoint_selection not in CHECKPOINT_SELECTION_MODES:
            raise ValueError(f"training_loop.checkpoint_selection must be one of {CHECKPOINT_SELECTION_MODES}, "
                             f"got {self.checkpoint_selection!r}")

        self.continuous_columns = list(continuous_columns)
        self.medication_columns = list(medication_columns)
        self.sex_categories = tuple(sex_categories)
        self.smoking_status_categories = list(smoking_status_categories)

        # AdamW for all three models. With weight_decay = 0 it is exactly the previous plain Adam.
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config.get("weight_decay", 0.0)),
        )
        self.scheduler = self._build_scheduler(model_config.get("scheduler"))

        # ONE run_preprocessing call: fits scalers, builds the FIXED val/test landmark
        # examples (seed-offset, epoch-independent -- see module docstring), and gives us the
        # deterministic train/val/test patient_id split (via patient_level_split, called
        # internally). Never called again after this -- see module docstring for why.
        splitting = training_config["splitting"]
        result = run_preprocessing(
            visits_df,
            outcomes_df,
            self.continuous_columns,
            self.medication_columns,
            self.sex_categories,
            self.smoking_status_categories,
            self.horizons,
            train_frac=float(splitting["train_frac"]),
            val_frac=float(splitting["val_frac"]),
            test_frac=float(splitting["test_frac"]),
            seed=self.seed,
            resample_each_epoch=False,
            epoch=0,
            split=split,
            eval_landmark=training_config.get("evaluation", {}).get("landmark", "first_post_index"),
        )
        self.eval_landmark = result.eval_landmark
        self.data_fingerprint = data_fingerprint(visits_df, outcomes_df)
        self.scalers = result.scalers
        self.feature_names = result.feature_names

        self.train_ids = [ex["patient_id"] for ex in result.train]
        self.val_ids = [ex["patient_id"] for ex in result.val]
        self.test_ids = [ex["patient_id"] for ex in result.test]

        # Fixed eval examples, keyed by patient_id, covering both val and test -- reused by
        # every evaluate(resample_landmarks=False) call across the whole training run.
        self._fixed_eval_examples = {ex["patient_id"]: ex for ex in result.val + result.test}
        # TRAINING-split examples at the EVALUATION landmark rule (same rule as val / test):
        # never used for training (training resamples random landmarks every epoch) -- only
        # as the evaluation pipeline's IPCW censoring reference (compass.evaluation.pipeline),
        # which must come from the training split and use the same landmark rule.
        self._fixed_train_examples = list(result.train_eval)

        # Raw per-patient data, for resampling fresh TRAINING landmarks each epoch (and for
        # evaluate(resample_landmarks=True)'s fresh draws).
        self._visits_by_patient = {
            pid: group.sort_values("visit_date").reset_index(drop=True) for pid, group in visits_df.groupby("patient_id")
        }
        self._outcomes_indexed = outcomes_df.set_index("patient_id")
        # Last visit per patient from the FULL visit table -- computed once, before any
        # landmark truncation (MALO/CVD censoring uses it; see compass.training.targets).
        self.last_visit_dates = compute_last_visit_dates(visits_df)

        self._resume_epoch = None
        self._resume_rng_state = None
        self._resume_selection_state = None

    def fixed_examples(self, split: str) -> list[dict]:
        """The split's fixed landmark examples (scaled + masked), in split order, all at the
        evaluation landmark rule (configs/training.yaml: evaluation.landmark).
        train: the IPCW reference only (see __init__); val / test: the same fixed landmarks
        evaluate(resample_landmarks=False) uses."""
        if split == "train":
            return list(self._fixed_train_examples)
        ids = {"val": self.val_ids, "test": self.test_ids}[split]
        return [self._fixed_eval_examples[pid] for pid in ids]

    def _build_scheduler(self, scheduler_config: dict | None):
        if not scheduler_config:
            return None
        scheduler_config = dict(scheduler_config)
        scheduler_type = scheduler_config.pop("type")
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type)
        return scheduler_cls(self.optimizer, **scheduler_config)

    def _build_examples_for(self, patient_ids: list, rng: np.random.Generator) -> list[dict]:
        """Sample ONE fresh landmark per patient (via sample_landmark), construct the raw
        landmark example, then scale+mask with the ALREADY-FIT self.scalers -- never refits.
        """
        raw_examples = []
        for pid in patient_ids:
            patient_visits = self._visits_by_patient[pid]
            patient_outcomes = self._outcomes_indexed.loc[pid]
            landmark_date = sample_landmark(
                patient_visits, rng, index_date=patient_outcomes["index_date"], death_date=death_date_of(patient_outcomes)
            )
            raw_examples.append(construct_landmark_example(patient_visits, patient_outcomes, landmark_date))
        return scale_and_mask_examples(
            raw_examples, self.scalers, self.continuous_columns, self.medication_columns,
            self.sex_categories, self.smoking_status_categories, self.feature_names,
        )

    def _iter_minibatches(self, patient_ids: list, rng: np.random.Generator | None) -> list[list]:
        ids = list(patient_ids)
        if rng is not None:
            order = rng.permutation(len(ids))
            ids = [ids[i] for i in order]
        return [ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

    def _run_batch(self, examples: list[dict], train: bool) -> dict[str, float]:
        batch = collate_examples(
            examples, self.outcomes_df, self.administrative_cutoff, self.bin_width_years, self.num_time_bins,
            last_visit_dates=self.last_visit_dates,
        )
        if train:
            self.optimizer.zero_grad()
            loss_dict = self.model.compute_loss(batch)
            loss_dict["total"].backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                loss_dict = self.model.compute_loss(batch)
        return {k: float(v.item()) for k, v in loss_dict.items()}

    def _average_over_batches(self, per_batch: list[tuple[dict[str, float], int]]) -> dict[str, float]:
        """Size-weighted average of per-batch loss dicts (weighted by minibatch size, so a
        smaller trailing minibatch doesn't get equal say to a full-size one)."""
        total_n = sum(n for _, n in per_batch)
        keys = per_batch[0][0].keys()
        return {k: sum(d[k] * n for d, n in per_batch) / total_n for k in keys}

    def train_one_epoch(self, train_patients: list, rng: np.random.Generator) -> dict:
        """One training epoch: fresh random landmark per patient (sample_landmark, respecting
        one_random_landmark_per_subject_per_epoch), minibatched (model_config's batch_size,
        order shuffled via `rng`), forward + compute_loss + backward + optimizer step per
        minibatch. `rng` is the SAME generator across the whole fit() call, advanced epoch to
        epoch (not re-seeded) -- that's what makes each epoch's landmarks different while the
        whole run stays reproducible from one initial seed.

        Returns the size-weighted epoch-average of whatever keys model.compute_loss returns
        (e.g. {"survival_nll", "ranking_loss", "auxiliary_loss", "total"} for Dynamic-DeepHit;
        {"survival_nll", "total"} for TraCeR/SurvLatent ODE, plus nll_<outcome> for each
        outcome from every model) -- generic over which keys exist.
        """
        self.model.train()
        per_batch = []
        for minibatch_ids in self._iter_minibatches(train_patients, rng):
            examples = self._build_examples_for(minibatch_ids, rng)
            per_batch.append((self._run_batch(examples, train=True), len(minibatch_ids)))
        return self._average_over_batches(per_batch)

    def evaluate(self, patients: list, resample_landmarks: bool = False) -> dict:
        """resample_landmarks=False (default): use the FIXED landmark examples built once at
        __init__ (self._fixed_eval_examples) -- same landmark for a given patient across every
        call, for comparable val loss epoch to epoch. resample_landmarks=True: draw a genuinely
        fresh random landmark per patient this call (not reproducible across calls by design --
        this is a diagnostic option, not the training-time eval path).
        """
        self.model.eval()
        if resample_landmarks:
            rng = np.random.default_rng()
            examples_by_id = {pid: ex for pid, ex in zip(patients, self._build_examples_for(patients, rng))}
        else:
            examples_by_id = self._fixed_eval_examples

        per_batch = []
        for minibatch_ids in self._iter_minibatches(patients, rng=None):
            examples = [examples_by_id[pid] for pid in minibatch_ids]
            per_batch.append((self._run_batch(examples, train=False), len(minibatch_ids)))
        return self._average_over_batches(per_batch)

    @staticmethod
    def _epoch_line(epoch: int, train_metrics: dict, val_metrics: dict, elapsed: float) -> str:
        """Console line per epoch: total and per-outcome survival NLL (nll_<outcome>) for
        train and val -- the same values written to training_log.jsonl."""
        def part(m):
            per_outcome = " ".join(f"{k[4:]}={m[k]:.4f}" for k in ("nll_mortality", "nll_malo", "nll_cvd") if k in m)
            return f"total={m['total']:.4f} [{per_outcome}]"
        return f"  epoch {epoch}: train {part(train_metrics)} | val {part(val_metrics)} | {elapsed:.1f}s"

    def _checkpoint(self, path: Path, epoch: int, rng: np.random.Generator, val_loss: float, selection_state: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "rng_state": rng.bit_generator.state,
                "val_loss": val_loss,
                "checkpoint_selection": self.checkpoint_selection,
                "selection_state": copy.deepcopy(selection_state),
            },
            path,
        )

    def _update_bests(self, state: dict, epoch: int, val_metrics: dict) -> list[str]:
        """Update the bests in `state` from this epoch's validation metrics; return the keys
        that improved. Outcome keys read ONLY the per-outcome survival NLL (nll_<outcome>),
        never "total" or any ranking / auxiliary term."""
        improved = []
        for key in ("total", *OUTCOME_NAMES):
            metric = "total" if key == "total" else f"nll_{key}"
            if metric not in val_metrics:
                if self.checkpoint_selection == "per_outcome" and key != "total":
                    raise KeyError(f"per-outcome checkpoint selection needs '{metric}' in the validation losses")
                continue
            value = float(val_metrics[metric])
            if value < state["best_val"][key]:
                state["best_val"][key] = value
                state["best_epoch"][key] = epoch
                improved.append(key)
        return improved

    def _write_selection(self, checkpoint_dir: Path, state: dict):
        per_outcome = self.checkpoint_selection == "per_outcome"
        finite = lambda v: None if math.isinf(v) else v  # noqa: E731
        summary = {
            "checkpoint_selection": self.checkpoint_selection,
            "checkpoints": {o: best_checkpoint_name(o) if per_outcome else BEST_TOTAL_CHECKPOINT for o in OUTCOME_NAMES},
            "best_epoch": {o: state["best_epoch"][o] for o in OUTCOME_NAMES},
            "best_val_nll": {o: finite(state["best_val"][o]) for o in OUTCOME_NAMES},
            "best_total_epoch": state["best_epoch"]["total"],
            "best_val_total": finite(state["best_val"]["total"]),
            "epochs_without_improvement": state["epochs_without_improvement"],
        }
        (checkpoint_dir / SELECTION_FILE).write_text(json.dumps(summary, indent=2) + "\n")

    def resume(self, checkpoint_path: str) -> np.random.Generator:
        """Load a checkpoint (model + optimizer + scheduler + rng state + the per-outcome /
        total bests and the early-stopping counter) and record where to continue from. Call
        this BEFORE fit() -- fit() checks for a pending resume and continues from the
        checkpointed epoch/rng state/bests/counter instead of starting fresh. Returns the
        restored rng (also stored internally), in case the caller wants it directly.
        A checkpoint from before selection state was saved resumes with fresh bests (warned).
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint["rng_state"]

        saved_mode = checkpoint.get("checkpoint_selection")
        if saved_mode is not None and saved_mode != self.checkpoint_selection:
            raise ValueError(f"checkpoint was trained with checkpoint_selection={saved_mode!r}; "
                             f"config says {self.checkpoint_selection!r}")
        if checkpoint.get("selection_state") is None:
            print("warning: checkpoint has no selection state (older run); resuming with fresh bests", flush=True)
        self._resume_selection_state = copy.deepcopy(checkpoint.get("selection_state"))

        self._resume_epoch = int(checkpoint["epoch"])
        self._resume_rng_state = checkpoint["rng_state"]
        return rng

    def fit(
        self,
        train_patients: list,
        val_patients: list,
        num_epochs: int,
        checkpoint_dir: str,
        early_stopping_patience: int | None = None,
        checkpoint_every_n_epochs: int = 1,
    ) -> dict:
        """Full training loop: iterate epochs, train_one_epoch + evaluate each epoch, log both
        (JSON lines, checkpoint_dir/training_log.jsonl, appended across resumes), checkpoint
        (checkpoint_dir/checkpoint_latest.pt every checkpoint_every_n_epochs epochs;
        checkpoint_dir/checkpoint_best.pt whenever val "total" loss improves; with
        per_outcome selection also checkpoint_best_<outcome>.pt whenever that outcome's val
        NLL improves). Early stopping (if early_stopping_patience is set): per_outcome --
        stop once NONE of the three outcomes' val NLL has improved for that many consecutive
        epochs; total -- stop once val "total" hasn't. Fully reproducible given the same
        training_config seed AND the same train_patients/val_patients ordering (the seed
        drives the one `rng` used for every epoch's landmark sampling and minibatch
        shuffling, start to finish).

        If resume() was called first, continues from that checkpoint's epoch/rng state,
        bests and early-stopping counter instead of starting fresh at epoch 0.

        After the loop, the model's weights are RELOADED from checkpoint_best.pt (the
        best-total epoch); history["best_epoch"] / ["best_val_loss"] record which epoch that
        was. With per_outcome selection, evaluation instead loads each outcome's own
        checkpoint (load_selected_models); history["best_epoch_per_outcome"] /
        ["best_val_nll_per_outcome"] record those.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path = checkpoint_dir / "training_log.jsonl"
        # Pin the split (+ data fingerprint + seed) next to the checkpoints: evaluation
        # loads it from here instead of re-deriving it (compass.data.splitting.load_split).
        save_split(checkpoint_dir, self.train_ids, self.val_ids, self.test_ids, self.data_fingerprint, self.seed)

        if self._resume_epoch is not None:
            start_epoch = self._resume_epoch + 1
            rng = np.random.default_rng()
            rng.bit_generator.state = self._resume_rng_state
        else:
            start_epoch = 0
            rng = np.random.default_rng(self.seed)

        state = copy.deepcopy(self._resume_selection_state) if self._resume_selection_state else _fresh_selection_state()
        stop_keys = OUTCOME_NAMES if self.checkpoint_selection == "per_outcome" else ("total",)
        history = {"train": [], "val": []}

        for epoch in range(start_epoch, num_epochs):
            t0 = time.monotonic()
            train_metrics = self.train_one_epoch(train_patients, rng)
            val_metrics = self.evaluate(val_patients, resample_landmarks=False)
            if self.scheduler is not None:
                self.scheduler.step()
            elapsed = time.monotonic() - t0

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics, "elapsed_seconds": elapsed}) + "\n")
            print(self._epoch_line(epoch, train_metrics, val_metrics, elapsed), flush=True)

            val_loss = val_metrics["total"]
            improved = self._update_bests(state, epoch, val_metrics)
            if any(k in improved for k in stop_keys):
                state["epochs_without_improvement"] = 0
            else:
                state["epochs_without_improvement"] += 1

            if "total" in improved:
                self._checkpoint(checkpoint_dir / BEST_TOTAL_CHECKPOINT, epoch, rng, val_loss, state)
            if self.checkpoint_selection == "per_outcome":
                for outcome in OUTCOME_NAMES:
                    if outcome in improved:
                        self._checkpoint(checkpoint_dir / best_checkpoint_name(outcome), epoch, rng, val_loss, state)
            if (epoch + 1) % checkpoint_every_n_epochs == 0:
                self._checkpoint(checkpoint_dir / "checkpoint_latest.pt", epoch, rng, val_loss, state)
            self._write_selection(checkpoint_dir, state)

            if early_stopping_patience is not None and state["epochs_without_improvement"] >= early_stopping_patience:
                break

        best_path = checkpoint_dir / BEST_TOTAL_CHECKPOINT
        if best_path.exists():
            best = torch.load(best_path, weights_only=False)
            self.model.load_state_dict(best["model_state_dict"])
            history["best_epoch"] = int(best["epoch"])
            history["best_val_loss"] = float(best["val_loss"])
        history["checkpoint_selection"] = self.checkpoint_selection
        history["best_epoch_per_outcome"] = {o: state["best_epoch"][o] for o in OUTCOME_NAMES}
        history["best_val_nll_per_outcome"] = {o: state["best_val"][o] for o in OUTCOME_NAMES}
        history["epochs_without_improvement"] = state["epochs_without_improvement"]
        return history
