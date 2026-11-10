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
shared model.compute_loss(batch) -> dict[str, Tensor] interface (base.py) and only ever reads
the guaranteed "total" key for optimization/checkpointing/early-stopping decisions, plus
whatever OTHER keys happen to be present (survival_nll, ranking_loss, ...) purely for logging
-- same principle as base.py's shared forward/predict_risk interface.

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

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compass.data.landmarking import construct_landmark_example, death_date_of, sample_landmark
from compass.data.preprocessing import run_preprocessing, scale_and_mask_examples
from compass.data.splitting import data_fingerprint, save_split
from compass.training.batching import collate_examples
from compass.training.targets import compute_last_visit_dates

# Fixed, epoch-independent seed offset for the ONE-TIME val/test landmark sample built at
# Trainer setup -- matches compass.data.preprocessing.run_preprocessing's own convention
# (seed + 100_000 for val, seed + 200_000 for test) exactly, since that initial
# run_preprocessing call is in fact how these fixed examples are built.
_RESAMPLE_LANDMARKS_SEED_OFFSET = 900_000  # used only when evaluate(resample_landmarks=True)


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
        learning_rate, batch_size, num_time_bins, bin_width_years; optional "scheduler" dict
        for an LR scheduler, e.g. {"type": "StepLR", "step_size": 10, "gamma": 0.5} --
        none of the current configs set one, so the default is no scheduler).
        training_config: from configs/training.yaml (reads seed, splitting.train_frac/
        val_frac/test_frac).
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

        self.continuous_columns = list(continuous_columns)
        self.medication_columns = list(medication_columns)
        self.sex_categories = tuple(sex_categories)
        self.smoking_status_categories = list(smoking_status_categories)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(model_config["learning_rate"]))
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
        )
        self.data_fingerprint = data_fingerprint(visits_df, outcomes_df)
        self.scalers = result.scalers
        self.feature_names = result.feature_names

        self.train_ids = [ex["patient_id"] for ex in result.train]
        self.val_ids = [ex["patient_id"] for ex in result.val]
        self.test_ids = [ex["patient_id"] for ex in result.test]

        # Fixed eval examples, keyed by patient_id, covering both val and test -- reused by
        # every evaluate(resample_landmarks=False) call across the whole training run.
        self._fixed_eval_examples = {ex["patient_id"]: ex for ex in result.val + result.test}
        # Fixed (seed-sampled) TRAINING-split landmark examples: never used for training
        # (training resamples landmarks every epoch) -- only as the evaluation pipeline's
        # IPCW censoring reference (compass.evaluation.pipeline), which must come from the
        # training split, never from the split being evaluated.
        self._fixed_train_examples = list(result.train)

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

    def fixed_examples(self, split: str) -> list[dict]:
        """The split's fixed landmark examples (scaled + masked), in split order.
        train: the seed-sampled training landmarks (IPCW reference only, see __init__);
        val / test: the same fixed landmarks evaluate(resample_landmarks=False) uses."""
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
        {"survival_nll", "total"} for TraCeR/SurvLatent ODE) -- generic over which keys exist,
        per-outcome breakdown is not available since no model's compute_loss exposes it.
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

    def _checkpoint(self, path: Path, epoch: int, rng: np.random.Generator, val_loss: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "rng_state": rng.bit_generator.state,
                "val_loss": val_loss,
            },
            path,
        )

    def resume(self, checkpoint_path: str) -> np.random.Generator:
        """Load a checkpoint (model + optimizer + scheduler + rng state) and record where to
        continue from. Call this BEFORE fit() -- fit() checks for a pending resume and
        continues from the checkpointed epoch/rng state instead of starting fresh. Returns the
        restored rng (also stored internally), in case the caller wants it directly.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint["rng_state"]

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
        checkpoint_dir/checkpoint_best.pt whenever val "total" loss improves). If
        early_stopping_patience is set, stop once val "total" loss hasn't improved for that
        many consecutive epochs. Fully reproducible given the same training_config seed AND
        the same train_patients/val_patients ordering (the seed drives the one `rng` used for
        every epoch's landmark sampling and minibatch shuffling, start to finish).

        If resume() was called first, continues from that checkpoint's epoch/rng state instead
        of starting fresh from training_config's seed at epoch 0.

        After the loop, the model's weights are RELOADED from checkpoint_best.pt (the
        best-validation epoch), so whatever happens next -- evaluation, saving -- uses the
        best weights, never the last epoch's. history["best_epoch"] / ["best_val_loss"]
        record which epoch that was.
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

        best_val_loss = float("inf")
        epochs_without_improvement = 0
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

            val_loss = val_metrics["total"]
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._checkpoint(checkpoint_dir / "checkpoint_best.pt", epoch, rng, val_loss)
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % checkpoint_every_n_epochs == 0:
                self._checkpoint(checkpoint_dir / "checkpoint_latest.pt", epoch, rng, val_loss)

            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                break

        best_path = checkpoint_dir / "checkpoint_best.pt"
        if best_path.exists():
            best = torch.load(best_path, weights_only=False)
            self.model.load_state_dict(best["model_state_dict"])
            history["best_epoch"] = int(best["epoch"])
            history["best_val_loss"] = float(best["val_loss"])
        return history
