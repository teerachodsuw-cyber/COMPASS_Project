"""Orchestrates the full COMPASS preprocessing pipeline in the order required for
correctness:

1. patient_level_split() -> train/val/test patient_ids (no overlap)
2. sample_landmark() per patient per split
3. construct_landmark_example() per patient
4. fit_scalers() using ONLY the training split's observed values
5. apply_scalers() then apply_masking(), in that order, applied to train/val/test alike
   using the training-fit scalers
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from compass.data.landmarking import construct_landmark_example, sample_landmark
from compass.data.masking import apply_masking
from compass.data.scaling import apply_scalers, fit_scalers
from compass.data.splitting import patient_level_split


@dataclass
class PreprocessedData:
    """Bundled, model-ready train/val/test data.

    train/val/test are each a list of per-patient landmark examples: dicts with
    patient_id, landmark_date, input_visits (raw), input_values (scaled + zero-filled),
    input_mask (1=observed/0=missing), and outcome_labels ({outcome: {horizon: 0/1}}).
    """

    train: list[dict]
    val: list[dict]
    test: list[dict]
    scalers: dict
    continuous_columns: list[str]
    horizons: list[int]


def run_preprocessing(
    visits_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    continuous_columns: list[str],
    horizons: list[int],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
    resample_each_epoch: bool = False,
    epoch: int = 0,
) -> PreprocessedData:
    """Run the full pipeline once and return a PreprocessedData bundle.

    resample_each_epoch / epoch: at training time, calling this repeatedly across epochs
    with resample_each_epoch=True and an increasing `epoch` draws a fresh random landmark
    per training patient each call (variety across epochs). val/test landmarks are always
    derived from a fixed seed independent of `epoch`, so evaluation stays comparable across
    calls regardless of this flag.
    """
    train_ids, val_ids, test_ids = patient_level_split(outcomes_df, train_frac, val_frac, test_frac, seed)
    outcomes_indexed = outcomes_df.set_index("patient_id")
    visits_by_patient = {
        pid: group.sort_values("visit_date").reset_index(drop=True) for pid, group in visits_df.groupby("patient_id")
    }

    def _build_examples(patient_ids: list, landmark_seed: int) -> list[dict]:
        rng = np.random.default_rng(landmark_seed)
        examples = []
        for pid in patient_ids:
            patient_visits = visits_by_patient[pid]
            landmark_date = sample_landmark(patient_visits, rng)
            patient_outcomes = outcomes_indexed.loc[pid]
            examples.append(construct_landmark_example(patient_visits, patient_outcomes, landmark_date, horizons))
        return examples

    train_landmark_seed = seed + epoch if resample_each_epoch else seed
    train_examples = _build_examples(train_ids, train_landmark_seed)
    val_examples = _build_examples(val_ids, seed + 100_000)
    test_examples = _build_examples(test_ids, seed + 200_000)

    train_visits_concat = pd.concat([ex["input_visits"] for ex in train_examples], ignore_index=True)
    train_mask_df = train_visits_concat[continuous_columns].notna().astype(int)
    scalers = fit_scalers(train_visits_concat, train_mask_df, continuous_columns)

    def _scale_and_mask(examples: list[dict]) -> list[dict]:
        if not examples:
            return []
        concat = pd.concat([ex["input_visits"] for ex in examples], ignore_index=True)
        mask_source = concat[continuous_columns].notna().astype(int)
        scaled = apply_scalers(concat, mask_source, scalers, continuous_columns)
        values_df, mask_df = apply_masking(scaled, continuous_columns)

        processed = []
        pos = 0
        for ex in examples:
            n = len(ex["input_visits"])
            processed.append(
                {
                    **ex,
                    "input_values": values_df.iloc[pos : pos + n].reset_index(drop=True),
                    "input_mask": mask_df.iloc[pos : pos + n].reset_index(drop=True),
                }
            )
            pos += n
        return processed

    return PreprocessedData(
        train=_scale_and_mask(train_examples),
        val=_scale_and_mask(val_examples),
        test=_scale_and_mask(test_examples),
        scalers=scalers,
        continuous_columns=list(continuous_columns),
        horizons=list(horizons),
    )
