"""Orchestrates the full COMPASS preprocessing pipeline in the order required for
correctness:

1. patient_level_split() -> train/val/test patient_ids (no overlap)
2. one landmark per patient per split: training = one random valid landmark
   (sample_landmark); val / test and the training-split IPCW reference = the evaluation rule
   (default: each patient's FIRST valid landmark, see eval_landmark)
3. construct_landmark_example() per patient
4. fit_scalers() using ONLY the training split's observed values
5. apply_scalers() then apply_masking(), in that order, applied to train/val/test alike
   using the training-fit scalers
6. fold continuous + medication + demographic features into ONE combined feature tensor per
   visit, in a fixed, documented order (see build_feature_names())
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from compass.data.landmarking import construct_landmark_example, death_date_of, sample_landmark, select_landmark
from compass.data.masking import apply_masking
from compass.data.scaling import apply_scalers, fit_scalers
from compass.data.splitting import patient_level_split


def build_feature_names(
    continuous_columns: list[str],
    medication_columns: list[str],
    sex_categories: tuple,
    smoking_status_categories: list,
) -> list[str]:
    """The single, canonical feature ordering every model must agree on: continuous
    variables, then medication flags, then a binary sex indicator, then one-hot
    smoking_status. sex_categories must have exactly 2 values; the indicator is 1 iff sex
    == sex_categories[1] (e.g. ("male", "female") -> a "sex_female" column).
    """
    if len(sex_categories) != 2:
        raise ValueError(f"sex_categories must have exactly 2 values, got {sex_categories}")
    sex_column = f"sex_{sex_categories[1]}"
    smoking_columns = [f"smoking_status_{c}" for c in smoking_status_categories]
    return list(continuous_columns) + list(medication_columns) + [sex_column] + smoking_columns


def _encode_medication_and_demographic_features(
    df: pd.DataFrame,
    medication_columns: list[str],
    sex_categories: tuple,
    smoking_status_categories: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode medication flags + demographics into (values_df, mask_df), matching the
    continuous-variable convention: mask=1 where observed, 0 where missing; missing values
    are zero-filled. Medication/demographic features are only ever missing if the source
    data actually has a null there (checked explicitly below, not assumed impossible).
    """
    med_values = df[medication_columns].fillna(0.0).astype(float)
    med_mask = df[medication_columns].notna().astype(int)

    sex_column = f"sex_{sex_categories[1]}"
    sex_observed = df["sex"].notna()
    sex_indicator = (df["sex"] == sex_categories[1]).astype(float)
    sex_indicator[~sex_observed] = 0.0
    sex_values = sex_indicator.to_frame(sex_column)
    sex_mask = sex_observed.astype(int).to_frame(sex_column)

    smoking_observed = df["smoking_status"].notna()
    smoking_columns = [f"smoking_status_{c}" for c in smoking_status_categories]
    smoking_values = pd.DataFrame(
        {f"smoking_status_{c}": (df["smoking_status"] == c).astype(float) for c in smoking_status_categories},
        index=df.index,
    )
    smoking_values.loc[~smoking_observed, :] = 0.0
    smoking_mask = pd.DataFrame(
        {col: smoking_observed.astype(int) for col in smoking_columns},
        index=df.index,
    )

    values = pd.concat([med_values, sex_values, smoking_values], axis=1)
    mask = pd.concat([med_mask, sex_mask, smoking_mask], axis=1)
    return values, mask


def scale_and_mask_examples(
    examples: list[dict],
    scalers: dict,
    continuous_columns: list[str],
    medication_columns: list[str],
    sex_categories: tuple,
    smoking_status_categories: list,
    feature_names: list[str],
) -> list[dict]:
    """Apply ALREADY-FIT `scalers` (never refit here) plus masking/encoding to a list of raw
    landmark examples (dicts with an "input_visits" key, from construct_landmark_example()),
    producing the "input_values"/"input_mask" columns the rest of the pipeline (collate_
    examples, model.forward) consumes.

    Extracted from run_preprocessing (which calls this once per split, using scalers freshly
    fit from the training split) into a standalone, public function so
    compass.training.train.Trainer can reuse the EXACT same logic to build fresh landmark
    examples each training epoch, using the SAME already-fit scalers -- not a second,
    potentially-divergent scale/mask implementation. Scalers themselves are fit ONCE (from an
    initial representative landmark sample, at Trainer setup) and never refit per epoch: only
    WHICH landmark is sampled per patient changes epoch to epoch, not the scaling statistics
    themselves -- refitting scalers from a shifting subset of training data every epoch would
    introduce unwanted epoch-to-epoch drift in the input feature scale itself, on top of the
    intended landmark variety.
    """
    if not examples:
        return []
    concat = pd.concat([ex["input_visits"] for ex in examples], ignore_index=True)
    mask_source = concat[continuous_columns].notna().astype(int)
    scaled = apply_scalers(concat, mask_source, scalers, continuous_columns)
    cont_values_df, cont_mask_df = apply_masking(scaled, continuous_columns)

    other_values_df, other_mask_df = _encode_medication_and_demographic_features(
        concat, medication_columns, sex_categories, smoking_status_categories
    )

    full_values = pd.concat([cont_values_df[continuous_columns], other_values_df], axis=1)[feature_names]
    full_mask = pd.concat([cont_mask_df[continuous_columns], other_mask_df], axis=1)[feature_names]

    processed = []
    pos = 0
    for ex in examples:
        n = len(ex["input_visits"])
        processed.append(
            {
                **ex,
                "input_values": full_values.iloc[pos : pos + n].reset_index(drop=True),
                "input_mask": full_mask.iloc[pos : pos + n].reset_index(drop=True),
            }
        )
        pos += n
    return processed


@dataclass
class PreprocessedData:
    """Bundled, model-ready train/val/test data.

    train/val/test are each a list of per-patient landmark examples: dicts with
    patient_id, landmark_date, input_visits (raw), input_values (scaled/encoded +
    zero-filled, columns in feature_names order), input_mask (1=observed/0=missing, same
    column order). No outcome labels: targets come from compass.training.targets.build_targets.

    train: one random valid landmark per training patient (also what the scalers are fit on).
    val / test: the evaluation landmark rule (eval_landmark).
    train_eval: the TRAINING split at the evaluation landmark rule -- the IPCW censoring
      reference for evaluation, built with the same rule as the evaluated split.
    """

    train: list[dict]
    val: list[dict]
    test: list[dict]
    scalers: dict
    continuous_columns: list[str]
    feature_names: list[str]
    horizons: list[int]
    train_eval: list[dict]
    eval_landmark: str


def run_preprocessing(
    visits_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    continuous_columns: list[str],
    medication_columns: list[str],
    sex_categories: tuple,
    smoking_status_categories: list,
    horizons: list[int],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
    resample_each_epoch: bool = False,
    epoch: int = 0,
    split: dict | None = None,
    eval_landmark: str = "first_post_index",
) -> PreprocessedData:
    """Run the full pipeline once and return a PreprocessedData bundle.

    resample_each_epoch / epoch: at training time, calling this repeatedly across epochs
    with resample_each_epoch=True and an increasing `epoch` draws a fresh random landmark
    per training patient each call (variety across epochs). val/test landmarks are always
    derived from a fixed seed independent of `epoch`, so evaluation stays comparable across
    calls regardless of this flag.

    split: a pinned split {"train": ids, "val": ids, "test": ids} (e.g. loaded from a run's
    split.json at evaluation) -- used as-is instead of re-deriving it with patient_level_split.

    eval_landmark (configs/training.yaml: evaluation.landmark): the landmark rule for val /
    test and for train_eval (the IPCW reference). "first_post_index" (default) = each
    patient's FIRST visit on/after index_date and before death; "random" = one random valid
    visit (seeded). Training examples always use one random valid landmark.
    """
    if split is not None:
        train_ids, val_ids, test_ids = list(split["train"]), list(split["val"]), list(split["test"])
        unknown = len(set(train_ids + val_ids + test_ids) - set(outcomes_df["patient_id"]))
        if unknown:
            raise ValueError(f"pinned split has {unknown} patient ID(s) not in outcomes_df")
    else:
        train_ids, val_ids, test_ids = patient_level_split(outcomes_df, train_frac, val_frac, test_frac, seed)
    outcomes_indexed = outcomes_df.set_index("patient_id")
    visits_by_patient = {
        pid: group.sort_values("visit_date").reset_index(drop=True) for pid, group in visits_df.groupby("patient_id")
    }

    def _build_examples(patient_ids: list, landmark_seed: int, rule: str) -> list[dict]:
        rng = np.random.default_rng(landmark_seed)
        examples = []
        for pid in patient_ids:
            patient_visits = visits_by_patient[pid]
            patient_outcomes = outcomes_indexed.loc[pid]
            landmark_date = select_landmark(
                patient_visits, rng, rule,
                index_date=patient_outcomes["index_date"], death_date=death_date_of(patient_outcomes),
            )
            examples.append(construct_landmark_example(patient_visits, patient_outcomes, landmark_date))
        return examples

    train_landmark_seed = seed + epoch if resample_each_epoch else seed
    train_examples = _build_examples(train_ids, train_landmark_seed, "random")
    val_examples = _build_examples(val_ids, seed + 100_000, eval_landmark)
    test_examples = _build_examples(test_ids, seed + 200_000, eval_landmark)
    train_eval_examples = _build_examples(train_ids, seed + 300_000, eval_landmark)

    train_visits_concat = pd.concat([ex["input_visits"] for ex in train_examples], ignore_index=True)
    train_mask_df = train_visits_concat[continuous_columns].notna().astype(int)
    scalers = fit_scalers(train_visits_concat, train_mask_df, continuous_columns)

    feature_names = build_feature_names(continuous_columns, medication_columns, sex_categories, smoking_status_categories)

    return PreprocessedData(
        train=scale_and_mask_examples(train_examples, scalers, continuous_columns, medication_columns, sex_categories, smoking_status_categories, feature_names),
        val=scale_and_mask_examples(val_examples, scalers, continuous_columns, medication_columns, sex_categories, smoking_status_categories, feature_names),
        test=scale_and_mask_examples(test_examples, scalers, continuous_columns, medication_columns, sex_categories, smoking_status_categories, feature_names),
        train_eval=scale_and_mask_examples(train_eval_examples, scalers, continuous_columns, medication_columns, sex_categories, smoking_status_categories, feature_names),
        eval_landmark=eval_landmark,
        scalers=scalers,
        continuous_columns=list(continuous_columns),
        feature_names=feature_names,
        horizons=list(horizons),
    )
