"""Model registry: name -> (class, config path), and build_model() from a model config.
Shared by scripts/train.py and scripts/evaluate.py so a checkpoint is always loaded into a
model built exactly the way it was trained."""

from compass.models.dynamic_deephit.model import DynamicDeepHitModel
from compass.models.survlatent_ode.model import SurvLatentODEModel
from compass.models.tracer.model import TracerModel

MODEL_REGISTRY = {
    "dynamic_deephit": (DynamicDeepHitModel, "configs/model_dynamic_deephit.yaml"),
    "tracer": (TracerModel, "configs/model_tracer.yaml"),
    "survlatent_ode": (SurvLatentODEModel, "configs/model_survlatent_ode.yaml"),
}


def build_model(name: str, model_config: dict, num_features: int, horizon_years: float):
    model_cls, _ = MODEL_REGISTRY[name]
    if name == "dynamic_deephit":
        return model_cls(
            num_features=num_features,
            hidden_dim=model_config["hidden_dim"],
            num_gru_layers=model_config["num_layers"],
            cause_head_hidden_dim=model_config["cause_head_hidden_dim"],
            num_time_bins=model_config["num_time_bins"],
            bin_width_years=model_config["bin_width_years"],
            ranking_weight=model_config["ranking_weight"],
            auxiliary_weight=model_config["auxiliary_weight"],
            horizon_years=horizon_years,
        )
    if name == "tracer":
        return model_cls(
            num_features=num_features,
            feature_embed_dim=model_config["feature_embed_dim"],
            covariate_attention_heads=model_config["covariate_attention_heads"],
            hidden_dim=model_config["hidden_dim"],
            temporal_attention_heads=model_config["temporal_attention_heads"],
            num_transformer_layers=model_config["num_transformer_layers"],
            cause_head_hidden_dim=model_config["cause_head_hidden_dim"],
            num_time_bins=model_config["num_time_bins"],
            bin_width_years=model_config["bin_width_years"],
            max_staleness_years=model_config["max_staleness_years"],
            tau_init_years=model_config["tau_init_years"],
        )
    if name == "survlatent_ode":
        return model_cls(
            num_features=num_features,
            latent_dim=model_config["latent_dim"],
            ode_hidden_dim=model_config["ode_hidden_dim"],
            cause_head_hidden_dim=model_config["cause_head_hidden_dim"],
            num_time_bins=model_config["num_time_bins"],
            bin_width_years=model_config["bin_width_years"],
            solver_method=model_config["solver_method"],
        )
    raise ValueError(f"unknown model name: {name}")
