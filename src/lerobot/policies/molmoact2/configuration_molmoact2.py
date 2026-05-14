from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """LeRobot eval wrapper for converted MolmoAct2 checkpoints."""

    checkpoint_path: str = ""
    num_steps: int | None = None
    inference_action_mode: str | None = None
    discrete_action_tokenizer: str | None = None
    discrete_generation_max_steps: int = 128
    enable_depth_reasoning: bool = False
    enable_adaptive_depth: bool = True
    enable_cuda_graph: bool = True
    normalize_language: bool = True
    # Compatibility aliases for older local eval YAMLs.
    use_depth_reasoning: bool | None = None
    use_adaptive_depth: bool | None = None
    depth_mode: int | None = None
    norm_tag: str = ""
    trust_remote_code: bool = True

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=[7])}
    )

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError("MolmoAct2 config is inference-only.")

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        return
