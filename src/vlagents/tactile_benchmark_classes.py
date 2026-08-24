from vlagents.policies import Agent, AGENTS
from vlagents.evaluator_envs import Obs, Act
from typing import Any
import numpy as np
import logging
from copy import deepcopy


class TactileBenchmarkAgent(Agent):
    def __init__(
        self,
        policy_name: str = "act-tact-bench",
        default_checkpoint_path: str = "lerobot/pi05_base",
        device: str = "cuda:0",
        n_action_steps: int = 30,
        temporal_ensemble_coeff: float | None = None,
        rename_map: dict[str, str] | None = None,
        second_backbone_pretrained_path: str | None = None,
        subtract_background: bool | None = None,
        tactile_image_keys: tuple[str, ...] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)

        self.policy_name = policy_name
        self.device = device
        self.n_action_steps = n_action_steps
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        checkpoint_path = self.checkpoint_path or self.default_checkpoint_path
        if self.checkpoint_step is not None:
            checkpoint_path = checkpoint_path.format(checkpoint_step=self.checkpoint_step)
        self.path = checkpoint_path
        self.second_backbone_pretrained_path = second_backbone_pretrained_path
        self._subtract_background_override = subtract_background
        self._configured_tactile_image_keys = tactile_image_keys

        default_rename_map = {
            "side": "side",
            "wrist": "wrist",
            "digit_right_right": "digit_right_right",
            "digit_right_left": "digit_right_left",
        }
        self.rename_map = {**default_rename_map, **(rename_map or {})}

    def initialize(self):
        from collections import deque
        import importlib

        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        from torchvision.transforms import v2
        import tactile_pipeline.policies.act.configuration_act
        import tactile_pipeline.policies.act.modeling_act

        # from vlagents import train_xvla

        policy_config = PreTrainedConfig.from_pretrained(self.path)
        policy_class = get_policy_class(policy_config.type)
        has_second_backbone = getattr(policy_config, "second_backbone", None)
        if has_second_backbone:
            if self.second_backbone_pretrained_path is not None:
                policy_config.second_backbone["pretrained_path"] = self.second_backbone_pretrained_path
            else:
                logging.warning("Policy config has a second backbone, but no pretrained path was provided. ")

        self.policy = policy_class.from_pretrained(self.path, config=policy_config, strict=True)
        self.policy.config.n_action_steps = self.n_action_steps
        checkpoint_subtract_background = bool(getattr(self.policy.config, "subtract_background", False))
        if self._subtract_background_override is None:
            self.subtract_background = checkpoint_subtract_background
        else:
            self.subtract_background = bool(self._subtract_background_override)
            if self.subtract_background != checkpoint_subtract_background:
                logging.warning(
                    "Inference subtract_background=%s overrides checkpoint value %s. "
                    "Use this only for a legacy checkpoint whose config did not persist the setting.",
                    self.subtract_background,
                    checkpoint_subtract_background,
                )
        logging.info(
            "Loaded policy: type=%s variant=%s class=%s checkpoint=%s device=%s chunk_size=%s n_action_steps=%s "
            "temporal_ensemble_coeff=%s freeze_variant_backbone=%s "
            "subtract_background=%s image_keys=%s",
            policy_config.type,
            getattr(self.policy.config, "act_variant", "n/a"),
            self.policy.__class__.__name__,
            self.path,
            self.device,
            getattr(self.policy.config, "chunk_size", "n/a"),
            self.policy.config.n_action_steps,
            getattr(self.policy.config, "temporal_ensemble_coeff", "n/a"),
            getattr(self.policy.config, "freeze_variant_backbone", "n/a"),
            self.subtract_background,
            [
                key.removeprefix("observation.images.")
                for key in self.policy.config.input_features
                if key.startswith("observation.images.")
            ],
        )
        self.policy_input_image_keys = [
            key.removeprefix("observation.images.")
            for key in self.policy.config.input_features
            if key.startswith("observation.images")
        ]
        if self._configured_tactile_image_keys is None:
            self.tactile_image_keys = {key for key in self.policy_input_image_keys if key.startswith("digit_")}
        else:
            self.tactile_image_keys = set(self._configured_tactile_image_keys)
        unknown_tactile_keys = self.tactile_image_keys.difference(self.policy_input_image_keys)
        if unknown_tactile_keys:
            raise ValueError(
                f"tactile_image_keys must be policy image inputs; unknown keys: {sorted(unknown_tactile_keys)}"
            )
        if getattr(self.policy, "name", None) == "act":
            policy_module = importlib.import_module(self.policy.__class__.__module__)
            ACTTemporalEnsembler = getattr(policy_module, "ACTTemporalEnsembler")

            if self.temporal_ensemble_coeff is not None:
                print("Temporal ensembling will be used")
                self.policy.config.temporal_ensemble_coeff = self.temporal_ensemble_coeff
                self.policy.temporal_ensembler = ACTTemporalEnsembler(
                    self.temporal_ensemble_coeff,
                    self.policy.config.chunk_size,
                )
            elif self.policy.config.temporal_ensemble_coeff is None:
                print("No temporal ensembling will be used")
                if hasattr(self.policy, "temporal_ensembler"):
                    delattr(self.policy, "temporal_ensembler")
                self.policy._action_queue = deque([], maxlen=self.policy.config.n_action_steps)

        self._expected_image_shapes = {
            key.removeprefix("observation.images."): tuple(feature.shape)
            for key, feature in self.policy.config.input_features.items()
            if key.startswith("observation.images.")
        }
        self._camera_transforms = {
            key: v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize((height, width)),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.ToPureTensor(),
                ]
            )
            for key, (_, height, width) in self._expected_image_shapes.items()
        }
        # self.policy.config.device = self.device
        self.policy.to(self.device)
        self.policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": self.device},
            # "rename_observations_processor": {"rename_map": self.rename_map},
        }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.path,
            preprocessor_overrides=preprocessor_overrides,
        )
        self._validate_normalization_stats(
            self.preprocessor,
            NormalizerProcessorStep,
            expected_features=self.policy.config.input_features,
            processor_name="preprocessor",
        )
        self._validate_normalization_stats(
            self.postprocessor,
            UnnormalizerProcessorStep,
            expected_features=self.policy.config.output_features,
            processor_name="postprocessor",
        )

    @staticmethod
    def _validate_normalization_stats(
        processor,
        normalization_step_type,
        expected_features,
        processor_name,
    ) -> None:
        """Fail if a saved processor would silently skip configured normalization."""
        normalization_steps = [step for step in processor.steps if isinstance(step, normalization_step_type)]
        if not normalization_steps:
            raise RuntimeError(
                f"Checkpoint {processor_name} has no {normalization_step_type.__name__}; "
                "normalization statistics were not loaded."
            )

        missing = []
        for key, feature in expected_features.items():
            step = next((item for item in normalization_steps if key in item.features), None)
            if step is None:
                missing.append(key)
                continue

            normalization_mode = step.norm_map.get(feature.type)
            mode_name = getattr(normalization_mode, "value", normalization_mode)
            selected_keys = getattr(step, "normalize_observation_keys", None)
            is_selected = selected_keys is None or key in selected_keys
            if mode_name not in (None, "IDENTITY", "identity") and (not is_selected or key not in step._tensor_stats):
                missing.append(key)

        if missing:
            raise RuntimeError(
                f"Checkpoint {processor_name} is missing normalization statistics for "
                f"configured features: {sorted(missing)}."
            )

        loaded = sorted(
            key for key in expected_features if any(key in step._tensor_stats for step in normalization_steps)
        )
        logging.info("Loaded %s normalization statistics for: %s", processor_name, loaded)

    def act(self, obs: Obs) -> Act:
        import torch

        super().act(obs)

        observation = {
            "observation.state": torch.as_tensor(np.array(obs.state, copy=True)).to(torch.float32),
            "task": self.instruction,
        }

        cameras = {self.rename_map.get(key, key): img_data for key, img_data in obs.cameras.items()}
        for key in self.policy_input_image_keys:
            if key not in cameras:
                continue
            expected_shape = self._expected_image_shapes.get(key)
            assert expected_shape is not None, (
                f"Unexpected camera key: {key}. Expected keys: {list(self._expected_image_shapes)}"
            )
            image = self._camera_transforms[key](np.array(cameras[key], copy=True))
            if self.subtract_background and key in self.tactile_image_keys:
                blank_key = f"{key}_blank"
                if blank_key not in cameras:
                    raise KeyError(f"Background subtraction requires camera {blank_key!r} for tactile input {key!r}.")
                blank = self._camera_transforms[key](np.array(cameras[blank_key], copy=True))
                image = image - blank
            observation[f"observation.images.{key}"] = image
        observation = self.preprocessor(observation)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
            # action = self.policy.predict_action_chunk(observation)
            action_raw = deepcopy(action)
        action = self.postprocessor(action)

        if isinstance(action, torch.Tensor):
            action = action.detach().float().cpu().numpy()

        action = np.squeeze(action, axis=0)
        # action[-2] += np.pi/4 # dirty fix before we clean the data and train again
        # Home pose
        return Act(action=np.asarray(action, dtype=np.float32))

    def reset(self, obs: Obs, instruction: Any, **kwargs) -> dict[str, Any]:
        info = super().reset(obs, instruction, **kwargs)
        self.policy.reset()
        return info


AGENTS["tb_agent"] = TactileBenchmarkAgent
