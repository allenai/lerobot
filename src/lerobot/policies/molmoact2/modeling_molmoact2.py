from __future__ import annotations

import time
from collections import defaultdict, deque
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoProcessor

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.pretrained import PreTrainedPolicy


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return _to_text(value[0]) if value else ""
    if isinstance(value, dict):
        for key in ("question", "task", "prompt", "instruction", "text", "content"):
            text = _to_text(value.get(key))
            if text:
                return text
        if "messages" in value:
            return _to_text(value["messages"])
        return ""
    if torch.is_tensor(value):
        if value.numel() == 0:
            return ""
        return str(value.reshape(-1)[0].item())
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    if array.size == 0:
        return ""
    return str(array.reshape(-1)[0])


def _find_nested_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for sub_value in value.values():
            found = _find_nested_value(sub_value, key)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_nested_value(item, key)
            if found is not None:
                return found
    return None


def _looks_like_single_observation(value: Any) -> bool:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return getattr(value, "ndim", 0) > 0
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return any(_looks_like_single_observation(subvalue) for subvalue in value.values())
    return False


def _batchify_single_observation(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0) if value.ndim > 0 else value.reshape(1)
        elif isinstance(value, np.ndarray):
            out[key] = np.expand_dims(value, axis=0) if value.ndim > 0 else value.reshape(1)
        elif isinstance(value, str):
            out[key] = [value]
        elif isinstance(value, dict):
            out[key] = _batchify_single_observation(value)
        else:
            out[key] = value
    return out


def _maybe_batchify_single_observation_batch(batch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(batch, dict):
        return batch
    if any(
        str(key).startswith("observation.images.")
        and (torch.is_tensor(value) or isinstance(value, np.ndarray))
        and getattr(value, "ndim", 0) == 3
        for key, value in batch.items()
    ):
        return _batchify_single_observation(batch)
    if "observation.state" in batch:
        value = batch["observation.state"]
        if (torch.is_tensor(value) or isinstance(value, np.ndarray)) and getattr(value, "ndim", 0) == 1:
            return _batchify_single_observation(batch)
    if _looks_like_single_observation(batch) and not any(
        (torch.is_tensor(value) or isinstance(value, np.ndarray)) and getattr(value, "ndim", 0) >= 4
        for value in batch.values()
    ):
        return _batchify_single_observation(batch)
    return batch


def _slice_batch_value(value: Any, idx: int, batch_size: int) -> Any:
    if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
        return value[idx]
    if isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == batch_size:
        return value[idx]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[idx]
    return value


def _infer_batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if (torch.is_tensor(value) or isinstance(value, np.ndarray)) and getattr(value, "ndim", 0) > 0:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (int, float, np.number)):
            return len(value)
    return 1


def _normalize_image(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    while arr.ndim > 3 and int(arr.shape[0]) == 1:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in {3, 4}:
        raise ValueError(f"Unsupported image shape for MolmoAct2 policy: {arr.shape}.")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype in (np.float16, np.float32, np.float64):
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _append_images(images: list[np.ndarray], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)) and value and not (
        torch.is_tensor(value[0]) or isinstance(value[0], np.ndarray)
    ):
        for item in value:
            _append_images(images, item)
        return
    images.append(_normalize_image(value))


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")


def _resolve_checkpoint_location(checkpoint_path: str) -> tuple[str, bool]:
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        raise ValueError("MolmoAct2 policy requires `checkpoint_path`.")

    local_path = Path(checkpoint_path).expanduser()
    if local_path.exists():
        return str(local_path), True
    return snapshot_download(repo_id=checkpoint_path, repo_type="model", token=_hf_token()), False


class MolmoAct2Policy(PreTrainedPolicy):
    config_class = MolmoAct2Config
    name = "molmoact2"

    def __init__(self, config: MolmoAct2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        del inputs, kwargs
        if self.config.inference_action_mode is None:
            raise ValueError(
                "MolmoAct2Policy requires `inference_action_mode` to be set "
                "explicitly to either 'continuous' or 'discrete'."
            )
        if self.config.inference_action_mode not in {"continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode={self.config.inference_action_mode!r}. "
                "Expected one of {'continuous', 'discrete'}."
            )
        if self.config.depth_mode is not None and int(self.config.depth_mode) not in {1, 2}:
            raise ValueError(f"MolmoAct2 depth_mode must be 1 or 2, got {self.config.depth_mode}.")
        self._action_queues: dict[int, deque] = defaultdict(lambda: deque())
        self._depth_caches: dict[int, Any] = {}
        self._last_depth_video_codes_by_batch: dict[int, np.ndarray] = {}
        self._last_model_inference_s = 0.0
        self._last_model_inference_calls = 0
        self._load_hf_model()

    def _load_hf_model(self) -> None:
        checkpoint_location, _ = _resolve_checkpoint_location(self.config.checkpoint_path)
        trust_remote_code = bool(getattr(self.config, "trust_remote_code", True))
        device = torch.device(self.config.device or "cpu")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_location,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            token=_hf_token(),
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_location,
            trust_remote_code=trust_remote_code,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.action_tokenizer = None
        if self.config.inference_action_mode == "discrete":
            tokenizer_name = str(self.config.discrete_action_tokenizer or "").strip()
            if not tokenizer_name:
                raise ValueError(
                    "MolmoAct2Policy with inference_action_mode='discrete' requires "
                    "`discrete_action_tokenizer` to be provided."
                )
            self.action_tokenizer = AutoProcessor.from_pretrained(
                tokenizer_name,
                trust_remote_code=trust_remote_code,
                token=_hf_token(),
            )

    def reset(self) -> None:
        self._action_queues = defaultdict(lambda: deque())
        self._depth_caches = {}
        self._last_depth_video_codes_by_batch = {}
        self._last_model_inference_s = 0.0
        self._last_model_inference_calls = 0

    def get_last_depth_video_codes(self) -> dict[int, np.ndarray]:
        return {
            int(batch_idx): np.asarray(codes, dtype=np.int64).copy()
            for batch_idx, codes in self._last_depth_video_codes_by_batch.items()
        }

    def get_optim_params(self) -> dict:
        raise NotImplementedError("MolmoAct2 policy is inference-only.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("MolmoAct2 policy is inference-only.")

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.select_action(batch, **kwargs)

    def _slice_observation_batch(self, batch: dict[str, Any], idx: int, batch_size: int) -> dict[str, Any]:
        return {
            key: _slice_batch_value(value, idx, batch_size)
            for key, value in batch.items()
        }

    @staticmethod
    def _extract_images(obs: dict[str, Any]) -> list[np.ndarray]:
        images: list[np.ndarray] = []
        for key, value in obs.items():
            if str(key).startswith("observation.images."):
                _append_images(images, value)
        if not images:
            for key in ("images", "image"):
                if key in obs:
                    _append_images(images, obs[key])
        if not images:
            raise ValueError("No image data found in observation for MolmoAct2 policy.")
        return images

    @staticmethod
    def _extract_state(obs: dict[str, Any]) -> np.ndarray:
        state = obs.get("observation.state")
        if state is None:
            state = obs.get("state")
        if state is None:
            raise ValueError("MolmoAct2 policy requires observation.state.")
        arr = np.asarray(_to_numpy(state), dtype=np.float32)
        while arr.ndim > 1 and int(arr.shape[0]) == 1:
            arr = arr[0]
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    @staticmethod
    def _extract_task(obs: dict[str, Any]) -> str:
        for key in ("task", "language_instruction", "instruction", "prompt", "question"):
            text = _to_text(obs.get(key))
            if text:
                return text
        for key in ("task", "language_instruction", "instruction", "prompt", "question"):
            text = _to_text(_find_nested_value(obs, key))
            if text:
                return text
        return ""

    @staticmethod
    def _enqueue_action_chunk(action_queue: deque, action_chunk: torch.Tensor) -> None:
        actions = action_chunk.detach()
        if actions.ndim == 3:
            if int(actions.shape[0]) != 1:
                raise ValueError(f"Expected one action batch from MolmoAct2 model, got {tuple(actions.shape)}.")
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.ndim != 2:
            raise ValueError(f"Expected action chunk shape [T, D], got {tuple(actions.shape)}.")
        for step in torch.unbind(actions, dim=0):
            action_queue.append(step)

    def _resolve_depth_flags(self) -> tuple[bool, bool]:
        enable_depth_reasoning = bool(getattr(self.config, "enable_depth_reasoning", False))
        legacy_depth_reasoning = getattr(self.config, "use_depth_reasoning", None)
        if legacy_depth_reasoning is not None:
            enable_depth_reasoning = enable_depth_reasoning or bool(legacy_depth_reasoning)

        depth_mode = getattr(self.config, "depth_mode", None)
        if depth_mode is not None:
            mode = int(depth_mode)
            if mode not in {1, 2}:
                raise ValueError(f"MolmoAct2 depth_mode must be 1 or 2, got {depth_mode}.")
            enable_adaptive_depth = mode == 2
        else:
            legacy_adaptive_depth = getattr(self.config, "use_adaptive_depth", None)
            if legacy_adaptive_depth is not None:
                enable_adaptive_depth = bool(legacy_adaptive_depth)
            else:
                enable_adaptive_depth = bool(getattr(self.config, "enable_adaptive_depth", True))

        return enable_depth_reasoning, enable_adaptive_depth

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        batch = _maybe_batchify_single_observation_batch(batch)
        batch_size = _infer_batch_size(batch)
        requested_norm_tag = (_to_text(kwargs.get("norm_tag")) or str(self.config.norm_tag or "")).strip()
        if not requested_norm_tag:
            raise ValueError("MolmoAct2 policy requires `norm_tag` for action inference.")
        requested_num_steps = kwargs.get("num_steps")
        if requested_num_steps is None:
            requested_num_steps = self.config.num_steps
        requested_n_action_steps = kwargs.get("n_action_steps")
        requested_generator = kwargs.get("generator")
        enable_depth_reasoning, enable_adaptive_depth = self._resolve_depth_flags()
        enable_cuda_graph = bool(getattr(self.config, "enable_cuda_graph", True))
        normalize_language = bool(getattr(self.config, "normalize_language", True))

        actions: list[Tensor] = []
        self._last_depth_video_codes_by_batch = {}
        inference_s = 0.0
        inference_calls = 0

        for idx in range(batch_size):
            action_queue = self._action_queues[idx]
            if not action_queue:
                obs = self._slice_observation_batch(batch, idx, batch_size)
                depth_cache = self._depth_caches.get(idx)

                start = time.perf_counter()
                with torch.inference_mode():
                    predict_kwargs = {
                        "processor": self.processor,
                        "images": self._extract_images(obs),
                        "task": self._extract_task(obs),
                        "state": self._extract_state(obs),
                        "norm_tag": requested_norm_tag,
                        "inference_action_mode": str(self.config.inference_action_mode),
                        "enable_depth_reasoning": enable_depth_reasoning,
                        "enable_adaptive_depth": enable_adaptive_depth,
                        "depth_cache": depth_cache,
                        "action_tokenizer": self.action_tokenizer,
                        "normalize_language": normalize_language,
                        "enable_cuda_graph": enable_cuda_graph,
                    }
                    if requested_num_steps is not None:
                        predict_kwargs["num_steps"] = requested_num_steps
                    if requested_n_action_steps is not None:
                        predict_kwargs["n_action_steps"] = requested_n_action_steps
                    if requested_generator is not None:
                        predict_kwargs["generator"] = requested_generator
                    output = self.model.predict_action(
                        **predict_kwargs,
                    )
                inference_s += time.perf_counter() - start
                inference_calls += 1
                if output.depth_cache is not None:
                    self._depth_caches[idx] = output.depth_cache
                if output.depth_bins is not None:
                    self._last_depth_video_codes_by_batch[idx] = (
                        output.depth_bins.detach().cpu().reshape(-1).numpy().astype(np.int64).copy()
                    )
                self._enqueue_action_chunk(action_queue, output.actions)

            if not action_queue:
                raise RuntimeError("MolmoAct2 model returned an empty action chunk.")
            action = action_queue.popleft()
            if action.ndim == 1:
                action = action.unsqueeze(0)
            actions.append(action)

        self._last_model_inference_s = float(inference_s)
        self._last_model_inference_calls = int(inference_calls)
        device = torch.device(self.config.device or "cpu")
        return torch.cat(actions, dim=0).to(device=device, dtype=torch.float32)
