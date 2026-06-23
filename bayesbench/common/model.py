"""Model loading, shared by all BayesBench task environments.

A single ``setup_model`` shared by all tasks: it autodetects the
GPU count (SLURM env var -> CUDA_VISIBLE_DEVICES -> torch), configures the vLLM
backend, and optionally activates a LoRA adapter. Context length and the
sequence/memory knobs are parameterized so each task can request what it needs
(e.g. 16k context for the multi-turn triage/social-judgment conversations).
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

from .models import BayesBenchModel
from .models.config import InferenceConfig
from .models.pool import ModelPool


def setup_model(
    model_name: str = "qwen7b",
    max_model_len: int = 8192,
    lora_path: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    max_num_seqs: Optional[int] = None,
) -> Tuple:
    """Initialize a model and return ``(model, tokenizer)``.

    Args:
        model_name: model nickname (e.g. "qwen7b", "llama8b", "gptoss").
        max_model_len: Context window. Single-turn tasks use 8192; the
            multi-turn conversational tasks pass 16384.
        lora_path: Optional LoRA adapter directory to load and activate on top
            of the base model.
        gpu_memory_utilization: Override the autodetected default
            (0.8 for >=4 GPUs, else 0.9).
        max_num_seqs: Override the autodetected default
            (32 for >=4 GPUs, else 256).
    """
    import os
    import torch

    # Detect GPU count: SLURM env var (most reliable), then
    # CUDA_VISIBLE_DEVICES, then torch.
    if 'SLURM_GPUS_ON_NODE' in os.environ:
        tp_size = int(os.environ['SLURM_GPUS_ON_NODE'])
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        tp_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        tp_size = torch.cuda.device_count() or 1

    large_model = tp_size >= 4
    is_moe = "gptoss" in model_name
    if max_num_seqs is None:
        max_num_seqs = 32 if large_model else 256
    if gpu_memory_utilization is None:
        gpu_memory_utilization = 0.8 if large_model else 0.9
    print(f"Loading model: {model_name} (tp={tp_size}, max_model_len={max_model_len}, "
          f"max_num_seqs={max_num_seqs}, moe={is_moe})...")

    inference_config = InferenceConfig(
        backend="vllm",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=large_model or is_moe,
        enable_lora=not large_model and not is_moe,
        max_loras=4,
        max_lora_rank=64,
    )

    pool = ModelPool(inference_config=inference_config)
    model = BayesBenchModel.__new__(BayesBenchModel)
    model.pool = pool
    model._current_model = None
    model._current_adapter = None
    model.use_model(model_name)

    if lora_path is not None:
        print(f"Loading LoRA adapter from {lora_path}...")
        model.load_adapter("lora", adapter_path=lora_path)
        model._current_adapter = "lora"
        print("LoRA adapter active.")

    tokenizer = model.get_tokenizer()
    print(f"Model loaded: {model_name}\n")

    return model, tokenizer
