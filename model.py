import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

try:  # optional dependency for LoRA adapters
    from peft import PeftModel  # type: ignore
except ImportError:  # pragma: no cover - peft not installed
    PeftModel = None  # type: ignore


def pick_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


class Interleaved2DRoPE(torch.nn.Module):
    def __init__(self, base_rope_module: torch.nn.Module, pair_indices_1based: Sequence[int]):
        super().__init__()
        self.config = base_rope_module.config
        self.rope_init_fn = base_rope_module.rope_init_fn
        self.attention_scaling = base_rope_module.attention_scaling
        self.register_buffer("inv_freq", base_rope_module.inv_freq, persistent=False)
        self.original_inv_freq = base_rope_module.original_inv_freq
        self.pair_indices = [i - 1 for i in pair_indices_1based]
        self.extra_pos2d: Optional[torch.Tensor] = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        if self.extra_pos2d is None:
            raise RuntimeError("extra_pos2d 未设置，请先调用 set_rope_pos2d")

        inv = self.inv_freq.to(dtype=torch.float32, device=x.device)
        pos1d = position_ids.to(dtype=torch.float32, device=x.device)
        freqs_1d = (inv[:, None] * pos1d.reshape(-1)[None, :]).T.reshape(pos1d.shape[0], pos1d.shape[1], -1)

        pos2d = self.extra_pos2d.to(dtype=torch.float32, device=x.device)
        freqs_x = (inv[:, None] * pos2d[..., 0].reshape(-1)[None, :]).T.reshape(pos1d.shape[0], pos1d.shape[1], -1)
        freqs_y = (inv[:, None] * pos2d[..., 1].reshape(-1)[None, :]).T.reshape(pos1d.shape[0], pos1d.shape[1], -1)

        F = freqs_1d.shape[-1]
        for p in self.pair_indices:
            if p < 0 or p >= F:
                raise ValueError(f"pair_indices 超出频率维度范围 [0, {F - 1}]")

        freqs_mix = freqs_1d.clone()
        for local_i, p in enumerate(self.pair_indices):
            freqs_mix[..., p] = freqs_x[..., p] if (local_i % 2 == 0) else freqs_y[..., p]

        emb = torch.cat((freqs_mix, freqs_mix), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def patch_model_with_interleaved_2d_rope(model: torch.nn.Module, pair_indices_1based: Sequence[int]) -> Interleaved2DRoPE:
    holder = _find_rotary_holder(model)
    base_rope = holder.rotary_emb
    new_rope = Interleaved2DRoPE(base_rope, pair_indices_1based)
    holder.rotary_emb = new_rope
    return new_rope


def set_rope_pos2d(model: torch.nn.Module, pos2d: torch.Tensor) -> None:
    device = next(model.parameters()).device
    holder = _find_rotary_holder(model)
    rope = holder.rotary_emb
    if not isinstance(rope, Interleaved2DRoPE):
        raise RuntimeError("模型未打补丁，请先调用 patch_model_with_interleaved_2d_rope")
    rope.extra_pos2d = pos2d.to(device)


def find_join_y_by_first_plus(main_text: str, tokenizer: AutoTokenizer) -> int:
    if "+" in main_text:
        idx = main_text.index("+")
        prefix = main_text[:idx]
        ids_pref = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
        return ids_pref.numel()
    ids_all = tokenizer(main_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    return ids_all.numel()


def pad_to_length(x: torch.Tensor, length: int, pad_id: int) -> torch.Tensor:
    if x.size(-1) >= length:
        return x[..., :length]
    pad = x.new_full((*x.shape[:-1], length - x.size(-1)), pad_id)
    return torch.cat([x, pad], dim=-1)


@dataclass
class LayoutMetadata:
    branch_ids: List[int]
    branch_lengths: List[int]
    branch_start_y: List[int]
    branch_pos1d_end: List[int]
    background_branch_id: int
    branch_pos1d_end: List[int]


@dataclass
class BatchLayout:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pos1d: torch.Tensor
    pos2d: torch.Tensor
    metadata: List[LayoutMetadata]
    pad_id: int
    time_ids: torch.Tensor


def build_columnar_causal_mask(time_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Construct a causal mask based on column-aligned time indices.

    Args:
        time_ids: Long tensor of shape (batch, seq_len) containing column indices for each token. Padding positions
            should be set to -1.
        pad_mask: Long/Bool tensor of shape (batch, seq_len) with 1 for valid tokens and 0 for padding.

    Returns:
        Float tensor of shape (batch, 1, seq_len, seq_len) where disallowed positions are filled with -inf.
    """

    device = time_ids.device
    batch, seq_len = time_ids.shape
    pad_bool = pad_mask.bool()
    time_clean = torch.where(pad_bool, time_ids, torch.full_like(time_ids, -1))

    time_i = time_clean.unsqueeze(2)
    time_j = time_clean.unsqueeze(1)

    allowed = time_j < time_i
    eye = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
    allowed = allowed | eye

    allowed = allowed & pad_bool.unsqueeze(2) & pad_bool.unsqueeze(1)

    mask = torch.full((batch, 1, seq_len, seq_len), fill_value=torch.finfo(torch.float32).min, device=device)
    mask = mask.masked_fill(allowed[:, None, :, :], 0.0)
    return mask


def build_incremental_causal_mask(time_list: List[int], device: torch.device) -> torch.Tensor:
    """Build a causal mask for a single new token appended to an existing sequence.

    Args:
        time_list: list of time indices including the newly appended token as the last element.
        device: target device for the mask tensor.
    """

    total_len = len(time_list)
    if total_len == 0:
        raise ValueError("time_list must contain at least one element")
    new_time = time_list[-1]
    mask = torch.full((1, 1, 1, total_len), fill_value=torch.finfo(torch.float32).min, device=device)
    for idx, t in enumerate(time_list):
        if t >= 0 and (t < new_time or idx == total_len - 1):
            mask[0, 0, 0, idx] = 0.0
    return mask


@dataclass
class BranchGeneration:
    branch_id: int
    token_ids: torch.Tensor
    text: str


@dataclass
class SampleGeneration:
    sequence_ids: torch.Tensor
    linear_text: str
    branches: List[BranchGeneration]


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    samples: List[SampleGeneration]

    def to_plain(self) -> List[Dict[str, Any]]:
        plain: List[Dict[str, Any]] = []
        for sample in self.samples:
            plain.append(
                {
                    "linear_text": sample.linear_text,
                    "branches": [
                        {
                            "branch_id": branch.branch_id,
                            "text": branch.text,
                            "token_ids": branch.token_ids.tolist(),
                        }
                        for branch in sample.branches
                    ],
                }
            )
        return plain


def build_flat_linear_layout(
    tokenizer: AutoTokenizer,
    samples: Sequence[Dict[str, Any]],
    device: torch.device,
    pad_to: Optional[int] = None,
) -> BatchLayout:
    tokenized: List[Dict[str, Any]] = []
    max_len = 0
    metadata: List[LayoutMetadata] = []

    for s in samples:
        main_ids = tokenizer(s["main"], add_special_tokens=False, return_tensors="pt").input_ids[0]
        branches = s.get("branches", [])
        branches_ids = [
            tokenizer(txt, add_special_tokens=False, return_tensors="pt").input_ids[0]
            for txt in branches
        ]

        total_len = main_ids.numel() + sum(x.numel() for x in branches_ids)
        tokenized.append(
            {
                "main_ids": main_ids,
                "branches_ids": branches_ids,
                "main_len": main_ids.numel(),
                "branches_lens": [x.numel() for x in branches_ids],
            }
        )
        max_len = max(max_len, total_len)

    global_T = pad_to if pad_to is not None else max_len
    pad_id = tokenizer.pad_token_id or (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)

    input_ids_batch: List[torch.Tensor] = []
    attn_batch: List[torch.Tensor] = []
    pos1d_batch: List[torch.Tensor] = []
    pos2d_batch: List[torch.Tensor] = []
    time_batch: List[torch.Tensor] = []

    for t in tokenized:
        main_tokens = t["main_ids"].to(device)
        branch_token_list = [ids.to(device) for ids in t["branches_ids"]]
        main_len = t["main_len"]

        branch_sequences: List[torch.Tensor] = []
        if main_len > 0:
            branch_sequences.append(main_tokens)
        branch_sequences.extend(branch_token_list)

        branch_ids = list(range(len(branch_sequences)))
        non_main_count = len(branch_sequences) - (1 if main_len > 0 else 0)
        max_branch_len = (
            max((seq.numel() for seq in branch_sequences[(1 if main_len > 0 else 0):]), default=0)
            if non_main_count > 0
            else 0
        )

        entries: List[Tuple[int, int, int, int]] = []
        branch_start_y: List[int] = []
        branch_pos1d_end = [-1 for _ in branch_sequences]

        for idx, (branch_id, tokens) in enumerate(zip(branch_ids, branch_sequences)):
            seq_len = tokens.numel()
            if seq_len == 0:
                start_col = 0 if idx == 0 else (main_len if main_len > 0 else 0)
                branch_start_y.append(start_col)
                continue

            if idx == 0 and main_len > 0:
                times = torch.arange(seq_len, device=device)
            else:
                if non_main_count > 0:
                    if main_len > 0:
                        base = max(main_len - max_branch_len, 0)
                    else:
                        base = 0
                    start_col = base + (max_branch_len - seq_len)
                else:
                    start_col = 0 if main_len == 0 else main_len
                times = torch.arange(seq_len, device=device) + start_col
            branch_start_y.append(int(times[0].item()))
            for order, token in enumerate(tokens):
                time_value = int(times[order].item())
                entries.append((time_value, branch_id, order, int(token.item())))

        entries.sort()
        token_count = len(entries)
        effective_len = min(token_count, global_T)
        entries_eff = entries[:effective_len]

        sorted_ids = [entry[3] for entry in entries_eff]
        sorted_branch = [entry[1] for entry in entries_eff]
        sorted_time = [entry[0] for entry in entries_eff]

        branch_lengths_eff = [0 for _ in branch_sequences]
        for pos_idx, (time_val, branch_id, _, _) in enumerate(entries_eff):
            if branch_id < len(branch_pos1d_end):
                branch_pos1d_end[branch_id] = pos_idx
                branch_lengths_eff[branch_id] += 1

        ids_tensor = (
            torch.tensor(sorted_ids, dtype=torch.long, device=device)
            if sorted_ids
            else torch.empty(0, dtype=torch.long, device=device)
        )
        seq_padded = pad_to_length(ids_tensor[None, :], global_T, pad_id)
        input_ids_batch.append(seq_padded)

        mask_row = torch.zeros(1, global_T, device=device, dtype=torch.long)
        mask_row[0, : effective_len] = 1
        attn_batch.append(mask_row)

        pos1d_row = torch.arange(global_T, device=device)[None, :]
        pos1d_batch.append(pos1d_row)

        pos2d_seq = torch.zeros(global_T, 2, device=device, dtype=torch.long)
        if effective_len > 0:
            pos2d_seq[:effective_len, 0] = torch.tensor(sorted_branch, dtype=torch.long, device=device)
            pos2d_seq[:effective_len, 1] = torch.tensor(sorted_time, dtype=torch.long, device=device)
        pos2d_batch.append(pos2d_seq[None, :, :])

        time_row = torch.full((global_T,), -1, device=device, dtype=torch.long)
        if effective_len > 0:
            time_row[:effective_len] = torch.tensor(sorted_time, dtype=torch.long, device=device)
        time_batch.append(time_row[None, :])

        branch_lengths = branch_lengths_eff
        metadata.append(
            LayoutMetadata(
                branch_ids=branch_ids,
                branch_lengths=branch_lengths,
                branch_start_y=branch_start_y,
                branch_pos1d_end=branch_pos1d_end,
                background_branch_id=0 if main_len > 0 else -1,
            )
        )

    input_ids = torch.cat(input_ids_batch, dim=0)
    attention_mask = torch.cat(attn_batch, dim=0)
    pos1d = torch.cat(pos1d_batch, dim=0)
    pos2d = torch.cat(pos2d_batch, dim=0)
    time_ids = torch.cat(time_batch, dim=0)

    return BatchLayout(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pos1d=pos1d,
        pos2d=pos2d,
        metadata=metadata,
        pad_id=pad_id,
        time_ids=time_ids,
    )


class ParallelDecoder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        pair_indices_1based: Sequence[int] = (8, 16, 24),
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model: Optional[AutoModelForCausalLM] = None,
        trust_remote_code: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        adapter_base_model: Optional[str] = None,
    ) -> None:
        if device is None or torch_dtype is None:
            picked_device, picked_dtype = pick_device_and_dtype()
            device = device or picked_device
            torch_dtype = torch_dtype or picked_dtype
        self.device = torch.device(device)
        self.dtype = torch_dtype
        self.model_name = model_name
        self.adapter_base_model = adapter_base_model

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}

        adapter_path: Optional[Path] = None
        tokenizer_source = model_name
        load_model_name = model_name
        candidate_path = Path(model_name)
        if candidate_path.is_dir() and (candidate_path / "adapter_config.json").exists():
            adapter_path = candidate_path
            if adapter_base_model is None:
                raise RuntimeError("检测到 LoRA adapter，但未提供 adapter_base_model")
            load_model_name = adapter_base_model
            tokenizer_source = adapter_base_model

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=trust_remote_code,
            **tokenizer_kwargs,
        )

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                load_model_name,
                torch_dtype=self.dtype,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            if adapter_path is not None:
                if PeftModel is None:
                    raise RuntimeError("peft 未安装，无法加载 LoRA adapter")
                model = PeftModel.from_pretrained(model, str(adapter_path))
                if hasattr(model, "merge_and_unload"):
                    model = model.merge_and_unload()
        self.model = model.to(self.device)
        self.model.eval()
        patch_model_with_interleaved_2d_rope(self.model, pair_indices_1based)

    def _slice_past(self, past_kv: Any, idx: int) -> Any:
        if past_kv is None:
            return None

        if isinstance(past_kv, tuple):
            legacy_layers = past_kv
            make_dynamic = False
        elif hasattr(past_kv, "to_legacy_cache"):
            legacy_layers = past_kv.to_legacy_cache()
            make_dynamic = True
        else:
            raise TypeError(f"不支持的 past_key_values 类型: {type(past_kv)}")

        sliced_layers = []
        for layer in legacy_layers:
            if layer is None:
                sliced_layers.append(None)
                continue
            if not (isinstance(layer, tuple) and len(layer) == 2):
                raise TypeError("期望每层为 (key, value) 元组")
            key, value = layer
            sliced_layers.append((key[idx : idx + 1].contiguous(), value[idx : idx + 1].contiguous()))

        if make_dynamic:
            return DynamicCache.from_legacy_cache(tuple(sliced_layers))
        return tuple(sliced_layers)

    def _trim_past_seq(self, past_kv: Any, target_len: int) -> Any:
        if past_kv is None or target_len is None:
            return past_kv
        if target_len < 0:
            target_len = 0

        def _trim_legacy(legacy_layers: Tuple[Any, ...]) -> Tuple[Any, ...]:
            trimmed: List[Any] = []
            for layer in legacy_layers:
                if layer is None:
                    trimmed.append(None)
                    continue
                if not (isinstance(layer, tuple) and len(layer) == 2):
                    trimmed.append(layer)
                    continue
                key, value = layer
                if key is not None and key.shape[-2] > target_len:
                    key = key[..., :target_len, :].contiguous()
                if value is not None and value.shape[-2] > target_len:
                    value = value[..., :target_len, :].contiguous()
                trimmed.append((key, value))
            return tuple(trimmed)

        if isinstance(past_kv, tuple):
            return _trim_legacy(past_kv)

        if hasattr(past_kv, "to_legacy_cache"):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = _trim_legacy(legacy)
            return DynamicCache.from_legacy_cache(trimmed_legacy)

        return past_kv

    def build_layout(self, samples: Sequence[Dict[str, Any]], pad_to: Optional[int] = None) -> BatchLayout:
        return build_flat_linear_layout(self.tokenizer, samples, device=self.device, pad_to=pad_to)

    @torch.no_grad()
    def forward(self, samples: Sequence[Dict[str, Any]], pad_to: Optional[int] = None, **forward_kwargs):
        layout = self.build_layout(samples, pad_to=pad_to)
        set_rope_pos2d(self.model, layout.pos2d)
        param_iter = iter(self.model.parameters())
        first_param = next(param_iter, None)
        param_dtype = first_param.dtype if first_param is not None else torch.float32
        causal_mask = build_columnar_causal_mask(
            layout.time_ids.to(self.device),
            layout.attention_mask.to(self.device),
        ).to(dtype=param_dtype)
        return self.model(
            input_ids=layout.input_ids,
            attention_mask=causal_mask,
            position_ids=layout.pos1d,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            **forward_kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        samples: Sequence[Dict[str, Any]],
        pad_to: Optional[int] = None,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        do_sample: bool = False,
        branch_schedule: Optional[Sequence[int]] = None,
    ) -> GenerationResult:
        layout = self.build_layout(samples, pad_to=pad_to)

        set_rope_pos2d(self.model, layout.pos2d)
        param_iter = iter(self.model.parameters())
        first_param = next(param_iter, None)
        param_dtype = first_param.dtype if first_param is not None else torch.float32
        causal_mask = build_columnar_causal_mask(
            layout.time_ids.to(self.device),
            layout.attention_mask.to(self.device),
        ).to(dtype=param_dtype)
        outputs = self.model(
            input_ids=layout.input_ids,
            attention_mask=causal_mask,
            position_ids=layout.pos1d,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        lengths_tensor = layout.attention_mask.sum(dim=-1).long()
        lengths_list = [int(l.item()) for l in lengths_tensor]

        seq_list = [layout.input_ids[b : b + 1, : lengths_list[b]].clone() for b in range(len(samples))]
        time_lists = [layout.time_ids[b, : lengths_list[b]].tolist() for b in range(len(samples))]
        past_list = [self._trim_past_seq(self._slice_past(past, b), lengths_list[b]) for b in range(len(samples))]

        def _kv_seq_len(cache: Any) -> Optional[int]:
            if cache is None:
                return None

            legacy_view: Optional[Tuple[Any, ...]] = None
            if isinstance(cache, tuple):
                legacy_view = cache
            elif hasattr(cache, "to_legacy_cache"):
                try:
                    legacy_view = cache.to_legacy_cache()
                except Exception:
                    legacy_view = None

            if legacy_view is not None:
                for layer in legacy_view:
                    if layer is None:
                        continue
                    if isinstance(layer, tuple) and layer and layer[0] is not None:
                        return layer[0].shape[-2]

            if hasattr(cache, "get_seq_length"):
                try:
                    return int(cache.get_seq_length())
                except Exception:
                    return None

            return None

        for idx in range(len(samples)):
            kv_len = _kv_seq_len(past_list[idx])
            if kv_len is None:
                continue
            if kv_len < lengths_list[idx]:
                seq_list[idx] = seq_list[idx][:, :kv_len]
                time_lists[idx] = time_lists[idx][:kv_len]
                lengths_list[idx] = kv_len

        branch_states: List[Dict[str, torch.Tensor]] = []
        branch_tokens_per_sample: List[List[List[torch.Tensor]]] = []

        for sample_idx, meta in enumerate(layout.metadata):
            branch_ids_tensor = torch.tensor(meta.branch_ids, device=self.device, dtype=layout.pos2d.dtype)
            branch_lengths_tensor = torch.tensor(meta.branch_lengths, device=self.device, dtype=layout.pos2d.dtype)
            branch_start_y_tensor = torch.tensor(meta.branch_start_y, device=self.device, dtype=layout.pos2d.dtype)
            branch_pos1d_tensor = torch.tensor(meta.branch_pos1d_end, device=self.device, dtype=layout.pos1d.dtype)
            last_valid_index = lengths_list[sample_idx] - 1 if lengths_list[sample_idx] > 0 else -1
            if last_valid_index >= 0:
                branch_pos1d_tensor = torch.clamp(branch_pos1d_tensor, max=last_valid_index)
            branch_ymax_tensor = torch.where(
                branch_lengths_tensor > 0,
                branch_start_y_tensor + branch_lengths_tensor - 1,
                branch_start_y_tensor - 1,
            )
            if time_lists[sample_idx]:
                max_time = max(time_lists[sample_idx])
                if max_time >= 0:
                    max_time_tensor = torch.full_like(branch_ymax_tensor, max_time)
                    branch_ymax_tensor = torch.minimum(branch_ymax_tensor, max_time_tensor)
                    updated_lengths = branch_ymax_tensor - branch_start_y_tensor + 1
                    branch_lengths_tensor = torch.clamp(updated_lengths, min=0)
            branch_states.append(
                {
                    "ids": branch_ids_tensor,
                    "ymax": branch_ymax_tensor,
                    "pos1d": branch_pos1d_tensor,
                    "background": torch.tensor(meta.background_branch_id, device=self.device, dtype=torch.long),
                }
            )
            branch_tokens_per_sample.append([[] for _ in meta.branch_ids])

        for _ in range(max_new_tokens):
            for sample_idx in range(len(samples)):
                ids_tensor = branch_states[sample_idx]["ids"]
                ymax_tensor = branch_states[sample_idx]["ymax"]

                if branch_schedule is None:
                    branch_order = range(len(ids_tensor))
                else:
                    order = []
                    lookup = {int(b.item()): i for i, b in enumerate(ids_tensor)}
                    for target in branch_schedule:
                        if target in lookup:
                            order.append(lookup[target])
                    if not order:
                        continue
                    branch_order = order

                pos1d_tensor = branch_states[sample_idx]["pos1d"]

                background_id = int(branch_states[sample_idx]["background"].item())

                for branch_idx in branch_order:
                    current_branch = int(ids_tensor[branch_idx].item())
                    if background_id >= 0 and current_branch == background_id:
                        continue
                    last_pos = int(pos1d_tensor[branch_idx].item())
                    if last_pos < 0:
                        continue

                    last_tokens = seq_list[sample_idx][:, last_pos : last_pos + 1]
                    pos1d_next = torch.tensor(
                        [[last_pos + 1]],
                        device=self.device,
                        dtype=layout.pos1d.dtype,
                    )

                    branch_x = ids_tensor[branch_idx].view(1, 1)
                    y_next = (ymax_tensor[branch_idx] + 1).view(1, 1)
                    pos2d_next = torch.stack([branch_x, y_next], dim=-1)
                    set_rope_pos2d(self.model, pos2d_next)

                    new_time = int(y_next.item())
                    increment_mask = build_incremental_causal_mask(
                        time_lists[sample_idx] + [new_time],
                        self.device,
                    ).to(dtype=param_dtype)

                    out = self.model(
                        input_ids=last_tokens,
                        attention_mask=increment_mask,
                        position_ids=pos1d_next,
                        use_cache=True,
                        past_key_values=past_list[sample_idx],
                        return_dict=True,
                    )

                    logits = out.logits[:, -1, :]
                    if do_sample and temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)

                    seq_list[sample_idx] = torch.cat([seq_list[sample_idx], next_token], dim=1)
                    past_list[sample_idx] = out.past_key_values
                    lengths_list[sample_idx] += 1
                    ymax_tensor[branch_idx] = ymax_tensor[branch_idx] + 1
                    branch_tokens_per_sample[sample_idx][branch_idx].append(next_token.detach())
                    time_lists[sample_idx].append(new_time)
                    pos1d_tensor[branch_idx] = torch.tensor(
                        lengths_list[sample_idx] - 1,
                        device=self.device,
                        dtype=layout.pos1d.dtype,
                    )

        pad_id = layout.pad_id
        max_len = max(seq.size(1) for seq in seq_list)
        padded_sequences: List[torch.Tensor] = []
        for seq in seq_list:
            if seq.size(1) < max_len:
                pad = seq.new_full((1, max_len - seq.size(1)), pad_id)
                seq = torch.cat([seq, pad], dim=1)
            padded_sequences.append(seq)
        gen_ids = torch.cat(padded_sequences, dim=0)

        linear_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        samples_out: List[SampleGeneration] = []
        for idx, text in enumerate(linear_texts):
            branches_out: List[BranchGeneration] = []
            ids_tensor = branch_states[idx]["ids"].detach().cpu()
            background_id = int(branch_states[idx]["background"].item())
            for branch_idx, branch_id in enumerate(ids_tensor.tolist()):
                if background_id >= 0 and branch_id == background_id:
                    continue
                tokens_seq = branch_tokens_per_sample[idx][branch_idx]
                if tokens_seq:
                    branch_tensor = torch.cat(tokens_seq, dim=1).squeeze(0)
                    decoded = self.tokenizer.decode(branch_tensor.cpu().tolist(), skip_special_tokens=True)
                else:
                    branch_tensor = torch.empty(0, dtype=torch.long)
                    decoded = ""
                branches_out.append(
                    BranchGeneration(
                        branch_id=branch_id,
                        token_ids=branch_tensor.detach().cpu(),
                        text=decoded,
                    )
                )
            samples_out.append(
                SampleGeneration(
                    sequence_ids=gen_ids[idx].detach().cpu(),
                    linear_text=text,
                    branches=branches_out,
                )
            )

        return GenerationResult(sequences=gen_ids.detach().cpu(), samples=samples_out)


__all__ = [
    "Interleaved2DRoPE",
    "ParallelDecoder",
    "pick_device_and_dtype",
    "build_flat_linear_layout",
    "set_rope_pos2d",
    "patch_model_with_interleaved_2d_rope",
    "BatchLayout",
    "GenerationResult",
    "build_columnar_causal_mask",
    "build_incremental_causal_mask",
]
def _find_rotary_holder(module: torch.nn.Module) -> torch.nn.Module:
    """Locate the submodule that actually exposes `rotary_emb`."""

    queue: deque[torch.nn.Module] = deque([module])
    visited: set[int] = set()

    candidate_attrs = (
        "language_model",
        "model",
        "base_model",
        "module",
        "wrapped_model",
        "encoder",
    )

    while queue:
        current = queue.popleft()
        ident = id(current)
        if ident in visited:
            continue
        visited.add(ident)

        if hasattr(current, "rotary_emb"):
            return current

        for attr in candidate_attrs:
            child = getattr(current, attr, None)
            if isinstance(child, torch.nn.Module):
                queue.append(child)

        get_base = getattr(current, "get_base_model", None)
        if callable(get_base):
            try:
                base = get_base()
            except Exception:  # pragma: no cover - defensive
                base = None
            if isinstance(base, torch.nn.Module):
                queue.append(base)

    raise RuntimeError("未在模型中找到 rotary_emb 模块，无法注入 2D RoPE")
