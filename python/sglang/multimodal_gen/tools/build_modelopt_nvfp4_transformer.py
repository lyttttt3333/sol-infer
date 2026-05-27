"""Build an SGLang-loadable ModelOpt NVFP4 diffusion transformer.

This tool keeps the ModelOpt-exported NVFP4 tensors for most transformer
modules, but can replace a validated subset of numerically sensitive modules
with their original BF16 tensors from the base transformer checkpoint.

It is intended for ModelOpt NVFP4 exports where:
- the base pipeline should remain separate from the quantized transformer
- fallback BF16 modules are model-family specific
- the serialized FP4 weight byte order may already match the runtime kernel
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX_FILENAMES = [
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
]
LTX2_X0_TRANSFORMER_CLASS_NAMES = {
    "AVTransformer3DModel",
    "LTX2VideoTransformer3DModel",
}

DEFAULT_FLUX1_NVFP4_FALLBACK_PATTERNS = [
    "transformer_blocks.*.norm1.linear*",
    "transformer_blocks.*.norm1_context.linear*",
    "transformer_blocks.*.ff.net.0.proj*",
    "transformer_blocks.*.ff.net.2*",
    "transformer_blocks.*.ff_context.net.0.proj*",
    "transformer_blocks.*.ff_context.net.2*",
    "single_transformer_blocks.*.norm.linear*",
    "single_transformer_blocks.*.proj_mlp*",
]
LTX2_NVFP4_FALLBACK_BLOCK_IDS = (0, 43, 44, 45, 46, 47)
LTX2_NVFP4_FALLBACK_ATTN_NAMES = (
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)
LTX2_NVFP4_FALLBACK_ATTN_PROJECTIONS = ("to_q", "to_k", "to_v", "to_out.0")
LTX2_NVFP4_FALLBACK_FF_NAMES = ("ff", "audio_ff")
LTX2_NVFP4_FALLBACK_FF_PROJECTIONS = ("proj_in", "proj_out")
DEFAULT_LTX2_NVFP4_FALLBACK_PATTERNS = [
    "adaln_single.emb.timestep_embedder.linear_*",
    "audio_adaln_single.emb.timestep_embedder.linear_*",
    "adaln_single.linear",
    "audio_adaln_single.linear",
    "audio_caption_projection.linear_*",
    "audio_patchify_proj",
    "audio_proj_out",
    "av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_*",
    "av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_*",
    "av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_*",
    "av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_*",
    "av_ca_a2v_gate_adaln_single.linear",
    "av_ca_audio_scale_shift_adaln_single.linear",
    "av_ca_v2a_gate_adaln_single.linear",
    "av_ca_video_scale_shift_adaln_single.linear",
    "caption_projection.linear_*",
    "patchify_proj",
    "proj_out",
    *[
        f"transformer_blocks.{block_id}.{attn_name}.{projection}"
        for block_id in LTX2_NVFP4_FALLBACK_BLOCK_IDS
        for attn_name in LTX2_NVFP4_FALLBACK_ATTN_NAMES
        for projection in LTX2_NVFP4_FALLBACK_ATTN_PROJECTIONS
    ],
    *[
        f"transformer_blocks.{block_id}.{ff_name}.{projection}"
        for block_id in LTX2_NVFP4_FALLBACK_BLOCK_IDS
        for ff_name in LTX2_NVFP4_FALLBACK_FF_NAMES
        for projection in LTX2_NVFP4_FALLBACK_FF_PROJECTIONS
    ],
]

_TENSOR_MODULE_SUFFIXES = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
    ".weight",
    ".bias",
)
LTX2_RUNTIME_NAME_REPLACEMENTS = [
    (r"^model\.diffusion_model\.(.*)$", r"\1"),
    (r"^proj_in$", r"patchify_proj"),
    (r"^time_embed\.(.*)$", r"adaln_single.\1"),
    (r"^audio_proj_in$", r"audio_patchify_proj"),
    (r"^audio_time_embed\.(.*)$", r"audio_adaln_single.\1"),
    (
        r"^av_cross_attn_video_scale_shift\.(.*)$",
        r"av_ca_video_scale_shift_adaln_single.\1",
    ),
    (
        r"^av_cross_attn_audio_scale_shift\.(.*)$",
        r"av_ca_audio_scale_shift_adaln_single.\1",
    ),
    (
        r"^av_cross_attn_video_a2v_gate\.(.*)$",
        r"av_ca_a2v_gate_adaln_single.\1",
    ),
    (
        r"^av_cross_attn_audio_v2a_gate\.(.*)$",
        r"av_ca_v2a_gate_adaln_single.\1",
    ),
    (r"(.*)ff\.net\.0\.proj$", r"\1ff.proj_in"),
    (r"(.*)ff\.net\.2$", r"\1ff.proj_out"),
]


def _resolve_transformer_dir(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return str(candidate)
    transformer_dir = candidate / "transformer"
    if (transformer_dir / "config.json").is_file():
        return str(transformer_dir)
    raise FileNotFoundError(f"Could not resolve a transformer directory from: {path}")


def _resolve_checkpoint_source(path: str) -> tuple[str, str | None]:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file() and candidate.name.endswith(".safetensors"):
        return str(candidate.parent), candidate.name
    return _resolve_transformer_dir(path), None


def _find_index_file(model_dir: str) -> str | None:
    for filename in INDEX_FILENAMES:
        candidate = os.path.join(model_dir, filename)
        if os.path.isfile(candidate):
            return filename

    matches = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors.index.json")
    )
    return matches[0] if matches else None


def _load_weight_map(
    model_dir: str,
    *,
    shard_filename: str | None = None,
) -> tuple[dict[str, str], str | None]:
    if shard_filename is not None:
        shard_path = os.path.join(model_dir, shard_filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            weight_map = {key: shard_filename for key in f.keys()}
        index_filename = f"{Path(shard_filename).stem}.safetensors.index.json"
        return weight_map, index_filename

    index_filename = _find_index_file(model_dir)
    if index_filename is not None:
        with open(os.path.join(model_dir, index_filename), encoding="utf-8") as f:
            index_data = json.load(f)
        return dict(index_data["weight_map"]), index_filename

    safetensors_files = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
    )
    if len(safetensors_files) != 1:
        raise ValueError(
            f"Expected an index file or a single safetensors shard in {model_dir}, "
            f"found {len(safetensors_files)} shard(s)."
        )

    shard_name = safetensors_files[0]
    with safe_open(
        os.path.join(model_dir, shard_name), framework="pt", device="cpu"
    ) as f:
        weight_map = {key: shard_name for key in f.keys()}
    index_filename = f"{Path(shard_name).stem}.safetensors.index.json"
    return weight_map, index_filename


def _load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _load_first_shard_metadata(
    model_dir: str,
    weight_map: Mapping[str, str],
) -> dict[str, str]:
    if not weight_map:
        return {}
    first_shard = next(iter(weight_map.values()))
    with safe_open(
        os.path.join(model_dir, first_shard), framework="pt", device="cpu"
    ) as f:
        return dict(f.metadata() or {})


def _write_config(model_dir: Path, config: Mapping[str, object]) -> None:
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_non_shard_files(source_dir: str, output_dir: str) -> None:
    ignored = set(INDEX_FILENAMES)
    for entry in os.listdir(source_dir):
        if entry.endswith(".safetensors") or entry in ignored:
            continue
        source_path = os.path.join(source_dir, entry)
        output_path = os.path.join(output_dir, entry)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, output_path)


def _load_selected_tensors_by_source_name(
    model_dir: str,
    weight_map: Mapping[str, str],
    source_to_base_tensor_names: Mapping[str, str],
):
    tensors = {}
    names_by_file: dict[str, list[str]] = defaultdict(list)
    base_to_source: dict[str, list[str]] = defaultdict(list)
    for source_name, base_name in source_to_base_tensor_names.items():
        names_by_file[weight_map[base_name]].append(base_name)
        base_to_source[base_name].append(source_name)

    for filename, names in names_by_file.items():
        shard_path = os.path.join(model_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for base_name in names:
                tensor = f.get_tensor(base_name).contiguous()
                for source_name in base_to_source[base_name]:
                    tensors[source_name] = tensor
    return tensors


def _module_name_for_tensor(tensor_name: str) -> str:
    for suffix in _TENSOR_MODULE_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _tensor_suffix(tensor_name: str) -> str:
    module_name = _module_name_for_tensor(tensor_name)
    return tensor_name[len(module_name) :]


def _ltx2_runtime_module_name_variants(module_name: str) -> list[str]:
    variants = [module_name]
    for pattern, replacement in LTX2_RUNTIME_NAME_REPLACEMENTS:
        for variant in list(variants):
            mapped = re.sub(pattern, replacement, variant)
            if mapped != variant:
                variants.append(mapped)

    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _ltx2_runtime_tensor_name_variants(tensor_name: str) -> list[str]:
    suffix = _tensor_suffix(tensor_name)
    return [
        f"{module_name}{suffix}"
        for module_name in _ltx2_runtime_module_name_variants(
            _module_name_for_tensor(tensor_name)
        )
    ]


def _matches_any_pattern(module_name: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    for pattern in patterns:
        regex_str = pattern.replace(".", r"\.").replace("*", r".*")
        if re.fullmatch(regex_str, module_name):
            return True
    return False


def _matches_any_module_variant(
    module_name: str,
    patterns: Sequence[str],
    *,
    pattern_preset: str,
) -> bool:
    variants = (
        _ltx2_runtime_module_name_variants(module_name)
        if pattern_preset == "ltx2-nvfp4"
        else [module_name]
    )
    return any(_matches_any_pattern(variant, patterns) for variant in variants)


def _resolve_base_tensor_name(
    source_tensor_name: str,
    base_weight_map: Mapping[str, str],
    *,
    pattern_preset: str,
) -> str | None:
    if source_tensor_name in base_weight_map:
        return source_tensor_name
    if pattern_preset == "ltx2-nvfp4":
        for variant in _ltx2_runtime_tensor_name_variants(source_tensor_name):
            if variant in base_weight_map:
                return variant
    return None


def _is_quant_aux_tensor(tensor_name: str) -> bool:
    return tensor_name.endswith((".weight_scale_2", ".weight_scale", ".input_scale"))


def _should_keep_ltx2_transformer_key(weight_name: str) -> bool:
    if not weight_name.startswith("model.diffusion_model."):
        return False
    connector_prefixes = (
        "model.diffusion_model.audio_embeddings_connector.",
        "model.diffusion_model.video_embeddings_connector.",
    )
    return not weight_name.startswith(connector_prefixes)


def _is_ltx2_x0_export(
    *,
    source_config: Mapping[str, object] | None,
    source_metadata: Mapping[str, str],
    source_weight_map: Mapping[str, str],
) -> bool:
    if source_config is not None and source_config.get("_class_name") == "X0Model":
        return any(
            name.startswith("model.diffusion_model.") for name in source_weight_map
        )

    if not any(name.startswith("model.diffusion_model.") for name in source_weight_map):
        return False
    try:
        metadata_config = json.loads(str(source_metadata.get("config", "")))
    except json.JSONDecodeError:
        return False
    transformer_config = metadata_config.get("transformer")
    return (
        isinstance(transformer_config, dict)
        and transformer_config.get("_class_name") in LTX2_X0_TRANSFORMER_CLASS_NAMES
    )


def _build_ltx2_output_config_from_metadata(
    source_metadata: Mapping[str, str],
    quant_config: Mapping[str, object],
) -> dict[str, object]:
    try:
        metadata_config = json.loads(str(source_metadata["config"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(
            "LTX-2 X0-style NVFP4 exports must include a safetensors metadata "
            "`config` entry with a `transformer` section."
        ) from exc

    transformer_config = metadata_config.get("transformer")
    if not isinstance(transformer_config, dict):
        raise ValueError(
            "LTX-2 X0-style NVFP4 metadata `config` is missing a transformer section."
        )

    output_config = dict(transformer_config)
    output_config["_class_name"] = "LTX2VideoTransformer3DModel"
    output_config["quantization_config"] = dict(quant_config)
    return output_config


def _infer_nvfp4_group_size_from_tensors(weight, scale) -> int | None:
    weight_shape = tuple(getattr(weight, "shape", ()))
    scale_shape = tuple(getattr(scale, "shape", ()))
    if len(weight_shape) < 2:
        return None

    input_size = int(weight_shape[1]) * 2
    if input_size <= 0:
        return None

    candidate_num_groups: list[int] = []
    if len(scale_shape) >= 2:
        candidate_num_groups.append(int(scale_shape[-1]))
    elif len(scale_shape) == 1:
        scale_len = int(scale_shape[0])
        if scale_len == int(weight_shape[0]):
            candidate_num_groups.append(1)
        candidate_num_groups.append(scale_len)
    else:
        candidate_num_groups.append(1)

    for num_groups in candidate_num_groups:
        if num_groups > 0 and input_size % num_groups == 0:
            return input_size // num_groups
    return None


def _infer_nvfp4_group_size_from_weight_map(
    model_dir: str,
    weight_map: Mapping[str, str],
) -> int | None:
    names_by_file: dict[str, set[str]] = defaultdict(set)
    for name, filename in weight_map.items():
        names_by_file[filename].add(name)

    for filename, names in names_by_file.items():
        shard_path = os.path.join(model_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            scale_names = sorted(
                name for name in names if name.endswith(".weight_scale")
            )
            for scale_name in scale_names:
                module_name = scale_name[: -len(".weight_scale")]
                weight_name = f"{module_name}.weight"
                if weight_name not in names:
                    continue
                group_size = _infer_nvfp4_group_size_from_tensors(
                    f.get_tensor(weight_name),
                    f.get_tensor(scale_name),
                )
                if group_size is not None:
                    return group_size
    return None


def _preset_patterns(pattern_preset: str) -> list[str]:
    if pattern_preset == "none":
        return []
    if pattern_preset == "flux1-nvfp4":
        return list(DEFAULT_FLUX1_NVFP4_FALLBACK_PATTERNS)
    if pattern_preset == "ltx2-nvfp4":
        return list(DEFAULT_LTX2_NVFP4_FALLBACK_PATTERNS)
    raise ValueError(f"Unsupported pattern preset: {pattern_preset}")


def _updated_quant_config(
    source_config: Mapping[str, object],
    *,
    fallback_patterns: Sequence[str],
    swap_weight_nibbles: bool,
) -> dict[str, object]:
    output_config = json.loads(json.dumps(source_config))
    quant_config = output_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError("Expected a flat quantization_config dict in config.json.")
    if (
        quant_config.get("quant_method") != "modelopt"
        or "FP4" not in str(quant_config.get("quant_algo", "")).upper()
    ):
        raise ValueError(
            "This tool only supports ModelOpt diffusion NVFP4 exports "
            "(quant_method=modelopt, quant_algo=FP4/NVFP4)."
        )

    ignore_patterns = list(quant_config.get("ignore", []) or [])
    for pattern in fallback_patterns:
        if pattern not in ignore_patterns:
            ignore_patterns.append(pattern)

    quant_config["ignore"] = ignore_patterns
    quant_config.setdefault(
        "quant_type", str(quant_config.get("quant_algo", "")).upper()
    )
    quant_config["swap_weight_nibbles"] = swap_weight_nibbles
    quant_config.setdefault("weight_scale_layout", "swizzled")
    return output_config


def _updated_quant_config_from_ltx2_x0_metadata(
    *,
    source_metadata: Mapping[str, str],
    source_weight_map: Mapping[str, str],
    source_dir: str,
    fallback_patterns: Sequence[str],
    swap_weight_nibbles: bool,
) -> dict[str, object]:
    quant_config: dict[str, object] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "quant_type": "NVFP4",
        "ignore": list(dict.fromkeys(fallback_patterns)),
        "swap_weight_nibbles": swap_weight_nibbles,
        "weight_scale_layout": "swizzled",
    }

    group_size = _infer_nvfp4_group_size_from_weight_map(source_dir, source_weight_map)
    if group_size is None:
        raise ValueError(
            "Could not infer NVFP4 group_size from LTX-2 X0-style safetensors."
        )
    quant_config["group_size"] = group_size
    return _build_ltx2_output_config_from_metadata(source_metadata, quant_config)


def build_modelopt_nvfp4_transformer(
    *,
    base_transformer_dir: str,
    modelopt_hf_dir: str,
    output_dir: str,
    pattern_preset: str = "none",
    keep_bf16_patterns: Sequence[str] | None = None,
    swap_weight_nibbles: bool | None = None,
    overwrite: bool = False,
) -> dict[str, int | bool]:
    source_dir, source_shard_filename = _resolve_checkpoint_source(modelopt_hf_dir)
    base_dir, base_shard_filename = _resolve_checkpoint_source(base_transformer_dir)
    source_weight_map_all, index_filename = _load_weight_map(
        source_dir,
        shard_filename=source_shard_filename,
    )
    source_metadata = _load_first_shard_metadata(source_dir, source_weight_map_all)
    source_config = (
        None if source_shard_filename is not None else _load_config(source_dir)
    )
    is_ltx2_export = _is_ltx2_x0_export(
        source_config=source_config,
        source_metadata=source_metadata,
        source_weight_map=source_weight_map_all,
    )

    patterns = _preset_patterns(pattern_preset)
    if keep_bf16_patterns:
        patterns.extend(keep_bf16_patterns)

    resolved_swap_weight_nibbles = (
        swap_weight_nibbles if swap_weight_nibbles is not None else False
    )
    output_config = (
        _updated_quant_config_from_ltx2_x0_metadata(
            source_metadata=source_metadata,
            source_weight_map=source_weight_map_all,
            source_dir=source_dir,
            fallback_patterns=patterns,
            swap_weight_nibbles=resolved_swap_weight_nibbles,
        )
        if is_ltx2_export
        else _updated_quant_config(
            source_config or {},
            fallback_patterns=patterns,
            swap_weight_nibbles=resolved_swap_weight_nibbles,
        )
    )
    quant_config = output_config["quantization_config"]
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if source_shard_filename is None:
        _copy_non_shard_files(source_dir, str(output_path))
    _write_config(output_path, output_config)

    source_weight_map = (
        {
            name: filename
            for name, filename in source_weight_map_all.items()
            if _should_keep_ltx2_transformer_key(name)
        }
        if is_ltx2_export
        else source_weight_map_all
    )
    base_weight_map, _ = _load_weight_map(
        base_dir,
        shard_filename=base_shard_filename,
    )

    fallback_source_to_base_tensor_names = {}
    for source_name in sorted(source_weight_map):
        if not _matches_any_module_variant(
            _module_name_for_tensor(source_name),
            patterns,
            pattern_preset=pattern_preset,
        ):
            continue
        base_name = _resolve_base_tensor_name(
            source_name,
            base_weight_map,
            pattern_preset=pattern_preset,
        )
        if base_name is not None:
            fallback_source_to_base_tensor_names[source_name] = base_name

    fallback_tensors = _load_selected_tensors_by_source_name(
        base_dir,
        base_weight_map,
        fallback_source_to_base_tensor_names,
    )
    fallback_modules = {
        _module_name_for_tensor(tensor_name)
        for tensor_name in fallback_source_to_base_tensor_names
    }

    weights_by_file: dict[str, list[str]] = defaultdict(list)
    for tensor_name, filename in source_weight_map.items():
        weights_by_file[filename].append(tensor_name)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    replaced_tensor_count = 0
    removed_aux_tensor_count = 0

    for filename, tensor_names in sorted(weights_by_file.items()):
        shard_path = os.path.join(source_dir, filename)
        shard_tensors = load_file(shard_path, device="cpu")
        selected_tensor_names = set(tensor_names)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})

        metadata.setdefault("format", "pt")
        metadata["_class_name"] = str(
            output_config.get("_class_name", metadata.get("_class_name", ""))
        )
        metadata["config"] = json.dumps(output_config, sort_keys=True)
        metadata["quantization_config"] = serialized_quant_config
        metadata["_quantization_metadata"] = serialized_quant_config

        for name in list(shard_tensors.keys()):
            if name not in selected_tensor_names:
                del shard_tensors[name]
                continue
            if "_quantizer." in name:
                del shard_tensors[name]
                removed_aux_tensor_count += 1
                continue

            module_name = _module_name_for_tensor(name)
            if module_name not in fallback_modules:
                continue

            if name in fallback_tensors:
                shard_tensors[name] = fallback_tensors[name]
                replaced_tensor_count += 1
            elif _is_quant_aux_tensor(name):
                del shard_tensors[name]
                removed_aux_tensor_count += 1

        save_file(shard_tensors, os.path.join(output_path, filename), metadata=metadata)

        for name, tensor in shard_tensors.items():
            updated_weight_map[name] = filename
            total_size += tensor.element_size() * tensor.numel()

    if index_filename is None:
        raise ValueError(
            "Expected a sharded or indexed ModelOpt HF export, but no index file was found."
        )

    with open(output_path / index_filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": updated_weight_map,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    return {
        "fallback_modules": len(fallback_modules),
        "replaced_tensors": replaced_tensor_count,
        "removed_aux_tensors": removed_aux_tensor_count,
        "output_shards": len(weights_by_file),
        "swap_weight_nibbles": resolved_swap_weight_nibbles,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an SGLang-loadable ModelOpt NVFP4 diffusion transformer and "
            "optionally keep selected modules in BF16."
        )
    )
    parser.add_argument(
        "--base-transformer-dir",
        required=True,
        help=(
            "Original BF16 transformer directory, parent model directory, or "
            "single safetensors checkpoint."
        ),
    )
    parser.add_argument(
        "--modelopt-hf-dir",
        required=True,
        help=(
            "ModelOpt --hf-ckpt-dir output, transformer subdirectory, or "
            "single safetensors checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the mixed transformer checkpoint.",
    )
    parser.add_argument(
        "--pattern-preset",
        choices=["none", "flux1-nvfp4", "ltx2-nvfp4"],
        default="none",
        help="Optional model-family BF16 fallback preset.",
    )
    parser.add_argument(
        "--keep-bf16-pattern",
        action="append",
        default=[],
        help=(
            "Glob-style pattern matched against module names without trailing tensor "
            "suffixes such as .weight or .bias."
        ),
    )
    parser.add_argument(
        "--swap-weight-nibbles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether the runtime should swap packed FP4 nibbles before padding. "
            "Defaults to false."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = build_modelopt_nvfp4_transformer(
        base_transformer_dir=args.base_transformer_dir,
        modelopt_hf_dir=args.modelopt_hf_dir,
        output_dir=args.output_dir,
        pattern_preset=args.pattern_preset,
        keep_bf16_patterns=args.keep_bf16_pattern,
        swap_weight_nibbles=args.swap_weight_nibbles,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
