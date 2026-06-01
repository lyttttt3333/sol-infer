from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.configs.models.adapter.ltx_2_connector import (
    LTX2ConnectorConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    filter_duplicate_safetensors_files,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class AdapterLoader(ComponentLoader):
    """Loader for small adapter-style modules (e.g., LTX-2 connectors).

    This loader intentionally avoids FSDP sharding and just:
    1) Instantiates the module from `config.json`.
    2) Loads one or more safetensors shards into the module state_dict.
    """

    component_names = ["connectors"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        config = get_diffusers_component_config(component_path=component_model_path)

        cls_name = config.pop("_class_name", None)
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        config.pop("_diffusers_version", None)
        config.pop("_name_or_path", None)

        server_args.model_paths["connectors"] = component_model_path

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        target_device = get_local_torch_device()
        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        with set_default_torch_dtype(default_dtype), skip_init_modules():
            connector_cfg = LTX2ConnectorConfig()
            connector_cfg.update_model_arch(config)
            model = model_cls(connector_cfg).to(
                device=target_device, dtype=default_dtype
            )

        safetensors_list = _list_safetensors_files(component_model_path)
        safetensors_list = filter_duplicate_safetensors_files(
            safetensors_list,
            component_model_path,
            "diffusion_pytorch_model.safetensors.index.json",
        )
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")

        loaded = {}
        for safetensors_path in safetensors_list:
            loaded.update(safetensors_load_file(safetensors_path))
        load_result = model.load_state_dict(loaded, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            missing = list(load_result.missing_keys)
            unexpected = list(load_result.unexpected_keys)
            logger.warning(
                "Loaded adapter %s with %d missing keys=%s and "
                "%d unexpected keys=%s",
                component_model_path,
                len(missing),
                missing[:20],
                len(unexpected),
                unexpected[:20],
            )

        return model
