# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Opt-in causal embedding model with paged KV inputs and prefix reuse."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from max import tree
from max.driver import CPU, Buffer, load_devices
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor, default_dtype
from max.graph import DeviceRef
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged, rope_split_store_ragged
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParams,
    PagedCacheValues,
)
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.config import cache_dtype_for_encoding
from max.pipelines.lib import (
    GraphRole,
    GraphSpec,
    MAXModelConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithStoredKVParams,
)
from max.pipelines.lib.interfaces.batch_processor import (
    ModuleV3SingleReplicaBatchProcessor,
)
from max.pipelines.lib.interfaces.pipeline_model import PipelineModelWithKVCache
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from typing_extensions import Self

from .inputs import Qwen3EmbeddingInputs
from .layers.attention import Qwen3AttentionNoCache
from .layers.pooling import last_token_pool, normalize_embeddings
from .layers.transformer import Qwen3EmbeddingTransformer
from .model import Qwen3EmbeddingModel
from .model_config import Qwen3EmbeddingConfig

rope_store = F.functional(rope_split_store_ragged)
paged_attention = F.functional(flash_attention_ragged)


@dataclass(kw_only=True)
class Qwen3CachedEmbeddingConfig(
    Qwen3EmbeddingConfig, ArchConfigWithStoredKVParams
):
    kv_params: KVCacheParams

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        if not model_config.pool_embeddings:
            raise ValueError(
                "Prefix caching supports last-token pooled embeddings only"
            )
        if pipeline_config.lora:
            raise ValueError(
                "Use fixed model weights, or a statically merged adapter, for embedding prefix caching"
            )
        if (
            model_config.data_parallel_degree != 1
            or len(model_config.device_specs) != 1
        ):
            raise ValueError("Embedding prefix caching requires one device")
        encoding = model_config.quantization_encoding or cls.DEFAULT_ENCODING
        kv_params = cls.construct_kv_params(
            model_config.huggingface_config,
            pipeline_config,
            [
                DeviceRef.from_device(device)
                for device in load_devices(model_config.device_specs)
            ],
            model_config.kv_cache,
            cache_dtype_for_encoding(
                encoding, model_config.kv_cache.kv_cache_format
            ),
        )
        if not isinstance(kv_params, KVCacheParams):
            raise ValueError(
                "Embedding prefix cache requires a single attention cache"
            )
        if kv_params.dtype != supported_encoding_dtype(encoding):
            raise ValueError(
                "Embedding prefix cache dtype must match the model dtype"
            )
        return cls(
            pipeline_config=pipeline_config,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
        )


@dataclass
class Qwen3CachedEmbeddingInputs(Qwen3EmbeddingInputs):
    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *tree.leaves(self.kv_cache_inputs),
        )


class Qwen3CachedEmbeddingBatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, Qwen3CachedEmbeddingInputs]
):
    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> Qwen3CachedEmbeddingInputs:
        return Qwen3CachedEmbeddingInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )


class Qwen3AttentionWithCache(Qwen3AttentionNoCache):
    """Qwen Q/K normalization followed by native per-request RoPE and KV store."""

    layer_idx: int
    kv_params: KVCacheParams

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        kv_collection: PagedCacheValues | None = None,
    ) -> Tensor:
        if kv_collection is None:
            return super().forward(x, input_row_offsets)
        q = self.q_norm(
            self.q_proj(x).reshape((-1, self.n_heads, self.head_dim))
        )
        k = self.k_norm(
            self.k_proj(x).reshape((-1, self.n_kv_heads, self.head_dim))
        )
        v = self.v_proj(x)
        qkv = F.concat(
            [
                q.reshape((-1, self.q_weight_dim)),
                k.reshape((-1, self.kv_weight_dim)),
                v,
            ],
            axis=-1,
        )
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())
        q = rope_store(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=self.rope.freqs_cis.cast(qkv.dtype).to(qkv.device),
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=False,
        ).reshape((-1, self.n_heads, self.head_dim))
        h = paged_attention(
            self.kv_params,
            input=q,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        return self.o_proj(h.reshape((-1, self.q_weight_dim)))


class Qwen3CachedEmbedding(Module[..., tuple[Tensor, ...]]):
    def __init__(
        self, transformer: Qwen3EmbeddingTransformer, kv_params: KVCacheParams
    ) -> None:
        super().__init__()
        self.language_model = transformer
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *args: Tensor,
    ) -> tuple[Tensor, ...]:
        kv_inputs = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in args)
        )
        kv_collection = tree.leaves(kv_inputs, leaf=KVCacheInputsPerDevice)[0]
        model = self.language_model
        h = model.embed_tokens(tokens)
        for layer in model.layers:
            attention = cast(
                Callable[[Tensor, Tensor, PagedCacheValues], Tensor],
                layer.self_attn,
            )
            h = h + attention(
                layer.input_layernorm(h), input_row_offsets, kv_collection
            )
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return (
            normalize_embeddings(
                last_token_pool(model.norm(h), input_row_offsets)
            ),
        )


class Qwen3CachedEmbeddingModel(
    Qwen3EmbeddingModel, PipelineModelWithKVCache[TextContext, None, Any]
):
    """Separate architecture so the existing cache-free graph stays unchanged."""

    model_config_cls: ClassVar[type[Any]] = Qwen3CachedEmbeddingConfig
    batch_processor_cls = Qwen3CachedEmbeddingBatchProcessor

    def graph_specs(self, model_config: None) -> dict[str, GraphSpec]:
        weights = self._prepare_state_dict(self._checkpoint, model_config)
        params = cast(KVCacheParams, self.kv_params)
        with F.lazy(), default_dtype(self.dtype):
            base = super()._instantiate_module(model_config)
            for idx, layer in enumerate(base.language_model.layers):
                old = layer.self_attn
                attention = Qwen3AttentionWithCache(
                    rope=old.rope,
                    num_attention_heads=old.n_heads,
                    num_key_value_heads=old.n_kv_heads,
                    hidden_size=old.hidden_size,
                    head_dim=old.head_dim,
                    scale=old.scale,
                    qk_norm_eps=getattr(
                        self.huggingface_config, "rms_norm_eps", 1e-6
                    ),
                )
                attention.layer_idx = idx
                attention.kv_params = params
                layer.self_attn = attention
            module = Qwen3CachedEmbedding(base.language_model, params)
            module.to(self.devices[0])
        assert self.batch_processor is not None
        return {
            GraphRole.ENCODER: GraphSpec(
                module,
                type(module).__qualname__,
                tuple(
                    self.batch_processor.get_symbolic_inputs(
                        kv_params=self.kv_params, device_refs=self.device_refs
                    )
                ),
                weights=weights,
            )
        }

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Qwen3CachedEmbeddingInputs)
        outputs = self.model(*model_inputs.buffers)
        return ModelOutputs(logits=cast(Buffer, outputs[0].driver_tensor))
