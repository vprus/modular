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
"""GPU equivalence of fused Qwen attention and causal prefix reuse."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor, default_dtype
from max.graph import DeviceRef
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged, rope_split_store_ragged
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams, PagedCacheValues
from max.pipelines.architectures.qwen3_embedding_modulev3.cached_model import (
    Qwen3AttentionWithCache,
    Qwen3CachedEmbedding,
    Qwen3CachedEmbeddingBatchProcessor,
)
from max.pipelines.architectures.qwen3_embedding_modulev3.layers.attention import (
    Qwen3AttentionNoCache,
)
from max.pipelines.architectures.qwen3_embedding_modulev3.layers.transformer import (
    Qwen3EmbeddingTransformer,
    Qwen3EmbeddingTransformerBlock,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.embedding_prefix_cache import EmbeddingPrefixCache
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.modeling.types import RequestID

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" or accelerator_count() == 0,
    reason="Requires a non-Apple GPU for paged Qwen attention",
)

rope_store = F.functional(rope_split_store_ragged)
paged_attention = F.functional(flash_attention_ragged)


class _UnfusedAttention(Qwen3AttentionNoCache):
    """The original paged graph is an independent oracle for Q/K/V fusion."""

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


Execute = Callable[[list[int], bool], npt.NDArray[np.float32]]


@dataclass
class _Executors:
    fused: Execute
    reference: Execute
    cache: EmbeddingPrefixCache
    dtype: DType


def _as_float32(value: Buffer) -> npt.NDArray[np.float32]:
    # Pooling promotes normalized embeddings to FP32 even for a BF16 model.
    # Retain the host owner until the array is copied out of the driver buffer.
    assert value.dtype == DType.float32
    host = value.to(CPU())
    return np.array(host.to_numpy(), dtype=np.float32, copy=True)


@pytest.fixture(
    scope="module", params=[DType.float32, DType.bfloat16], ids=["fp32", "bf16"]
)
def execute_embeddings(request: pytest.FixtureRequest) -> _Executors:
    """Compile one fused/original pair per dtype; reuse them for all payloads."""
    gpu, dtype = Accelerator(), request.param
    session = InferenceSession(devices=[gpu])
    device = DeviceRef.GPU()
    params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=1,
        head_dim=128,
        num_layers=2,
        page_size=128,
        devices=[device],
        enable_prefix_caching=True,
    )

    def model(
        attention_type: type[Qwen3AttentionWithCache | _UnfusedAttention],
    ) -> Qwen3CachedEmbedding:
        with F.lazy(), default_dtype(dtype):
            rope = RotaryEmbedding(
                dim=512,
                n_heads=4,
                theta=10000,
                max_seq_len=512,
                device=gpu,
                head_dim=128,
                interleaved=False,
            )
            layers = []
            for i in range(2):
                attention = attention_type(
                    rope=rope,
                    num_attention_heads=4,
                    num_key_value_heads=1,
                    hidden_size=512,
                    head_dim=128,
                )
                attention.layer_idx, attention.kv_params = i, params
                layers.append(
                    Qwen3EmbeddingTransformerBlock(
                        attention=attention,
                        mlp=MLP(
                            hidden_dim=512, feed_forward_length=1024, bias=False
                        ),
                        attention_norm=RMSNorm(512, eps=1e-6),
                        mlp_norm=RMSNorm(512, eps=1e-6),
                    )
                )
            result = Qwen3CachedEmbedding(
                Qwen3EmbeddingTransformer(
                    layers=layers,
                    norm=RMSNorm(512, eps=1e-6),
                    embedding=Embedding(256, dim=512),
                ),
                params,
            )
            result.to(gpu)
        return result

    fused, reference = model(Qwen3AttentionWithCache), model(_UnfusedAttention)
    assert {name for name, _ in fused.parameters} == {
        name for name, _ in reference.parameters
    }
    rng = np.random.default_rng(983)
    weights = {
        name: (
            # Non-unit gamma catches a change in BF16 norm rounding order.
            rng.uniform(0.5, 1.5, tuple(map(int, param.shape)))
            if "norm" in name
            else rng.standard_normal(tuple(map(int, param.shape))) * 0.02
        ).astype(np.float32)
        for name, param in fused.parameters
    }
    values = [
        rng.integers(0, 256, n, dtype=np.int64).tolist()
        for n in [1, 33, 75, 127, 129, 160, 300]
    ]
    values += [
        values[2][:32] + rng.integers(0, 256, 43, dtype=np.int64).tolist(),
        values[6][:144] + rng.integers(0, 256, 156, dtype=np.int64).tolist(),
    ]
    caches = []

    def build(module: Qwen3CachedEmbedding) -> Execute:
        compiled = module.compile(
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=params, device_refs=[device]
            ),
            weights=weights,
            auto_cast=True,
            session=session,
        )
        cache = EmbeddingPrefixCache(params, 32, session, 16)
        caches.append(cache)
        processor = object.__new__(Qwen3CachedEmbeddingBatchProcessor)
        processor.runtime = cast(
            BatchProcessorRuntime,
            SimpleNamespace(devices=[gpu], lora_manager=None),
        )

        def execute(
            indices: list[int], enabled: bool
        ) -> npt.NDArray[np.float32]:
            cache.enabled = enabled
            contexts = [
                TextContext(
                    request_id=RequestID(),
                    max_length=512,
                    tokens=TokenBuffer(np.array(values[i], np.int64)),
                )
                for i in indices
            ]
            try:
                for ctx in contexts:
                    cache.claim(ctx)
                    cache.alloc(ctx)
                inputs = processor.prepare_initial_token_inputs(
                    [contexts], cache.runtime_inputs([contexts])
                )
                outputs = compiled(*inputs.buffers)
                result = _as_float32(outputs[0].driver_tensor)
                for ctx in contexts:
                    cache.step(ctx)
                return result
            finally:
                for ctx in contexts:
                    if cache.contains(ctx):
                        cache.release(ctx)

        return execute

    return _Executors(build(fused), build(reference), caches[0], dtype)


def test_fused_cold_warm_and_divergent_prefixes(
    execute_embeddings: _Executors,
) -> None:
    """Fusion preserves the original graph, including cached RoPE and GQA."""
    fixture = execute_embeddings
    reference = fixture.reference(list(range(9)), False)
    fused_reference = fixture.fused(list(range(9)), False)
    for indices, enabled in [
        (list(range(9)), False),
        (list(range(9)), True),
        (list(reversed(range(9))), True),
        ([2, 7, 6, 8, 0, 2, 7, 6], True),
    ]:
        actual = fixture.fused(indices, enabled)
        expected = reference[indices]
        if fixture.dtype == DType.float32:
            # Different GEMM/normalization kernels need not agree bit for bit.
            # Keep a separate, tighter guard on reuse within the fused graph.
            np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-4)
            np.testing.assert_allclose(
                actual, fused_reference[indices], atol=2e-5, rtol=2e-4
            )
        else:
            np.testing.assert_allclose(actual, expected, atol=4e-3, rtol=2e-2)
        actual64, expected64 = (
            actual.astype(np.float64),
            expected.astype(np.float64),
        )
        cosines = (
            np.sum(actual64 * expected64, axis=1)
            / np.linalg.norm(actual64, axis=1)
            / np.linalg.norm(expected64, axis=1)
        )
        assert np.min(cosines) > (
            0.999999 if fixture.dtype == DType.float32 else 0.9999
        )
    assert fixture.cache.cache_tokens > 0
    assert fixture.cache.copied_pages > 0
