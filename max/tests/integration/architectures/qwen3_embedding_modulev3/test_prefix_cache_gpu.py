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
"""Numerical GPU coverage for causal Qwen3 embedding prefix reuse."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.qwen3_embedding_modulev3.cached_model import (
    Qwen3AttentionWithCache,
    Qwen3CachedEmbedding,
    Qwen3CachedEmbeddingBatchProcessor,
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


@pytest.fixture(scope="module")
def execute_embeddings() -> tuple[
    Callable[[list[int], bool], npt.NDArray[np.float32]], EmbeddingPrefixCache
]:
    """Compile one deterministic two-layer GQA model for every payload case."""
    gpu = Accelerator()
    session = InferenceSession(devices=[gpu])
    device = DeviceRef.GPU()
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=128,
        num_layers=2,
        page_size=128,
        devices=[device],
        enable_prefix_caching=True,
    )
    with F.lazy(), default_dtype(DType.float32):
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
            attention = Qwen3AttentionWithCache(
                rope=rope,
                num_attention_heads=4,
                num_key_value_heads=1,
                hidden_size=512,
                head_dim=128,
            )
            attention.layer_idx = i
            attention.kv_params = params
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
        module = Qwen3CachedEmbedding(
            Qwen3EmbeddingTransformer(
                layers=layers,
                norm=RMSNorm(512, eps=1e-6),
                embedding=Embedding(256, dim=512),
            ),
            params,
        )
        module.to(gpu)
    rng = np.random.default_rng(983)
    weights = {
        name: (
            np.ones(tuple(map(int, param.shape)), np.float32)
            if "norm" in name
            else (
                rng.standard_normal(tuple(map(int, param.shape))) * 0.02
            ).astype(np.float32)
        )
        for name, param in module.parameters
    }
    compiled = module.compile(
        *modulev3_ragged_kv_symbolic_inputs(
            kv_params=params, device_refs=[device]
        ),
        weights=weights,
        session=session,
    )
    cache = EmbeddingPrefixCache(params, 32, session, 16)
    processor = object.__new__(Qwen3CachedEmbeddingBatchProcessor)
    processor.runtime = cast(
        BatchProcessorRuntime, SimpleNamespace(devices=[gpu], lora_manager=None)
    )
    values = [
        rng.integers(0, 256, n, dtype=np.int64).tolist()
        for n in [33, 75, 127, 129, 160, 300]
    ]
    values += [
        values[1][:32] + rng.integers(0, 256, 43, dtype=np.int64).tolist(),
        values[5][:144] + rng.integers(0, 256, 156, dtype=np.int64).tolist(),
    ]

    def execute(indices: list[int], enabled: bool) -> npt.NDArray[np.float32]:
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
            result = compiled(*inputs.buffers)[0].driver_tensor.to_numpy()
            for ctx in contexts:
                cache.step(ctx)
            return np.asarray(result, dtype=np.float32)
        finally:
            for ctx in contexts:
                if cache.contains(ctx):
                    cache.release(ctx)

    return execute, cache


def test_fp32_cold_warm_and_divergent_prefixes(
    execute_embeddings: tuple[
        Callable[[list[int], bool], npt.NDArray[np.float32]],
        EmbeddingPrefixCache,
    ],
) -> None:
    """Reused K/V preserves embeddings across token/page boundaries and batches."""
    execute, cache = execute_embeddings
    reference = execute(list(range(8)), False)
    for indices in [
        list(range(8)),
        list(reversed(range(8))),
        [1, 6, 5, 7, 0, 1, 6, 5],
    ]:
        actual = execute(indices, True)
        np.testing.assert_allclose(
            actual, reference[indices], atol=2e-5, rtol=2e-4
        )
    assert cache.cache_tokens > 0
    assert cache.copied_pages > 0
