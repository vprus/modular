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
"""Stage the fused Qwen embedding attention without physical GPU hardware."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

import pytest
from max import tree
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor, default_device, default_dtype
from max.graph import DeviceRef, TensorType
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    PagedCacheValues,
)
from max.pipelines.architectures.qwen3_embedding_modulev3.cached_model import (
    Qwen3AttentionWithCache,
)


class _AttentionGraph(Module[..., Tensor]):
    def __init__(
        self, attention: Qwen3AttentionWithCache, params: MHAKVCacheParams
    ) -> None:
        self.attention = attention
        self.params = params

    def forward(self, x: Tensor, offsets: Tensor, *args: Tensor) -> Tensor:
        inputs = self.params.unflatten_kv_inputs(
            iter(arg._graph_value for arg in args)
        )
        cache = tree.leaves(inputs, leaf=KVCacheInputsPerDevice)[0]
        attention = cast(
            Callable[[Tensor, Tensor, PagedCacheValues], Tensor], self.attention
        )
        return attention(x, offsets, cache)


@pytest.fixture(
    scope="module", params=[DType.float32, DType.bfloat16], ids=["fp32", "bf16"]
)
def staged_attention(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[str, set[str]]]:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    try:
        gpu, device, dtype = Accelerator(), DeviceRef.GPU(), request.param
        params = MHAKVCacheParams(
            dtype=dtype,
            n_kv_heads=1,
            head_dim=128,
            num_layers=1,
            page_size=128,
            devices=[device],
            enable_prefix_caching=True,
        )
        with F.lazy(), default_device(gpu), default_dtype(dtype):
            rope = RotaryEmbedding(
                dim=512,
                n_heads=4,
                theta=10000,
                max_seq_len=512,
                device=gpu,
                head_dim=128,
                interleaved=False,
            )
            attention = Qwen3AttentionWithCache(
                rope=rope,
                num_attention_heads=4,
                num_key_value_heads=1,
                hidden_size=512,
                head_dim=128,
            )
            attention.layer_idx = 0
            attention.kv_params = params
            module = _AttentionGraph(attention, params)
            staged, registry = module.stage(
                TensorType(dtype, ["tokens", 512], device),
                TensorType(DType.uint32, ["rows"], device),
                *params.flattened_kv_inputs(),
            )
        yield str(staged), set(registry)
    finally:
        set_virtual_device_count(0)


def test_native_checkpoint_names_are_preserved(
    staged_attention: tuple[str, set[str]],
) -> None:
    _, names = staged_attention
    assert names == {
        "attention.q_proj.weight",
        "attention.k_proj.weight",
        "attention.v_proj.weight",
        "attention.o_proj.weight",
        "attention.q_norm.weight",
        "attention.k_norm.weight",
    }


def test_qkv_projection_is_combined(
    staged_attention: tuple[str, set[str]],
) -> None:
    text, _ = staged_attention
    # One Q/K/V projection and one output projection; preserve the original
    # per-head norm and RoPE/store operators rather than changing their math.
    assert text.count("rmo.matmul") == 2
    assert text.count("mo.reduce.rms_norm") == 2
    assert text.count("mo.rope_split_store.ragged.paged") == 1
    assert "mo.fused_qkv_matmul" not in text
    assert "interleaved = false" in text
