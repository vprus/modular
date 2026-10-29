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
"""Explicit opt-in; ordinary Qwen3 embedding model selection is unchanged."""

from dataclasses import replace

from .arch import qwen3_embedding_modulev3_arch
from .cached_model import (
    Qwen3CachedEmbeddingBatchProcessor,
    Qwen3CachedEmbeddingConfig,
    Qwen3CachedEmbeddingModel,
)

qwen3_cached_embedding_modulev3_arch = replace(
    qwen3_embedding_modulev3_arch,
    name="Qwen3ForCausalLM_PrefixCache_ModuleV3",
    pipeline_model=Qwen3CachedEmbeddingModel,
    config=Qwen3CachedEmbeddingConfig,
    batching=Qwen3CachedEmbeddingBatchProcessor,
)
