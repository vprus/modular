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

"""Pipeline for running text embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast, final

import numpy as np
import numpy.typing as npt
from max.driver import load_devices
from max.engine import InferenceSession
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.nn.kv_cache import KVCacheParams, compute_num_device_blocks
from max.nn.transformer import ReturnLogits
from max.pipelines.context import BaseContextType, TextContext
from max.pipelines.kv_cache.embedding_prefix_cache import EmbeddingPrefixCache
from max.pipelines.modeling.types import (
    EmbeddingsContext,
    EmbeddingsGenerationInputs,
    EmbeddingsGenerationOutput,
    Pipeline,
    PipelineTokenizer,
    RequestID,
    TextGenerationRequest,
)
from max.profiler import Tracer, traced

if TYPE_CHECKING:
    from .config import PipelineConfig

from max.support.algorithm import flatten2d

from .interfaces import PipelineModel
from .interfaces.pipeline_model import PipelineModelWithKVCache
from .memory_estimation import MemoryPlan

logger = logging.getLogger("max.pipelines")

EmbeddingsPipelineType = Pipeline[
    EmbeddingsGenerationInputs, EmbeddingsGenerationOutput
]
"""Base type of :class:`EmbeddingsPipeline`, a :class:`Pipeline` mapping
embeddings inputs to embeddings outputs.
"""


@final
class EmbeddingsPipeline(EmbeddingsPipelineType):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[EmbeddingsContext]],
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            BaseContextType, npt.NDArray[np.integer[Any]], TextGenerationRequest
        ],
        memory_plan: MemoryPlan,
    ) -> None:
        del tokenizer  # Unused.
        self._pipeline_config = pipeline_config
        self._max_batch_size = memory_plan.planned_max_batch_size
        self._weight_adapters = weight_adapters
        devices = load_devices(list(memory_plan.require_device_specs()))
        session = InferenceSession(devices=[*devices])
        self._pipeline_config.configure_session(session)

        # Resolve weight paths (downloads from HF if needed).
        weight_paths = self._pipeline_config.model.resolved_weight_paths()

        # Load weights
        weights = load_weights(weight_paths)
        huggingface_config = self._pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"Embeddings pipeline requires a HuggingFace config for '{self._pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            devices=devices,
            kv_cache_config=self._pipeline_config.model.kv_cache,
            weights=weights,
            adapter=self._weight_adapters.get(
                weights_format(weight_paths), None
            ),
            return_logits=ReturnLogits.ALL,
            memory_plan=memory_plan,
            max_batch_size=self._max_batch_size,
        )
        self._kv_manager: EmbeddingPrefixCache | None = None
        if isinstance(self._pipeline_model, PipelineModelWithKVCache):
            params = cast(KVCacheParams, self._pipeline_model.kv_params)
            budget = memory_plan.available_cache_memory
            if budget is None:
                raise ValueError("Cached embeddings require a KV memory budget")
            pages = compute_num_device_blocks(
                params=params,
                available_cache_memory=budget - params.bytes_per_block,
                max_batch_size=self._max_batch_size,
                max_seq_len=self._pipeline_model.max_seq_len,
                require_max_seq_len_fits=True,
            )
            self._kv_manager = EmbeddingPrefixCache(
                params=params,
                total_num_pages=pages,
                session=session,
                max_batch_size=self._max_batch_size,
            )

    @property
    def max_batch_size(self) -> int:
        """Return the maximum number of requests that can be processed in one batch."""
        return self._max_batch_size

    @traced
    def execute(
        self,
        inputs: EmbeddingsGenerationInputs,
    ) -> dict[RequestID, EmbeddingsGenerationOutput]:
        """Processes the batch and returns embeddings.

        Given a batch, executes the graph and returns the list of embedding
        outputs per request.
        """
        manager = self._kv_manager
        if manager is None:
            return self._execute(inputs)
        contexts: list[TextContext] = []
        for batch in inputs.batches:
            for ctx in batch.values():
                if not isinstance(ctx, TextContext):
                    raise ValueError(
                        "Prefix caching requires text embedding contexts"
                    )
                contexts.append(ctx)
        try:
            for ctx in contexts:
                manager.claim(ctx)
                manager.alloc(ctx)
            result = self._execute(inputs)
            # _execute copies every result to the host. KV is now complete;
            # failed forwards never publish their partially written pages.
            for ctx in contexts:
                manager.step(ctx)
            return result
        finally:
            for ctx in contexts:
                if manager.contains(ctx):
                    manager.release(ctx)

    def _execute(
        self,
        inputs: EmbeddingsGenerationInputs,
    ) -> dict[RequestID, EmbeddingsGenerationOutput]:
        tracer: Tracer = Tracer()
        replica_batches = [
            list(replica_batch.values()) for replica_batch in inputs.batches
        ]
        # Flatten our batch for consistent indexing.
        context_batch = flatten2d(replica_batches)

        tracer.next("prepare_initial_token_inputs")
        # Prepare inputs for the first token in multistep execution.
        model_inputs = self._pipeline_model.prepare_initial_token_inputs(
            replica_batches=replica_batches,
            kv_cache_inputs=self._kv_manager.runtime_inputs(
                cast(list[list[TextContext]], replica_batches)
            )
            if self._kv_manager is not None
            else None,
        )

        tracer.next("execute")
        model_outputs = self._pipeline_model.execute(model_inputs)

        assert model_outputs.logits
        # Do the copy to host for each token generated.
        tracer.next("logits.to(CPU())")
        batch_embeddings = model_outputs.logits.to_numpy()

        # Prepare the response.
        res: dict[RequestID, EmbeddingsGenerationOutput] = {}
        tracer.push("prepare_response")
        for batch_index, request_id in enumerate(inputs.batch.keys()):
            request_embeddings = batch_embeddings[batch_index]
            if not self._pipeline_config.model.pool_embeddings:
                # Remove padded tokens from embeddings
                request_embeddings = request_embeddings[
                    : context_batch[batch_index].tokens.active_length, :
                ]
            res[request_id] = EmbeddingsGenerationOutput(request_embeddings)
        return res

    def release(self, request_id: RequestID) -> None:
        """Releases resources for the request (no-op for embeddings)."""
        # Nothing to release.
