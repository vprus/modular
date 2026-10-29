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
"""Prefix identity, page ownership, eviction, and failed-forward publication."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import CPU, Buffer, batch_inplace_copy
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.embedding_prefix_cache import EmbeddingPrefixCache
from max.pipelines.lib.embeddings_pipeline import EmbeddingsPipeline
from max.pipelines.modeling.types import RequestID


@pytest.fixture
def cache() -> EmbeddingPrefixCache:
    return EmbeddingPrefixCache(
        params=MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=1,
            head_dim=128,
            num_layers=1,
            page_size=128,
            devices=[DeviceRef.CPU()],
            enable_prefix_caching=True,
        ),
        total_num_pages=16,
        session=InferenceSession(devices=[CPU()]),
        max_batch_size=4,
    )


def context(values: list[int], salt: str | None = None) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=1024,
        tokens=TokenBuffer(np.array(values, dtype=np.int64)),
        cache_salt=salt,
    )


def admit(
    cache: EmbeddingPrefixCache, values: list[int], salt: str | None = None
) -> TextContext:
    ctx = context(values, salt)
    cache.claim(ctx)
    cache.alloc(ctx)
    cache._flush_copies()
    return ctx


def finish(cache: EmbeddingPrefixCache, ctx: TextContext) -> None:
    cache.step(ctx)
    cache.release(ctx)


def test_short_prefix_and_final_token(cache: EmbeddingPrefixCache) -> None:
    values = list(range(33))
    cold = admit(cache, values)
    assert cold.tokens.active_length == 33
    finish(cache, cold)
    warm = admit(cache, values)
    assert warm.cached_prefix_length == 32
    assert warm.tokens.active_length == 1
    finish(cache, warm)
    assert cache.copied_pages == 1
    assert cache._pool.num_free_blocks == 15


def test_divergent_suffix_is_private(cache: EmbeddingPrefixCache) -> None:
    first = admit(cache, list(range(75)))
    src_id = cache._block_manager.req_to_blocks[first.request_id][0].bid
    buffer = next(iter(cache._kv_buffers[0].to_memory().values())).buffers[0]
    page = buffer[src_id, :]
    batch_inplace_copy(
        [page], [Buffer.from_numpy(np.full(tuple(page.shape), 7, np.uint8))]
    )
    finish(cache, first)
    left = admit(cache, list(range(29)) + [900] * 40)
    right = admit(cache, list(range(29)) + [901] * 40)
    assert left.cached_prefix_length == right.cached_prefix_length == 16
    left_id = cache._block_manager.req_to_blocks[left.request_id][0].bid
    right_id = cache._block_manager.req_to_blocks[right.request_id][0].bid
    assert len({left_id, right_id, src_id}) == 3
    np.testing.assert_array_equal(
        buffer[left_id, :].to_numpy(), page.to_numpy()
    )
    # A write to one request cannot corrupt either the snapshot or its sibling.
    batch_inplace_copy(
        [buffer[left_id, :]],
        [Buffer.from_numpy(np.full(tuple(page.shape), 9, np.uint8))],
    )
    assert np.all(page.to_numpy() == 7)
    assert np.all(buffer[right_id, :].to_numpy() == 7)
    cache.release(left)
    cache.release(right)


def test_full_pages_share_partial_page_copies(
    cache: EmbeddingPrefixCache,
) -> None:
    values = list(range(300))
    cold = admit(cache, values)
    old_ids = [
        block.bid
        for block in cache._block_manager.req_to_blocks[cold.request_id]
    ]
    finish(cache, cold)
    warm = admit(cache, values)
    new_ids = [
        block.bid
        for block in cache._block_manager.req_to_blocks[warm.request_id]
    ]
    assert warm.cached_prefix_length == 288
    assert new_ids[:2] == old_ids[:2]
    assert new_ids[2] != old_ids[2]
    cache.release(warm)


def test_salt_isolation(cache: EmbeddingPrefixCache) -> None:
    values = list(range(75))
    finish(cache, admit(cache, values, "a"))
    other = admit(cache, values, "b")
    assert other.cached_prefix_length == 0
    cache.release(other)
    same = admit(cache, values, "a")
    assert same.cached_prefix_length == 64
    cache.release(same)


def test_failed_forward_not_published(cache: EmbeddingPrefixCache) -> None:
    values = list(range(75))
    failed = admit(cache, values)
    cache.release(failed)
    retry = admit(cache, values)
    assert retry.cached_prefix_length == 0
    cache.release(retry)
    assert cache._pool.num_free_blocks == 16
    assert not cache._pending
    assert not cache._copy_pins


def test_eviction_and_reset_release_references(
    cache: EmbeddingPrefixCache,
) -> None:
    cache._max_cached_pages = 2
    for i in range(10):
        finish(cache, admit(cache, [i] * 75))
    assert cache._cached_pages == 2
    assert cache._pool.num_free_blocks == 14
    old = admit(cache, [0] * 75)
    assert old.cached_prefix_length == 0
    cache.release(old)
    active = admit(cache, [9] * 75)
    assert active.cached_prefix_length == 64
    cache.reset_prefix_cache()
    assert not cache._index
    cache.release(active)
    assert cache._pool.num_free_blocks == 16


def test_shorter_than_match_unit(cache: EmbeddingPrefixCache) -> None:
    for _ in range(2):
        ctx = admit(cache, list(range(16)))
        assert ctx.cached_prefix_length == 0
        finish(cache, ctx)
    assert not cache._entries


def test_pipeline_failure_releases_batch(cache: EmbeddingPrefixCache) -> None:
    pipeline = object.__new__(EmbeddingsPipeline)
    pipeline._kv_manager = cache
    pipeline._execute = MagicMock(
        side_effect=RuntimeError("injected GPU failure")
    )
    contexts = [context(list(range(75))), context([900] * 75)]
    inputs = MagicMock(batches=[{ctx.request_id: ctx for ctx in contexts}])
    with pytest.raises(RuntimeError, match="injected GPU failure"):
        pipeline.execute(inputs)
    assert cache._pool.num_free_blocks == 16
    assert not cache._entries
    assert not cache._block_manager.req_to_replica


def test_partial_copies_submit_once_per_batch(
    cache: EmbeddingPrefixCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    from max.pipelines.kv_cache import embedding_prefix_cache as module

    finish(cache, admit(cache, list(range(75))))
    original = module.batch_inplace_copy
    copies = MagicMock(side_effect=original)
    monkeypatch.setattr(module, "batch_inplace_copy", copies)
    contexts = [context(list(range(29)) + [i] * 40) for i in [900, 901, 902]]
    for ctx in contexts:
        cache.claim(ctx)
        cache.alloc(ctx)
    assert copies.call_count == 0
    cache._flush_copies()
    assert copies.call_count == 1
    assert len(copies.call_args.args[0]) == 3
    for ctx in contexts:
        cache.release(ctx)


def test_admission_failure_releases_shared_and_copy_pins(
    cache: EmbeddingPrefixCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    from max.pipelines.kv_cache.paged_kv_cache.block_utils import (
        InsufficientBlocksError,
    )

    finish(cache, admit(cache, list(range(300))))
    before = cache._pool.num_free_blocks
    pipeline = object.__new__(EmbeddingsPipeline)
    pipeline._kv_manager = cache
    pipeline._execute = MagicMock()
    monkeypatch.setattr(
        cache,
        "_reserve",
        MagicMock(side_effect=InsufficientBlocksError("injected")),
    )
    ctx = context(list(range(300)))
    with pytest.raises(InsufficientBlocksError):
        pipeline.execute(MagicMock(batches=[{ctx.request_id: ctx}]))
    pipeline._execute.assert_not_called()
    assert cache._pool.num_free_blocks == before
    assert not cache._pending
    assert not cache._copy_pins
    assert not cache._pending_copies
    assert not cache._block_manager.req_to_replica


@pytest.mark.parametrize("page_size", [16, 64, 192])
def test_rejects_unsupported_gpu_page_size(page_size: int) -> None:
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        page_size=page_size,
        enable_prefix_caching=True,
    )
    with pytest.raises(ValueError, match="multiple of 128"):
        EmbeddingPrefixCache(params, 16, InferenceSession(devices=[CPU()]), 4)
