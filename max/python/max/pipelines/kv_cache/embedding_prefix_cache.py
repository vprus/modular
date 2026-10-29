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
"""Short-prefix reuse for single-pass, causal, last-token embeddings.

Cache lookup has 16-token granularity independently of the GPU page size.
Complete pages are immutable and shared. A partial last page is copied before
suffix execution, so two requests with a common prefix cannot overwrite one
another. No output embeddings are cached; at least the final token is executed.

This intentionally supports one local, full-attention cache on one device.
Generation schedulers, tiered caches, and multimodal contexts use their existing
managers. The embedding pipeline owns each claim for exactly one synchronous
forward, including its device-to-host result copy.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from max.driver import Buffer, batch_inplace_copy
from max.engine import InferenceSession
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.pipelines.context import TextContext
from max.pipelines.modeling.types import RequestID

from .kv_connector import KVTransfer
from .paged_kv_cache._common import compute_block_hashes
from .paged_kv_cache.block_pool import BlockPool
from .paged_kv_cache.block_utils import InsufficientBlocksError, KVCacheBlock
from .paged_kv_cache.cache_manager import PagedKVCacheManager

logger = logging.getLogger("max.pipelines")


@dataclass
class _Prefix:
    hashes: tuple[bytes, ...]
    blocks: tuple[KVCacheBlock, ...]


class EmbeddingPrefixCache(PagedKVCacheManager):
    """A bounded LRU of immutable prompt KV snapshots with partial-page COW."""

    MATCH_TOKENS = 16

    def __init__(
        self,
        params: KVCacheParams,
        total_num_pages: int,
        session: InferenceSession,
        max_batch_size: int,
    ) -> None:
        if len(params.devices) != 1 or params.data_parallel_degree != 1:
            raise ValueError("Embedding prefix caching requires one device")
        if params.page_size < 128 or params.page_size % 128:
            raise ValueError(
                "Embedding KV pages must be a multiple of 128 tokens; "
                "prefix matching independently uses 16-token chunks"
            )
        self.enabled = params.enable_prefix_caching
        # This manager owns prefix identity and COW, rather than the ordinary
        # manager's full-page hash table. All allocation/staging stays native.
        super().__init__(
            params=replace(params, enable_prefix_caching=False),
            total_num_pages=total_num_pages,
            session=session,
            max_batch_size=max_batch_size,
        )
        self._entries: OrderedDict[int, _Prefix] = OrderedDict()
        self._index: dict[bytes, int] = {}
        self._pending: dict[RequestID, tuple[bytes, ...]] = {}
        self._copy_pins: dict[RequestID, list[KVCacheBlock]] = {}
        self._pending_copies: dict[RequestID, tuple[int, int]] = {}
        self._next_entry = 0
        self._cached_pages = 0
        # Keep half the pool available for active batches. Admission can evict
        # additional snapshots when a large batch needs the other half too.
        self._max_cached_pages = total_num_pages // 2
        self.requests = 0
        self.cache_tokens = 0
        self.input_tokens = 0
        self.copied_pages = 0

    @property
    def _pool(self) -> BlockPool:
        return self._block_manager.device_block_pool

    def _evict(self) -> None:
        key, entry = self._entries.popitem(last=False)
        for digest in entry.hashes:
            if self._index.get(digest) == key:
                del self._index[digest]
        for block in entry.blocks:
            self._pool.free_block(block)
        self._cached_pages -= len(entry.blocks)

    def _reserve(self, count: int) -> None:
        while self._pool.num_free_blocks < count and self._entries:
            self._evict()
        if self._pool.num_free_blocks < count:
            raise InsufficientBlocksError(
                "Embedding batch exceeds KV pool capacity"
            )

    def _flush_copies(self) -> None:
        # Submit all partial-page copies together. A separate driver call per
        # request would serialize host dispatch and defeat short-prefix reuse.
        if not self._pending_copies:
            return
        dst_pages: list[Buffer] = []
        src_pages: list[Buffer] = []
        for unit in self._kv_buffers[0].to_memory().values():
            for buffer in unit.buffers:
                for dst, src in self._pending_copies.values():
                    dst_pages.append(buffer[dst, :])
                    src_pages.append(buffer[src, :])
        batch_inplace_copy(dst_pages, src_pages)
        self.copied_pages += len(self._pending_copies)
        self._pending_copies.clear()

    def runtime_inputs(
        self,
        batches: Sequence[Sequence[TextContext]],
        **kwargs: Any,
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Copy private partial pages before the graph can append its suffixes."""
        self._flush_copies()
        return super().runtime_inputs(batches, **kwargs)

    def alloc(self, ctx: TextContext, **kwargs: Any) -> KVTransfer:
        """Claim cached pages, copy a partial page, and allocate the suffix."""
        if ctx.tokens.processed_length or ctx.tokens.actively_chunked:
            raise ValueError(
                "Embedding cache requires a complete, fresh prompt"
            )
        hashes = (
            tuple(
                compute_block_hashes(
                    ctx, [], self.MATCH_TOKENS, "sha256", bytes(32)
                )
            )
            if self.enabled
            else ()
        )
        self._pending[ctx.request_id] = hashes
        hit = 0
        entry = None
        for idx in range(len(hashes) - 1, -1, -1):
            entry_id = self._index.get(hashes[idx])
            if entry_id is not None:
                entry = self._entries[entry_id]
                self._entries.move_to_end(entry_id)
                hit = (idx + 1) * self.MATCH_TOKENS
                break
        blocks = self._block_manager.req_to_blocks[ctx.request_id]
        full_pages, partial = divmod(hit, self.params.page_size)
        # Pin the reusable pages before eviction; the active request owns these
        # references even if their snapshot is removed to make admission room.
        if entry is not None:
            for block in entry.blocks[:full_pages]:
                self._pool.touch(block)
                blocks.append(block)
            if partial:
                src = entry.blocks[full_pages]
                self._pool.touch(src)
                self._copy_pins[ctx.request_id] = [src]
        needed = (
            len(ctx.tokens) + self.params.page_size - 1
        ) // self.params.page_size
        self._reserve(needed - len(blocks))
        if partial:
            dst, _ = self._pool.alloc_block()
            blocks.append(dst)
            self._pending_copies[ctx.request_id] = (
                dst.bid,
                self._copy_pins[ctx.request_id][0].bid,
            )
        if hit:
            ctx.tokens.skip_processing(hit)
        # Base allocation sees the spliced pages and allocates only the tail.
        result = super().alloc(ctx, **kwargs)
        ctx.cached_prefix_length = hit
        self.requests += 1
        self.cache_tokens += hit
        self.input_tokens += ctx.tokens.active_length
        if self.requests % 1024 == 0:
            logger.info(
                "Embedding prefix cache: requests=%d cached_tokens=%d input_tokens=%d copied_pages=%d",
                self.requests,
                self.cache_tokens,
                self.input_tokens,
                self.copied_pages,
            )
        return result

    def step(self, ctx: TextContext) -> None:
        """Publish only after the entire forward and host result copy succeed."""
        hashes = self._pending[ctx.request_id]
        if not hashes or all(digest in self._index for digest in hashes):
            return
        # Only hold pages containing a reusable token. The last token itself
        # is deliberately excluded by compute_block_hashes.
        num_pages = (
            len(hashes) * self.MATCH_TOKENS + self.params.page_size - 1
        ) // self.params.page_size
        if num_pages > self._max_cached_pages:
            return
        while self._cached_pages + num_pages > self._max_cached_pages:
            self._evict()
        blocks = tuple(
            self._block_manager.req_to_blocks[ctx.request_id][:num_pages]
        )
        for block in blocks:
            self._pool.touch(block)
        key = self._next_entry
        self._next_entry += 1
        self._entries[key] = _Prefix(hashes, blocks)
        # Point common prefixes at a recently completed snapshot. Entries are
        # private to this model instance; standard hashing also honors cache_salt.
        for digest in hashes:
            self._index[digest] = key
        self._cached_pages += len(blocks)

    def release(self, ctx: TextContext) -> None:
        """Drop active and copy-source references, including failed forwards."""
        super().release(ctx)
        for block in self._copy_pins.pop(ctx.request_id, []):
            self._pool.free_block(block)
        self._pending.pop(ctx.request_id, None)
        self._pending_copies.pop(ctx.request_id, None)

    def reset_prefix_cache(self) -> None:
        """Evict every snapshot; active requests keep their own references."""
        while self._entries:
            self._evict()
