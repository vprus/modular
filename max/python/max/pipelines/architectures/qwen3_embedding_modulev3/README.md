# Experimental embedding prefix cache

The ordinary `Qwen3ForCausalLM_ModuleV3` architecture remains cache-free.
To opt into this prototype, set the checkpoint's `config.json` architecture to
`Qwen3ForCausalLM_PrefixCache_ModuleV3`, then launch MAX with
`--task embeddings_generation --pool-embeddings --enable-prefix-caching`.
Use a separate checkpoint config; the weight files can remain shared.

The prototype supports causal, last-token pooled Qwen3 text embeddings on one
device, with the cache dtype matching the model dtype. Fixed, statically merged
adapter weights work like any other fixed checkpoint. Dynamic adapter selection,
multimodal contexts, unpooled outputs, tiered caches, and distributed serving
are outside this prototype's scope.

Prefix identity uses MAX's salted SHA-256 token hash chain at 16-token
boundaries. Reuse always leaves at least one token to execute, so the response
is computed from the final hidden state rather than an embedding response cache.
RoPE positions include the number of cached tokens independently for each
request. Q/K normalization and native grouped-query attention are retained.

GPU pages keep their supported native size (128 tokens by default). Complete
pages are shared without writes. A partial last page is copied into a private
page before the new suffix is written. All partial-page copies in one forward
use a single batched driver submission. This supports short instructions
without changing the attention kernels' page layout.

Snapshots are published only after a successful forward and host result copy.
An LRU limits retained snapshot references to half the allocated page pool;
admission can evict more snapshots to make room for an active batch. Active
requests pin their shared pages and copy sources, including across eviction.
Every claim is released when synchronous execution finishes or raises.

This is a memory-heavy initial implementation for short requests: retaining a
partial prefix still retains its physical page. Compact partial-page storage is
future work. The general implementation returns the full embedding dimension;
benchmark-specific dimension slicing belongs in the benchmark setup.

CPU ownership tests are in
`max/tests/tests/pipelines/lib/test_embedding_prefix_cache.py`. GPU correctness
must also be checked with actual weights, mixed prompt lengths, warm and cold
prefixes, reordered requests, and diverging suffixes. BF16 results can differ
when reuse changes matrix shapes and reduction order; compare normalized vectors
with a numerical tolerance rather than assuming bitwise equality.

The GPU regression target is
`//max/tests/integration/architectures/qwen3_embedding_modulev3:tests_gpu`.
It compiles one small, deterministic FP32 model and checks cache equivalence
across full pages, partial pages, mixed lengths, duplicates, and diverging
suffixes. It needs no downloaded checkpoint.
