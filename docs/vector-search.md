# Vector search in BookRAG

This note explains what BookRAG's vector store is actually doing, and why,
for anyone extending the retrieval layer later (see "Future work" at the
bottom for the planned HNSW/IVF/PQ/FAISS experiments).

## What "ANN" means

**ANN = Approximate Nearest Neighbor search.** Given a query embedding,
finding the chunks whose embeddings are closest to it (by cosine similarity
here) is a nearest-neighbor search problem. An *exact* nearest-neighbor
search compares the query against every single vector in the collection —
guaranteed correct, but linear in the number of vectors. An *approximate*
search uses an index structure that can skip most of the collection while
still returning the true nearest neighbors with very high probability, in
exchange for a small, tunable chance of missing the true best match.

## Why vector databases use ANN instead of exact search

For a few thousand chunks (BookRAG's scale for a single book), exact search
is actually fast enough that you would barely notice the difference. ANN
starts to matter as the collection grows: exact k-NN search over N vectors
costs O(N) similarity computations per query, which becomes the bottleneck
once N reaches the tens/hundreds of thousands to millions of vectors typical
of large multi-document corpora. Vector databases default to ANN so the same
codebase scales from a single book to a large corpus without a rewrite —
BookRAG gets that scalability for free by using Chroma's default index, even
though at this project's current scale exact search would work just as well.

The trade-off ANN makes is **recall vs. speed**: a well-tuned ANN index
typically returns 95-99%+ of the same top-K results as exact search, at a
small fraction of the compute cost, and that recall/speed trade-off is
usually adjustable via the index's construction and search parameters.

## How HNSW works (conceptually)

Chroma's default index is **HNSW** (Hierarchical Navigable Small World
graphs). The intuition:

1. Every vector becomes a node in a graph, connected to a handful of its
   nearest neighbors — like a "small world" social network where anyone is
   reachable from anyone else in a few hops.
2. That graph is built in **layers**: the top layer has very few nodes with
   long-range links (for taking big jumps across the vector space quickly),
   and each layer below has progressively more nodes with shorter, denser
   links, down to the bottom layer which contains every vector.
3. A search starts at an entry point in the sparse top layer, greedily walks
   toward the query vector (always moving to whichever neighbor is closer),
   and once it can't get any closer in that layer, it drops down to the next,
   denser layer and keeps refining — like coarse-to-fine navigation.
4. This reaches a very good (usually optimal) answer in roughly logarithmic
   time relative to the number of vectors, instead of scanning everything.

HNSW's main knobs (exposed by Chroma if you need to tune them) are how many
neighbors each node keeps (`M`) and how wide the search frontier is during
construction/search (`ef_construction`/`ef_search`) — more neighbors and a
wider frontier trade memory and speed for higher recall.

## What index Chroma is using here

BookRAG creates its collection with `hnsw:space: "cosine"` (see
`app/retrieval/vector_store.py`), so Chroma indexes the chunk embeddings with
HNSW using cosine distance — a good default for normalized sentence
embeddings like the ones `sentence-transformers` produces here (BookRAG
always L2-normalizes embeddings, see `embed_documents`/`embed_query`, so
cosine similarity and dot product agree). BookRAG does not configure HNSW
manually beyond that; Chroma's defaults are appropriate at this project's
scale (a single book, low thousands of chunks).

## Exact search vs. ANN, concretely

| | Exact (brute-force) | ANN (HNSW) |
|---|---|---|
| Guarantee | Always the true top-K | Usually the true top-K (tunable recall) |
| Cost per query | O(N) | ~O(log N) |
| Index build cost | None | Graph construction, done once at ingest time |
| Good for | Small N, or when 100% recall is required | Larger N, latency-sensitive search |

For BookRAG's current single-book scale, either approach would feel
instant; HNSW is used because it is Chroma's default and it is what this
project will need once it stops being single-book.

## Future work

This project is expected to later experiment with alternative ANN
strategies for comparison:

- **HNSW** (current default via Chroma) — graph-based, as described above.
- **IVF** (Inverted File Index) — clusters vectors (e.g. via k-means) into
  buckets ("cells"); a query only searches the handful of buckets nearest to
  it instead of the whole collection. Faster to build than HNSW, generally
  a bit lower recall for the same speed budget.
- **PQ** (Product Quantization) — compresses each vector into a short code
  by splitting it into sub-vectors and quantizing each sub-vector against a
  small codebook; distance can then be approximated from compact codes
  instead of full-precision vectors. Dramatically reduces memory at some
  recall cost, and is often combined with IVF (`IVF-PQ`) for large corpora.
- **FAISS** — a library implementing all of the above (and combinations like
  `IVF-PQ`) with fine-grained control over the recall/speed/memory
  trade-off; a natural place to prototype these comparisons outside of
  Chroma's managed HNSW index.

None of these are implemented in V1 — BookRAG intentionally uses Chroma's
built-in ANN as-is rather than hand-rolling an index.
