# ggml_cgraph: Data Structure, Initialization, and Graph Building

This document analyzes the `ggml_cgraph` structure — the computation graph
at the heart of every ggml-based program — and explains how it is
initialized, populated via `ggml_build_forward_expand()`, and eventually
executed.

---

## 1. Data Structure

Defined in `ggml/src/ggml-impl.h:329`:

```c
struct ggml_cgraph {
    int size;    // maximum number of nodes/leafs/grads/grad_accs
    int n_nodes; // number of nodes currently in use
    int n_leafs; // number of leafs currently in use

    struct ggml_tensor ** nodes;     // tensors with data that can change
    struct ggml_tensor ** grads;     // gradients of the nodes (training)
    struct ggml_tensor ** grad_accs; // gradient accumulators (training)
    struct ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of times each tensor is used

    struct ggml_hash_set visited_hash_set;

    enum ggml_cgraph_eval_order order;
    uint64_t uid;
};
```

### 1.1 Field semantics

| Field | Type | Purpose |
|-------|------|---------|
| `size` | `int` | Maximum capacity of `nodes[]`, `leafs[]`, and hash table slots |
| `n_nodes` | `int` | Current number of op nodes in `nodes[]` |
| `n_leafs` | `int` | Current number of leaf tensors in `leafs[]` |
| `nodes[]` | `ggml_tensor **` | Array of compute nodes, ordered for execution |
| `leafs[]` | `ggml_tensor **` | Leaf tensors (op == GGML_OP_NONE, not marked PARAM) |
| `grads[]` | `ggml_tensor **` | Per-node gradient tensors (NULL if `grads=false` at init) |
| `grad_accs[]` | `ggml_tensor **` | Per-node gradient accumulators (NULL if no grads) |
| `use_counts[]` | `int32_t *` | Reference count per hash-table slot; the allocator uses this for memory planning |
| `visited_hash_set` | `ggml_hash_set` | Hash table mapping `ggml_tensor *` → slot index |
| `order` | enum | `LEFT_TO_RIGHT` (default) or `RIGHT_TO_LEFT` — traversal order |
| `uid` | `uint64_t` | Optional unique id; 0 means not set |

### 1.2 Key constants

```
GGML_DEFAULT_GRAPH_SIZE  2048       // default size for ggml_new_graph()
GGML_MAX_SRC             10         // maximum source tensors per op
GGML_MAX_DIMS            4          // tensor dimensions
GGML_MAX_OP_PARAMS       64 (bytes) // op parameter storage per tensor
ggml_hash_size(4096)     = 4099     // hash table size for a graph of size 2048
```

### 1.3 `nodes` vs `leafs`

```
leafs: tensors with op == GGML_OP_NONE and NOT marked PARAM
        → data is constant (weights, pre-filled inputs)
        → counted in cgraph->n_leafs

nodes: ALL other tensors in the graph
        → op != GGML_OP_NONE (compute nodes)
        → OR op == GGML_OP_NONE but flagged GGML_TENSOR_FLAG_PARAM
        → counted in cgraph->n_nodes
```

Both arrays are populated automatically by `ggml_build_forward_expand()`.

### 1.4 `visited_hash_set` and `use_counts`

- Every tensor added to the graph gets a slot in the hash table.
- The hash table size is `ggml_hash_size(cgraph->size * 2)` — roughly double
  the node capacity, to accommodate both nodes and leafs.
- `use_counts[slot]` tracks how many other nodes reference this tensor.
  The allocator uses this to decide when a tensor's buffer can be reused
  (released after the last consumer).

### 1.5 Evaluation order

```c
enum ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,  // default
    GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
};
```

Controls the order in which source operands are visited during graph
expansion. `LEFT_TO_RIGHT` visits `src[0]`, then `src[1]`, ...;
`RIGHT_TO_LEFT` visits `src[GGML_MAX_SRC-1]` down to `src[0]`.
This affects the final ordering of `nodes[]`.

---

## 2. Memory Layout

Graphs are allocated fromggml_context's linear memory pool via
`ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, ...)`.

```c
// ggml_graph_nbytes(size=2048, grads=false):
//
// Bytes  Field
// ─────  ────────────────────────────────────────────────
//    ...
//    0    struct ggml_cgraph                           (sizeof ≈ 120)
//  120    nodes[2048]     → ggml_tensor *               ( 16 KB)
//  ...    leafs[2048]     → ggml_tensor *               ( 16 KB)
//  ...    use_counts[4099]→ int32_t                     ( 16 KB)
//  ...    hash_keys[4099] → ggml_tensor *               ( 32 KB)
//  ...    hash_used       → bitset                      (520 B)
// ─────
// Total: ~82 KB for size=2048 without grads; ~148 KB with grads
```

This is a **single linear allocation** — the struct header and all arrays
live in one contiguous block of memory. No dynamic resizing; if `n_nodes`
reaches `size`, the graph overflows (assertion failure).

---

## 3. Initialization

### 3.1 `ggml_new_graph(ctx)` — standard path

```c
struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx) {
    return ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}
```

Creates a graph with capacity 2048 nodes, no gradient support.

### 3.2 `ggml_new_graph_custom(ctx, size, grads)` — full control

```c
struct ggml_cgraph * ggml_new_graph_custom(
    struct ggml_context * ctx,
    size_t size,              // max nodes/leafs
    bool grads)               // allocate grads + grad_accs arrays?
```

Steps:
1. Compute total byte size via `ggml_graph_nbytes(size, grads)`.
2. Allocate a `GGML_OBJECT_TYPE_GRAPH` object in the context pool.
3. Lay out arrays sequentially after the struct header using `incr_ptr_aligned`:
   - `nodes[]`, `leafs[]`, `use_counts[]`, `hash_keys[]`, `grads[]`, `grad_accs[]`, `hash_used[]`
4. Initialize the hash set and clear gradient arrays if `grads=true`.

### 3.3 `ggml_graph_clear(cgraph)`

Resets `n_nodes` and `n_leafs` to 0 and clears the hash set. Does **not**
free memory. The graph can be reused for a new forward expansion.

### 3.4 `ggml_graph_reset(cgraph)`

For training: resets gradient accumulators. Sets loss-node grad to 1.0,
all others to 0. Also clears AdamW momenta.

---

## 4. Building the Graph: `ggml_build_forward_expand()`

### 4.1 Call chain

```
ggml_build_forward_expand(cgraph, tensor)
  → ggml_build_forward_impl(cgraph, tensor, expand=true, compute=true)
    → ggml_visit_parents_graph(cgraph, tensor, compute=true)
        (recursive depth-first walk via tensor->src[])
```

### 4.2 The visitor algorithm: `ggml_visit_parents_graph()`

```c
static size_t ggml_visit_parents_graph(
    struct ggml_cgraph * cgraph,
    struct ggml_tensor * node,
    bool compute)
```

This is a recursive DFS that walks backward from a target tensor through
its `src[]` pointers. For each tensor encountered:

**Step 1: Look up in hash set**
```
hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node)
```

**Step 2: Already visited?**
```
if (already in hash set as "used"):
    if (compute):
        recurse into unvisited src tensors (update COMPUTE flag)
    return hash_pos  ← stop; don't re-add
```

This deduplication prevents a tensor that appears in multiple sub-expressions
from being added to `nodes[]` twice.

**Step 3: First visit — register in hash set and explore**
```
hash_set.keys[hash_pos] = node
bitset_set(hash_set.used, hash_pos)
use_counts[hash_pos] = 0

for each src[i] of node (in evaluation order):
    src_hash_pos = ggml_visit_parents_graph(..., src[i], ...)
    use_counts[src_hash_pos]++  ← increment reference count
```

**Step 4: Classify as node or leaf**
```
if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)):
    → leafs[n_leafs++] = node
else:
    → nodes[n_nodes++] = node
```

The order of `nodes[]` is the DFS *post-order*: children appear before
parents. This ensures that when iterating `nodes[0..n_nodes-1]`, every
tensor's sources have already been computed.

### 4.3 `COMPUTE` flag propagation

When `compute=true`, `GGML_TENSOR_FLAG_COMPUTE` is set on every node
and propagated up to all ancestor ops. This marks the subgraph that
must actually run.

`ggml_build_forward_select()` uses this to build a graph with multiple
outputs but only compute one of them — the others have their flag unset
and the scheduler skips them.

### 4.4 Example: Building a graph

```c
struct ggml_context * ctx = ggml_init(params);
struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);  // leaf
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);  // leaf
struct ggml_tensor * c = ggml_add(ctx, a, b);     // node (src: a, b)
struct ggml_tensor * d = ggml_mul(ctx, c, c);     // node (src: c, c)

struct ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, d);
```

After `ggml_build_forward_expand`, the graph contains:

```
leafs[]:   [a, b]
nodes[]:   [c, d]
use_counts: [hash(a)=1, hash(b)=1, hash(c)=2, hash(d)=0]

Execution order: c (add), then d (mul)
```

### 4.5 Expanding vs building from scratch

`ggml_build_forward_expand()` has `expand=true`, meaning it appends to
the existing graph. New nodes are added after `cgraph->n_nodes`.
This lets you build a graph incrementally across multiple calls.

`ggml_build_forward()` (internal, called with `expand=false`) clears the
graph first via `ggml_graph_clear()`.

---

## 5. Inspecting and Manipulating Graphs

| Function | Purpose |
|----------|---------|
| `ggml_graph_node(cgraph, i)` | Get node by index (negative = from end) |
| `ggml_graph_nodes(cgraph)` | Get raw `nodes[]` array pointer |
| `ggml_graph_n_nodes(cgraph)` | Get node count |
| `ggml_graph_size(cgraph)` | Get max capacity |
| `ggml_graph_add_node(cgraph, tensor)` | Manually append a node |
| `ggml_graph_get_tensor(cgraph, name)` | Find tensor by name in leafs or nodes |
| `ggml_graph_get_grad(cgraph, tensor)` | Get gradient tensor for a node |
| `ggml_graph_get_grad_acc(cgraph, tensor)` | Get gradient accumulator |
| `ggml_graph_view(cgraph, i0, i1)` | Create a sub-graph view of `nodes[i0..i1)` |
| `ggml_graph_cpy(src, dst)` | Deep-copy graph contents |
| `ggml_graph_dup(ctx, cgraph, grads)` | Allocate a duplicate graph in a new context |

---

## 6. Execution

The graph itself is just a data structure — computation requires a
backend:

```c
// Execute on CPU
ggml_backend_graph_compute(backend, cgraph);

// Or via the older context-based API
ggml_graph_compute_with_ctx(ctx, &cgraph, n_threads);
```

Both walk `nodes[0..n_nodes-1]` sequentially, dispatching each op to
the backend's implementation. The `use_counts[]` array is consumed
by the memory allocator (ggml-alloc) to determine buffer lifetimes.

### 6.1 Graph plan (`ggml_cplan`)

```c
struct ggml_cplan {
    size_t    work_size;  // scratch buffer size
    uint8_t * work_data;  // scratch buffer pointer
    int       n_threads;  // parallelism hint
    struct ggml_threadpool * threadpool;
};
```

Call `ggml_graph_plan(cgraph, n_threads)` before compute to pre-allocate
scratch buffers and configure thread pools.

---

## 7. Training / Backward Pass

To enable gradient computation, initialize with `grads=true`:

```c
struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 2048, /*grads=*/true);
```

After building the forward graph, run the backward expansion:

```c
ggml_build_backward_expand(ctx, gf, grad_accs_array);
```

This populates `grads[]` and `grad_accs[]` with the gradient tensors for
each node. The forward and backward nodes are both in the same `nodes[]`
array — backward nodes are appended after the forward nodes.

Call `ggml_graph_reset(gf)` between training iterations to reset gradient
accumulators.

---

## 8. Graph Lifetime Rules

1. A `ggml_cgraph` lives in the `ggml_context`'s memory pool.
   `ggml_free(ctx)` destroys it along with all tensors.
2. Graphs CANNOT outlive their context.
3. Nodes in the graph (tensor pointers in `nodes[]`) CAN outlive the graph
   itself — but they still cannot outlive the context.
4. `ggml_graph_view()` returns a *stack copy* of the struct with its own
   `nodes[]` pointing into the parent's array. It is not a heap allocation.
5. Calling `ggml_build_forward_expand()` on the same graph multiple times
   is safe — new nodes are appended. Call `ggml_graph_clear()` first if
   you want to start over.

---

## 9. Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│ ggml_context (memory pool)                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ggml_cgraph struct + all arrays (single allocation)   │   │
│  │                                                       │   │
│  │  size = 2048                                           │   │
│  │  n_nodes = 3    n_leafs = 2                           │   │
│  │                                                       │   │
│  │  leafs[0] ──────────→ a (ggml_tensor, op=NONE)        │   │
│  │  leafs[1] ──────────→ b (ggml_tensor, op=NONE)        │   │
│  │                                                       │   │
│  │  nodes[0] ──────────→ c = ggml_add(a, b)              │   │
│  │  nodes[1] ──────────→ d = ggml_gelu(c)                │   │
│  │  nodes[2] ──────────→ e = ggml_mul_mat(weight, d)     │   │
│  │                                                       │   │
│  │  use_counts:                                          │   │
│  │    hash(a)=1  hash(b)=1  hash(c)=1                    │   │
│  │    hash(d)=1  hash(e)=0                               │   │
│  │                                                       │   │
│  │  visited_hash_set: {a b c d e} → slot indices         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ggml_tensor objects (also in context pool)             │   │
│  │  a.src = {}    b.src = {}                              │   │
│  │  c.src = {a,b}  d.src = {c}   e.src = {weight,d}      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Execution order (nodes[]):
  1. c = add(a, b)
  2. d = gelu(c)
  3. e = mul_mat(weight, d)
```
