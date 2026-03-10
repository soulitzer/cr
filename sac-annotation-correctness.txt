# Annotation mechanism refactor: motivation and correctness

## Why the annotation mechanism changed

The policy function needs to inspect op outputs (e.g. to support `checkpoint_name()`, where a tensor is named after production and the policy decides based on the name). This requires calling the policy **after** the op runs.

The old mechanism set `fx_traceback.current_meta["recompute"] = policy` **before** the op, so FX nodes created during the op inherited the annotation. This can't work post-op because the nodes already exist by then. The new mechanism snapshots the graph length before the op, runs the op, calls the policy, then directly sets `node.meta["recompute"]` on all nodes created since the snapshot.

## Definitions

- **SAC dispatch**: a call to `_CachingTorchDispatchMode.__torch_dispatch__` for a single op `func`.
- **Annotation**: the value of `node.meta["recompute"]` on an FX node in the joint graph.
- **Policy**: the value returned by `policy_fn(ctx, func, ...)` for a given op.
- **Top-level dispatch**: a SAC dispatch that is not a re-entrant call from within another SAC dispatch (e.g. not a decomposed sub-op).

## Correctness claim

**For every top-level non-ignored op dispatched through `_CachingTorchDispatchMode`, all FX nodes created by that op (including nodes from decomposed sub-ops) receive exactly the annotation returned by the policy for that op.**

## Proof

Consider a top-level dispatch for op `F` with policy result `P`.

**Step 1**: At line 1415, we snapshot `graph_len_before = len(nodes)`.

**Step 2**: At line 1417, `func(*args, **kwargs)` executes. This may:
- (a) Create a single FX node via proxy tensor mode (common case).
- (b) Decompose via `maybe_handle_decomp` or `func.decompose()` in proxy_tensor.py, creating multiple FX nodes. Each decomposed sub-op re-enters `_CachingTorchDispatchMode.__torch_dispatch__`, which calls the policy for each sub-op and tags the sub-op's nodes.
- (c) For multi-output ops, proxy mode creates the op node plus `getitem` nodes.

**Step 3**: At line 1440, the policy is called, returning `P`.

**Step 4**: At lines 1447-1449, we iterate all nodes from `graph_len_before` onward and set `node.meta["recompute"] = P`.

In case (a): the single node gets `P`. Correct.

In case (b): each sub-op was tagged by its own re-entrant dispatch (step 2), but step 4 **overwrites** all of those with `P`. The outer op's policy wins. This is the desired behavior: the user's policy targets `F` (e.g. `silu`), and all implementation-detail nodes inherit that annotation.

In case (c): the op node and all getitem nodes are created between the snapshot and the tagging. Step 4 tags all of them with `P`. Correct.

## SAC_IGNORED_OPS

Ignored ops (detach, metadata ops) are handled separately at lines 1407-1410. They:
1. Set `fx_traceback.current_meta["recompute"] = PREFER_RECOMPUTE` (pre-op, since the op hasn't run yet).
2. Return early without calling the policy.

This uses the old `current_meta` mechanism, which is correct here because the value is set immediately before the op runs, so there's no window for stale state.

**Pre-fix bug**: Before this PR, ignored ops had no compile-time handling. They returned early without setting any annotation, so nodes created by ignored ops inherited whatever stale `current_meta` was left by the previous op. This is the leak fixed by this PR, tested in `test_pre_mode_decomp_has_sac_ignored_ops`.

## Assumptions

1. **No node creation between lines 1417 and 1448 outside of `func` execution.** Lines 1422-1443 do schema inspection, counter updates, tensor tracking, and the policy call. If any of these were to dispatch an op through proxy mode, spurious nodes would be incorrectly tagged with `P`. This holds today but is an implicit invariant.

2. **`proxy_mode` is non-None and stable during compilation.** The snapshot and tagging use the same `proxy_mode.tracer.graph`. If proxy mode were swapped between the snapshot and tagging, the node indices would be inconsistent.

3. **Graph node list is append-only.** The snapshot relies on `len(nodes)` being a valid lower bound for nodes created before the op. If nodes were removed or reordered between the snapshot and tagging, the window would be wrong.

4. **Re-entrant sub-op dispatches complete before step 4.** All decomposed sub-ops finish executing (and create their FX nodes) within `func(*args, **kwargs)`. The outer tagging at step 4 then overwrites all of them. If a sub-op deferred node creation to after step 4, that node would not be tagged.

## Known trade-offs

- **O(n) per dispatch**: `len(list(proxy_mode.tracer.graph.nodes))` materializes the full node list at each dispatch. For very large graphs, this is quadratic overall. This affects compile time, not correctness.

- **Outer policy wins for decomposed ops**: If a user's policy returns `MUST_SAVE` for `silu`, all of silu's decomposed sub-ops get `MUST_SAVE`, even if the policy would have returned `PREFER_RECOMPUTE` for the individual sub-ops. This is intentional — the user targets the op they see, not its implementation details. But it means the sub-ops' individual policy calls (from re-entrant dispatch) are wasted work.
