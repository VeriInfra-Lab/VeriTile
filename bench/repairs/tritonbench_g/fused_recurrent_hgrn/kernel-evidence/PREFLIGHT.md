# HGRN kernel-only preflight

Selected source: `fused_recurrent_hgrn_repaired.py`, SHA256
`f920993bd8b61bc71450ab7a1775ab8dc4c350435e81b83728834df1af922cde`.
The source and guards are unchanged. The 33 earlier files were pinned before
editing; the two former public wrapper endpoints are retained under
`Proofs/HistoricalCorrect.lean` and `Proofs/HistoricalFPEquiv.lean`.

For each `(bh,d)`, independently of the DSL interpreter:

* `h[0] = h0` when a seed is supplied, otherwise zero;
  `h[t+1] = g[t] * h[t] + x[t]`, and `o[t] = h[t+1]`.
* Optional `ht = h[T]`, including `T = 0`.
* Reverse carry starts at supplied `dht` when final state was requested,
  otherwise zero. At time `t`, add `do[t]`, write that value to `dx[t]`,
  write its product with `h[t]` to `dg[t]`, then multiply carry by `g[t]`.
* The Real derivative is the directional derivative of
  `sum(do[t] * h[t+1]) + dht * h[T]`. The seed is held constant, matching
  the original detach. Missing `do` is the already zero-filled kernel input.

The input domain remains ordinary contiguous CUDA FP32 `[B,H,T,D]`, with
`B,H,D > 0`, `T >= 0`, same-device contiguous seed/cotangents, ordinary view
flags, aligned contained storage, and all original launch/address guards.
Nonzero offsets and readonly aliases are retained. Forward and backward may
choose different BD values, from 32/64/128 and all four original warp counts.

The old ignored-final-cotangent counterexample is `T=1`, `g=1`, `x=h0=do=0`,
`dht=1`: the derivative requires `dx=1`, but the old source produced zero.
The existing CPU harness reproduces this failure and checks the repaired source.
No new counterexample was found in the audited domain.

Native arithmetic retains source expression graphs. A supplied seed executes
an actual `+0` operation; backward time zero loads the original seed directly.
Saved forward output is evaluated before its backward multiplication. The
model has precisely scalar addRN, mulRN, and fmaRN word laws; a compiler schedule
can swap operands or contract an eligible multiply/add using graph topology,
never input word values. The theorem proves equality to the software model
under those explicit local laws. These are model boundaries, not deferred
numerical experiments or assumed whole-result equalities.

The 64 offline SM80 profiles used actual wrapper arguments and the actual JIT
binder, default fusion, all 12 autotune candidates, and T=0/1/2/3/4/17/1024 branches.
PTX preserves seed+0 (with operand reordering), forward FMA, backward add/mul,
and FMA across unrolled reverse-carry steps. The T=17/1024 reverse loops preserve
separate add/multiply. The T=3, BD=128, warps=1 backward candidate also retains
its loop and separate operations; the other audited short reverse candidates
contract. `Kernel.profileSchedule` exposes these two observed
topology-only choices, including seed+0 operand order. Reification of an arbitrary binary
as this ordinary compiler model remains external, as does Torch's implementation.
No GPU execution or native numerical experiment was performed.

The kernel result uses actual Tensor handles and their ordinary coordinate
laws, including nonzero storage offsets. The typed state observes each kernel
argument at its relevant entry time. Cotangents supplied after forward may
alias saved outputs. Their observed values are arbitrary inputs; this does not
require a host allocation, returned-object, or final unified byteheap theorem.
Every source store, loop, mask, grid owner and phase frame remains in scope.
