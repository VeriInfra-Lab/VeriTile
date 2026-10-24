# softmax_flaggems host/grid composition — review package

Status: additive proof/evidence candidate, **not whole-program acceptance**. Acceptance increment is zero. Repository and Git were not modified. The frozen matrix-transpose package was not modified.

The nine final Lean sources are under `after/bench/tritonbench_g/softmax_flaggems/`. `HostCorrect.lean` contains `softmax_autograd_source_admitted_real`. `Proofs/HostGridHost.lean` also retains the explicitly dtype-erased helper `softmax_autograd_host_grid_real`. The latter is mathematical execution even for frontend-rejected profiles; only the former composes explicit source admission and ordered rejection with execution.

Every scheduled CTA calls the original kernel's `eraseDType.toAlgKernel` through `exec`, using actual program IDs and the previous mutable memory. The four grid theorems cover every output offset and preserve the full exterior frame. There is no output-value callback, independent-snapshot merge, or assumed launch success. Both backward formulas retain the source's arbitrary undefined padding samples.

The host model normalizes integer dim, derives M/N/K, preserves dtype/device/shape and the actual primitive-produced strides, uses every source heuristic and all six autotune candidates, saves the actual produced output, runs the backward producer/consumer chain, and returns `(gradient,None,None)`. A benchmark trace is an explicit list of actual configuration invocations, allowing arbitrary warmup/repetition counts; `[]` represents cache hit. Each trial uses the previous state. The final launch rewrites every output and is related to the pre-trial inputs by proved frame/undef preservation. Saved-output/output-gradient aliasing is supported: contiguous conversion may preserve that alias.

`LayoutPrimitives` is the explicit trusted Torch metadata boundary: `contiguous(shape,inputStrides)` feeds `emptyLike(shape,contiguousStrides)`, and backward allocation applies `emptyLike` to the produced output strides. These callbacks return stride lists only; no tensor values, launches or success facts are supplied. The 24 layout probes include three noncanonical singleton-stride outcomes, including shape `(2,1,1,3)` with output strides `(3,1,1,1)`. The discarded canonical-stride assertion and attempts remain preserved.

The scalar `WavesAboveOne` parameter denotes the trusted ordinary Python expression `(num_blocks / NUM_SMS) > 1`; it is not an execution or value oracle. Correctness is uniform in its Boolean result. The package does not replace Python binary64 division with unbounded integer comparison. Real device SM counts are positive; native bounded scalar arguments remain a separate finite domain obligation.

## Source outcomes that must remain visible

- Forward fallback rejects fp64 **input**; backward fallback rejects fp64 **saved-output storage**. This is the existing `SourceProfileSupported` boundary, confirmed on Triton 3.7.1.
- A cold backward autotune cache tries `[32,64,128,256,512,1024]` in source order. Ordinary `CompilationError` escapes the installed `_bench`; a runnable final choice cannot hide an earlier failed candidate. Cold fp64 inner succeeds in this frontend partition only for `N≤32`; cold fp64 noninner only for `N≤1`. A cache hit checks only its selected configuration. First-rejection proofs preserve the exact state after successful earlier trials.
- Positive M,N with K=0 enters the inner branch in the original source. For `(2,0), dim=0`, allocations have extent zero but the kernel launches active M*N addresses. This is an unsafe source branch, not a source guard. The successful safe theorem requires K>0; the source path and its failure are retained in the evidence.
- M=0 or N=0 raises division-by-zero after contiguous/empty_like. Scalar rank or out-of-range dim raises the assertion before allocation. `invokeChecked` records error outcomes; exact ordinary allocation/error ordering is established by the unchanged original-host AST probe, not by a new allocator model.
- The unused JIT helper `next_multiple_of` contains `tl.cidv`; direct use rejects in the frontend. It has no production caller. Production `prev_multiple_of` compiles and remains in the kernel source dependency.

## Evidence and limits

`evidence/host-probe-scope-addendum.json` labels the original 960 host cases as selected-configuration metadata probes. It must accompany the earlier receipt; `host_replayed` does not mean native wrapper success. There are also 352 heuristic cases, seven boundary cases and 24 saved-output/gradient alias cases and 24 singleton-layout cases.

`evidence/frontend-profile-receipt.json`: 80 exact kernel-body profiles, 70 frontend successes, ten fp64 fallback rejections. `evidence/autotune-probe-receipt.json`: 252 routes (36 cold, 216 cache-hit), with 28 propagated compiler rejections. The installed real Autotuner/Heuristics methods and unchanged original decorated AST execute. The JIT sink performs original-body ASTSource frontend generation; synthetic benchmark costs and two repetitions per candidate make ordering observable without a GPU. These probes establish no native backend, timing, resource or GPU value result.

`evidence/attempts/` preserves every Lean attempt. Final staged sources compile with clean axiom footprints (standard `propext`, `Quot.sound`, `Classical.choice` only), and independent-reference checks pass. `evidence/final-compiles.json` binds successful receipts to exact final source hashes. `evidence/repo-dependencies.json` freezes the repository import closure and observed imported object hashes; existing dependencies were reused, not all rebuilt here.

`remaining-scope.json` states the finite remaining work. In particular these are mathematical dtype-erased source executions, not a bit-level fp16/bf16/fp32/fp64 softmax theorem. No new numerical assumption has been registered. Mixed casts, finite rounded arithmetic, exp, reductions and nonfinite/undefined values need a faithful typed reference or exact valid numerical registrations; false cast/rounding identities must not be waived. Native scalar/index/byte bounds require a concrete API/target domain, rather than a generic compiler/runtime proof.

## Reproduction and integration

`stage.py` translates temporary `SoftmaxHost.*` imports to the nine final additive paths. `compile_staged.py RELATIVE_PATH` compiles into `build-overlay/`, using existing repository objects read-only. The overlay contains symlinks to repository `bench` objects because Lean resolves the top-level module prefix as a unit. Only the nine `.lean` files under `after/` are proposed repository additions; do not copy build artifacts or symlinks.

Integration metadata is a proposal, not an independent review. Root must substantively review the proof/source bindings, choose the finite native/domain and numerical follow-ups, bind current public statements in schema2, and run the unchanged official checks. No scope row or acceptance counter was modified.
