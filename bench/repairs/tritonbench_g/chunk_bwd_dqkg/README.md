# chunk_bwd_dqkg: kernel IO proof

Optional proof entries are collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [Misc/HostCorrect.lean](Misc/HostCorrect.lean), [Misc/HostFPEquiv.lean](Misc/HostFPEquiv.lean), [Misc/HostReader.lean](Misc/HostReader.lean), [Misc/Reader.lean](Misc/Reader.lean). Their statements, proof bodies and recorded scopes are retained.
`Misc/Reader.lean` exports `Reader.realApplication/nativeApplication` for Tensor applications; `Misc/HostReader.lean` exports `HostReader.realApplication/nativeApplication` for host applications. The distinct namespaces allow both readers and all task `Misc/` entries to be imported with the primary entries. See the [namespace-fix validation](../../../review-checkpoints/2026-10-22/misc-import-fix/README.md).

The primary pair is `Equivalence.tail_bound_equiv : original ≡[[]] tailBoundCSE`.
The complete ordinary candidate replaces the final gate bound
`min(BT, T - i_t*BT) - 1` by
`(min(i_t*BT + BT, T) - i_t*BT) - 1`. It exposes the same absolute chunk end
already computed for `last_idx`, permitting CSE. No measured speedup is claimed.
The historical dotLeft candidate remains available; swapping its add/dot order
would affect the current native model's matrix carry and provenance routing,
so its old Real proof is not used as a native equality assumption.

`Proofs/PairConnection.lean` proves whole-state run equality, including failure,
and actual source access-safety equivalence for arbitrary unchanged backends.
The exact Nat identity holds for every register value, including saturated
subtraction cases. `Proofs/PairIO.lean` connects both complete actual source
grids for every legal CTA permutation, arbitrary compound Node inputs, the
runtime scale word, all dq/dk/per-key-tile dg outputs, tags and both raw frames.
All floating operations and their source provenance remain unchanged; the pair
retains the same arbitrary Instructions and Routing parameters and requires
no floating numerical rewrite or experimental rule.

The independent `Correct.lean` result and previous ordinary single-source native
results in `Misc/NativeCorrect.lean` retain their complete statements and domains.
`Proofs/SourcePreservation.lean` proves that replacing the former source macro
with complete ordinary quotations preserves every prior storage/swap interface
and the complete source AST. The archived macro is outside the main closure.

The selected canonical source is `chunk_bwd_dqkg_repaired.py`, SHA256
`7c2c6aef92283c9c3f9f4980291ab2367eaf6096436b5847c683a560335a1ac6`.
This pass leaves its Python text unchanged. The original benchmark and all 88
pre-existing task files are preserved in the evidence baseline.

The authoritative typed source is `Kernels.lean`; `Source.lean` retains its
selected-source connection. `Contract.lean` and `Correct.lean` expose the public
contract and Real specification; `FPEquiv.lean` declares the complete two-kernel
equivalence. `Misc/Reader.lean` imports
these entries and the optional Tensor bridge. They describe one complete kernel
grid, in any permutation of its CTAs, with actual predecessor memory and fresh
per-CTA program IDs. Output buffers already exist at kernel entry.

## Mathematical objective

For one chunk and the key coordinates owned by a CTA, with entry history H,
exit-history cotangent F, output cotangent D, cumulative log gates g, and scale s:

```
O[r,j] = s exp(g[r]) sum_e q[r,e] H[e,j]
       + s sum_{u<=r} exp(g[r]-g[u]) sum_e q[r,e] k[u,e] v[u,j]
Hout[e,j] = exp(g[last]) H[e,j]
          + sum_u exp(g[last]-g[u]) k[u,e] v[u,j]
L = sum_{r,j} D[r,j] O[r,j] + sum_{e,j} F[e,j] Hout[e,j]
```

`KernelMath.loss` gives the independently contracted form of this primal loss.
`KernelMath.gradients` proves its directional derivative is the pairing with
dq, dk and dg. `KernelObjective` binds the coefficients to logical input
contractions and proves that dq/dk and every active per-key-tile dg are exactly
the formulas in `Spec`. Padded rows and keys have zero logical input values.
The final-row correction uses gated dk before the attention update. H and F
are explicit entry-memory inputs, so no saved-forward execution premise is
needed. Summing dg over key tiles and the reverse cumulative sum belong to
separate downstream kernels.

## Source, storage, and precision

The readable typed quote is the complete canonical FP32 kernel. Existing
source proofs establish every load, loop step, output cell and exact regional
frame. The Real endpoint executes the quote's explicit dtype erasure and
matches the independent objective formulas. The native endpoint executes the
typed quote with a distinguished runtime scale slot and matches a separately
written finite instruction recurrence.

`KernelContract.check` is an executable metadata checker: exact shapes and
canonical dense strides, CUDA FP32, same device, no lazy flags, positive K/V,
natural alignment, pairwise disjoint live spans, signed32 source-index envelope,
u64 formed-pointer bounds and CUDA grid bounds. Empty B/H/T has no CTAs. The
native ABI binds both supported warp choices (4 and 8), Config's 3 stages,
constexprs, integer-one/runtime specialization and actual pointer alignment.
Python numeric-kind and FP32 scale conversion are a local ABI boundary; the
admitted word is finite and its rational projection gives the Real scale.
Negative-zero scale bits are retained in the native proof.

The optional complete Tensor applications use the shared
`VeriTile.Torch.FP32.Tensor` storage/index model.
The source's `.real` compatibility cells and `.fp32` cells contain the same
kind of FP32 word in the native path. `KernelTensor.physical_eq_logical` proves
the input address interpretation; the output placement derives finite coverage,
bounds and separation from the metadata contract. `Memory.API` commits exactly
the source output stores. Every output Tensor cell is read back at its actual
storage address, and all backing bytes outside those stores are preserved.
Disjoint views into one backing allocation are allowed.

## Native boundary and evidence

The new native path supports the observed SM90 WGMMA m64n{8,16,32,64}k8 TF32
instruction families. Variable-width dots are explicit K8 chains; reductions
have complete padded-lane trees. Local routing records FMA selection, operand
order, seed initialization/accumulation, negation and reduction order using
value-erased source provenance. The exp path is multiplication by the actual
FP32 log2(e) word followed by ex2.approx.f32.

The compiled-object decoder, these local routing choices, scalar/WGMMA ISA
word functions, and numeric-to-FP32 conversion are abstract model data. No
executable verified binary decoder or concrete Lean routing instance for the
saved profiles is claimed. There is no whole-dot/reduction/output oracle and
no primitive-equals-software numerical premise. The historical SM80 graph and
host proofs remain in the preserved historical modules and Host* entry files.

Evidence is under `.veritile-cache/kernel-evidence/chunk_bwd_dqkg/`:

- The original CPU contract harness passed 12 admitted cases and 93 rejection
  controls, both warp choices, complete outputs and backing-byte frames.
- Five independently differentiated primal cases passed comparisons against
  the actual kernel AST on CPU, including time/key/value tails, multiple value
  blocks, nonzero history/gates/cotangents, and negative-zero scale.
- Twelve exact-AST, actual-installed-binder SM90 offline profiles compiled,
  yielding TTIR, TTGIR, LLIR, PTX and cubin files for both warp choices. Their
  defaults, specializations and instruction-family summary are retained.

CPU arithmetic is not GPU/TF32 numerical evidence, and offline compilation is
not device execution. These checks corroborate the source and boundary choices;
the kernel theorems do not depend on a deferred numerical experiment. Host
allocation, fills, dispatch-device behavior, autotuner lifecycle, synchronization,
resource admission and downstream reductions are outside this kernel-only task.
Reused non-task objects are pinned as an explicit cache boundary, independently
of current source hashes when a matching source receipt is unavailable.

## Public IO and proof organization

`IO.lean` exposes seven dense input channels and three output channels, their
exact region names, lane counts, active windows, dtypes and the complete CTA
schedule. q/k/dq/dk have shape `[B*H,T,K]`, v/do `[B*H,T,V]`, h/dh
`[B*H,ceil(T/64),K,V]`, g `[B*H,T]`, and dg `[ceil(K/BK),B*H,T]`.
The finite arrays flatten those shapes in source order. There is no private
scratch memory. Empty grids and arbitrary full-grid permutations are retained.

`Correct.lean` has one specification using the shared
`PreparedAlgorithmProgramIO.Implements` judgment and independent
`Target.mathematical` dq/dk/dg formulas. Each Real CTA keeps the original fresh
registers and zero undefined-load value. `Misc/NativeCorrect.lean` retains an
ordinary native theorem using shared `ProgramIO.Observes`: `NativeTarget.words` evaluates the independent
source-ordered native recurrence. All original symbolic input terms remain
admitted. The `.fp32` and `.real` channel tags exactly match the existing typed
interpreter, with explicit source FP32 arithmetic unchanged.

The shared judgments connect declared windows to actual source access safety,
successful execution, every logical output, exact output tags and the exterior
memory frame. `Proofs/RealIOResults.lean` reuses the existing Real source and
safety proofs. The typed entry, value-loop and final-store safety proofs in
`Proofs/Typed*Safety.lean` follow the same actual successor states as the retained
execution proof. `Proofs/NativeObservation.lean` composes that safety with the
complete existing native execution and independent formula binding. Neither
program success nor access safety is a numerical assumption.

`Proofs/KernelResults.lean` retains the former raw execution endpoints, gradient
corollaries and empty-grid identities. `Misc/TensorBridge.lean` preserves the complete
original Tensor theorem statements, including physical readback, input words,
backing-byte frames and compiler/scalar ABI bindings; it derives them from the
same source-execution foundation. Helpers do not import public reader entries.
The Real specification and optional native theorem pass `#stmtSurfaceSubset`, and their targets pass
non-circularity and standard-axiom audits. `Misc/Reader.lean` also audits the retained
complete Tensor applications.

This format pass does not change Python, Kernels, Contract, numerical assumptions,
independent replay or GPU execution status.
