# chunk_gate_recurrence: FP32 kernels

Current FP entry status (2026-10-21): complete original/sum-left program pair.
`FPEquiv.chunk_gate_recurrence_sumLeft_equiv` compares both complete selected
sources on the unchanged public IO, including the repaired four-stride seed
load. The original product-left update and candidate product-right update
lower to the identical FMA ports with identical multiplicand order. The
proof admits every raw FP32 word, arbitrary instruction ports and the same
complete DG reduction plan; it uses no numerical rewrite rule or experimental
assumption. `Misc/NativeCorrect.lean` separately retains
`chunk_gate_recurrence_kernel_equiv` with its previous statement and proof.

`Kernels.lean` directly presents the complete forward, forward sum-left,
backward and backward sum-left programs as ordinary `triton` quotations.
Full-source equality theorems preserve both previous backward quotations.
`PairIO.lean` retains the original channels, dtypes, parameter binding,
addresses, masks, observers and exterior frames. The task-local `ProgramPair`,
`PairExecution`, `PairSafety`, `PairSourceSafety` and `PairResult` helpers prove
the local FMA connection, complete execution equality and both safety traces;
they do not import optional native or Tensor applications.

The canonical repair supports the optional seed’s four actual strides and an
empty sequence. This proof update does not change its Python source, input
checks, original contiguous preparations, allocations, default fusion, saved
tensors, final-slice zeroing, Torch reduction, or optional seed-gradient return.

## Input domain and mathematical result

Inputs are ordinary strided FP32 CUDA tensors on one device, with resolved
negative/conjugate view bits, nonnegative strides and valid backing storage.
Shapes are `x[B,H,N,K,V]`, `gate[B,H,N]`, optional `seed[B,H,K,V]`, and incoming
`[B,H,N,K,V]`. B/H are positive; K is a positive multiple of 64; V is a positive
multiple of 16; N may be zero. The existing `.contiguous()` preparations support
noncontiguous x, gate and incoming gradients. Seeds may be transposed, expanded
or offset. Readonly caller arguments may alias, and backward incoming may alias
the saved forward output.

Nonempty launches retain `B*H < 2^31`, `K/64 <= 65535`, `V/16 <= 65535`, current
CUDA device agreement, and every original signed-int32 intermediate check.
All formed uint64 pointers are checked, including dead negative reverse
pointers and final unused advances. Active accesses have four-byte storage
containment and scalar alignment. N=0 performs no kernel launch; this remains
in the wrapper domain. Kernel execution theorems therefore require N>0.

The forward result is the exclusive recurrence:

```
O[0] = seed or zero
O[t+1] = O[t]*gate[t] + x[t]
```

The kernel writes N states, so the last x/gate slice does not enter a returned
state. Define `E[N]=0`, `E[t]=E[t+1]*gate[t]+incoming[t]`. The backward kernel
writes `DI[t]=E[t+1]` and one complete 64x16 contraction
`DG[t,tile]=sum_tile(E[t+1]*savedO[t])` for t<N-1, plus `DL=E[0]` for every
launched backward program. The final DI/DG slices remain untouched by the
kernel. Their original host zeroing, the host sum over DG tiles, and the
optional return of DL are separate wrapper operations.

`Proofs/Math.lean` reuses the independent closed prefix/suffix mathematics and
proves `MathTarget.adjoint_pairing` for arbitrary perturbations. The native
word formulas preserve the actual FP32 instruction order; they make no
FP-to-Real equality or error-bound claim.

## Kernel proof structure

`IO.lean` declares finite input/output channels, their exact dtypes, address
windows, optional-seed mask and one complete program invocation. Only program
ID axes 0/1/2 are installed; other IDs, num-program values and registers retain
their original semantics. Forward inputs contain N-1 source/gate slices plus
the optional strided seed. Backward reads N gate/incoming slices, N-1 saved
slices, and writes only N-1 DI/DG slices plus the full DL tile. In particular,
the last backward gate remains an input even when its carry starts at zero.

`Proofs/RealReference.lean` states the independent mathematical recurrence,
and `Misc/NativeReference.lean` states the optional native word recurrence
on those finite arrays. `Correct.lean` exposes one `ProgramIO.Implements`
specification. `Misc/NativeCorrect.lean` exposes `ProgramIO.Observes` for the retained
decoded-source/selected-object binding in `Misc/NativeIO.lean`. The native graph
observation is an auxiliary theorem in [GraphObservation.lean](Misc/GraphObservation.lean).
Both observations include actual source access safety, successful complete-program
execution, all typed outputs and the exact exterior frame. Arbitrary native
input nodes remain admitted; observation reads only the final payload bits.
Neither main statement exposes a state, runner or existential final memory.

`Proofs/KernelSafety.lean` proves evaluated loads/stores in both actual source
prefixes, every loop iteration and the backward DL postlude stay within the
declared channel capacities. `IOBinding` connects finite inputs, bounds and
the unchanged exact footprint to execution. `Proofs/PublicIO.lean` and `Misc/NativePublicIO.lean`
compose these facts with existing execution and numerical-graph proofs.
The previous raw results, adjoint, grid coverage and tail theorems remain as
ordinary helper theorems in `Misc/RawCorrect.lean` and `Misc/RawNative.lean`.

The optional `Misc/TensorBridge.lean` exposes
ordinary Tensor real/native application theorems, logical-input correspondence,
check equivalence and physical pointer/range/frame bindings in `TensorBridge`.
Its native Tensor application still states the reference directly on actual
logical input words. Neither primary entry imports this bridge. Historical
wrapper entries remain in `Misc/Legacy/HostCorrect.lean` and `Misc/Legacy/HostFPEquiv.lean`.

`KernelForwardInit`, `KernelForwardLoop`, `KernelReverseInit`,
`KernelReverseLoop`, and `KernelPost` construct execution of the complete actual
typed source, including initial stores, all loop iterations, all pointer
updates, and DL’s final store. `KernelExecution` ties successful execution,
every output and the complete exterior Cell frame to the same final state.
`KernelTail` makes the untouched final DI/DG slices explicit.

`KernelReal` connects that execution to the independent closed formulas.
`KernelNative` connects it to input-only raw-word recurrences and the complete
1024-product tree. `KernelCompiler` binds selected object bytes, exact source
hash, argument order, specialization, compiler/options and decoded instruction
structure. H/N use their runtime i32 or specialized ABI; value one retains the
selected specialization. Seed strides are constexpr. FP32 payload, scratch,
output and accumulator precision and default fusion are retained.

`KernelTensor`, `KernelTensorInputs`, `KernelLogical`, `KernelLaunch`, `KernelBounds`, and
`KernelBound` bind actual shared FP32 Tensor objects to physical entry reads,
logical input words, all output stores, storage offsets, seed strides, checked
machine expressions, full four-byte address spans, readonly input preservation,
and exterior bytes. Their finite store commit is a pure memory interpretation,
not a new host copyback. Output placement contains only finite address coverage
and ordinary buffer separation. No allocator, whole execution receipt, returned
object, or desired numerical answer is a primary endpoint premise.

The Tensor result is complete for each actual program. `KernelGrid` proves
coverage of every head, time and feature coordinate and uniqueness across
programs. A composed whole-grid final Tensor heap is not claimed here.
Historical wrapper results remain available separately.

## Model boundaries and evidence

The explicit model boundaries are ordinary Tensor metadata/pointer/storage
semantics, local memory read/write/byte-frame laws, selected-object decoding,
and primitive FP32 add/multiply/FMA bit semantics. These are model data, not
pending numerical experiments. There is no split-versus-fused equality premise
and no new mathematical transformation hypothesis.

The selected singleton backward postlude retains `FMA(gate,+0,incoming)`;
general recurrence sites retain `FMA(carry,gate,incoming)`. Product-tree nodes
retain individual multiply, add and FMA operations. The compiler decoder has
no Tensor payload or output assertion as an input. Its binding is explicit;
there is no universal verified-compiler or installed-binary decoder claim.

CPU validation (`check_contract.py`) passes 22 complete forward/backward paths
and 11 prelaunch rejections. It covers N=0/1/2/4, multiple heads and both tile
axes, absent/dense/transposed/expanded/offset seeds, original preparations,
caller aliasing, incoming/output aliasing, saved output, optional returns,
caller frames and allocation tails. It also reproduces the original
strided-seed error and empty-sequence invalid first store.

`.veritile-cache/chunk-gate-kernel-profiles/` records 15 exact-source offline
Triton 3.7.1 CUDA SM90 profiles, each with TTIR/TTGIR/LLVM/PTX/cubin artifacts.
The bounded schedule inspection checks all 75 artifact hashes, singleton
operand order, the DG local FMA sequence, warp/interwarp reductions and a full
1024-leaf permutation. These profiles are bounded compiler evidence. CUDA
launches and independent native numerical replay have not been performed.

The previous 48-module proof closure and its evidence remain applicable to
the shared execution foundation. The IO-format revision is validated by
targeted builds of every new helper, both public entries, the unchanged Tensor
bridge and the joint `KernelAudit`. No new axiom, `sorry`, execution-success
premise or desired-output premise is used. This reorganization runs no GPU
experiment or independent native numerical replay.

Optional proof layout (2026-10-21): `Misc/GraphObservation.lean`, `Misc/NativeCorrect.lean`, `Misc/KernelAudit.lean`, `Misc/TensorBridge.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.
Historical wrapper readers are physically retained in `Misc/Legacy/HostCorrect.lean`, `Misc/Legacy/HostFPEquiv.lean`. They remain optional historical interfaces and are checked separately from the current primary/auxiliary import union.

Additional optional reader paths: `Misc/RawCorrect.lean`, `Misc/RawNative.lean`. Their declaration bodies are retained. Historical readers remain separate from current primary/auxiliary checks.
