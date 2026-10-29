# FP32 backward RoPE kernel

The selected source is `rope_backward_transform_repaired.py`, SHA256
`2fb8908af0d5a2b419c2aa493ebf377bfff19bff2676655f18a354263c11b93c`.
The current proof update leaves the Python source and its guards unchanged.
`Kernels.lean` quotes the complete two-branch JIT, including all four gradient
loads, coefficient loads, casts and stores. `Contract.source_binding` identifies that quote with the
canonical source used by the re-proved execution and memory results.
Its proof source is a thin reference to `Kernels.lean`; no second statement
list is maintained. Four pointer storage families, seven runtime integer
bindings and nine constexpr roles are outside the quotations. Both independent
coefficient column strides are retained. The index-sharing Python counterpart
is explicitly authored from the existing candidate, and both complete bodies
are checked by `jit-correspondence.json`. The public Python call selects
`BACKWARD_PASS=True`, which is the scope of the public theorems.

At the Tensor application boundary, Q and K are contiguous FP32 Tensor views of
shape `[B,S,H,D]`, with independent head counts. The selected launch explicitly
passes `H*D` as each gradient row stride. Coefficients are actual FP32 tables
`[L,D//2]`, with `L >= S`, independent nonnegative row and column strides,
arbitrary valid storage offsets, and a common CUDA device. Zero coefficient
strides and readonly cosine/sine aliases are included. Q and K backing
allocations are separate from each other and from both coefficient allocations.
Shared Tensor storage/index laws and the ordinary strided-table index law bind
the source's pointer reads to actual logical Tensor values.

For each row, head and half-channel, the independent real target is
`(a*c+b*s, b*c-a*s)`. This is the input adjoint of the fixed-coefficient forward
map `(a*c-b*s, b*c+a*s)`, proved by `Spec.adjoint`. No unit-circle premise is
needed, and the adjoint need not be an inverse. Odd last channels are unchanged.
Empty rows, `D < 2`, or two empty head sets submit no rows. A single empty head
set is valid; masks suppress its accesses while the other gradient is updated.

`Correct.lean` exposes `rope_backward_kernel_correct` using the shared
`ReadProgramIO.Implements` judgment and independent adjoint formula. `IO.lean`
shows both in-place Q/K channels, separately strided coefficient channels, the
complete row schedule, and the distinct writable/result-read windows. Every
result column is observed, including unchanged odd tails; only the rotated
halves are writable. `Proofs/AccessSafety.lean` checks every reached source
load/store, and `Proofs/IOGrid.lean` connects the actual shared invocation
schedule to the existing complete grid. The original raw correctness and
no-launch results remain ordinary supporting theorems in `Proofs/KernelResults`.
`TensorBridge.rope_backward_kernel_correct` retains the complete Tensor input
binding, derived padded signed-i32/u64 access bounds, logical observations and
physical word frame.
The grid relation explicitly invokes `Structural.exec` for each program and
threads the same source memory through completion. The retained Grid proof
also supports arbitrary distinct row completion orders.

`Misc/NativeCorrect.lean` exposes `rope_backward_kernel_fp32` with concrete shared FP32
primitives under `FP.Config`. Its independent ordered native graph is
`low = fma(c,a,mul(s,b))` and `high = fma(c,b,neg(mul(s,a)))`.
Both multiplications are coefficient-first, FMA is genuine fused arithmetic,
and the source's FP32 casts remain present. The primary theorem uses the same shared IO judgment with an independent
native formula. Initial tagged FP32 inputs remain unrestricted; its observer
removes only internal product provenance and retains every output word.
Complete source execution, actual access safety, all native observations and
the exact frame are conclusions. This optional single-source result is retained.
`TensorBridge.rope_backward_kernel_fp32` combines actual input bytes, Tensor
reads, source execution, all Q/K words read from one
completed byte heap, all four bytes per coordinate, and the full exterior byte
frame. The regional-to-physical interpretation has initial-memory and individual
source-store proofs. Every active source store belongs to the proved footprint;
all coefficient backing bytes, odd tails, allocation prefixes and suffixes are
preserved. The little-endian codec reuses the frozen fast RoPE proof.

The Tensor application uses shared metadata/storage/indexing laws. The kernel
boundaries are
primitive FP32 instruction semantics under the selected configuration, the
source-specific FMA lowering, and correspondence of Python with the readable
typed quote. Source success, all outputs, and the frame are conclusions, not
premises. Neither public endpoint assumes local real exactness or numerical
replay results. Python host preparation, returned-view identity, compiler
internals and GPU-to-primitive conformance are outside these kernel endpoints.
The prior host theorems remain separately available in `Proofs/HostContract.lean`,
`Misc/HostCorrect.lean` and `Misc/HostFPEquiv.lean`; no new host proof was added.

Validation includes targeted Lean builds, the combined `InterfaceChecks`
standard-base axiom/noncircularity audit, the existing 12 CPU contract paths and
12 rejections, and reproduction of the original strided-coefficient defect.
`check_kernel_edges.py` adds 18 actual-source layout cases for independent/zero
coefficient strides, storage offsets, readonly aliases, single empty head sets,
all outputs and frame; it also records branch, nonunit and fused-rounding
counterexamples. `check_cuda_lowering.py` extracts the selected JIT and checks
all ordered Q/K store graphs in four offline Triton 3.7.1 profiles: forward and
backward on SM80 (`D=7,Hq=3,Hk=1`) and SM90 (`D=34,Hq=5,Hk=3`).
These finite CPU/compiler checks support the explicit boundaries; they are not
universal numerical premises. No GPU was executed.

The public Real/native entries are independent of TensorBridge. Supporting
Proofs do not import reader entries; combined reader audits are in the
`Misc/InterfaceChecks.lean` file. TensorBridge retains its complete admitted Tensor
statements and reuses the same source execution foundation. This IO-format
pass changes no Python, kernel source, contract, floating primitive, GPU result
or historical validation evidence.


Public FP entry status, 2026-10-21: **complete source pair**.
`Kernels.indexShared` is a complete directly readable candidate. It reuses the
already computed `cos_offsets` and `cos_mask` in the Q/K index and mask tiles,
removing repeated lane-vector generation and half-width comparisons. Both
source branches, coefficient strides, floating instructions, casts and all four
stores remain intact. This is an index common-subexpression elimination
candidate; no measured speedup is claimed.

`FPEquiv.source_candidate_equivalence` compares the original and this candidate
as actual complete native FP32 grids with `≡[[]]`. `Proofs/IndexCSE` proves
full-state execution equality and access-trace safety for any scalar backend,
and `Proofs/ProgramPair` lifts them across every original scheduled row. The
shared `ReadProgramIO` judgment includes both safe successful executions,
every Q/K result read (including odd untouched tails), and both exact exterior
frames using the original smaller writable footprint. All original tagged
payloads, nonfinite words, empty cases, independent coefficient strides and
head counts remain admitted. The proof changes no floating operation, so it
uses no experimental numerical rule and creates no numerical-validation debt.
The independent `Correct.lean` adjoint theorem and retained optional native and
Tensor applications remain available.


可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/InterfaceChecks.lean`, `Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。
