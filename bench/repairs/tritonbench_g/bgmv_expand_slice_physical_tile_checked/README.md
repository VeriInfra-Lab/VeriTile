# BGMV expand slice: checked physical ranges and static tile

This separate candidate fixes the two preserved canonical-source defects by
changing wrapper guards only. Its JIT kernel AST is identical to
`bgmv_expand_slice/bgmv_expand_slice_repaired.py`; the original benchmark,
historical proofs, earlier diagnosis and compiler artifacts remain preserved.
The canonical selection, ledger and Lake integration are owned by root review.

For active batch `b`, adapter `a=indices[b]`, and `0<=n<N`, the independent Real
specification is

```
Y_after[b, offset+n] = sum(k<K, X_before[b,k] * W_before[a,n,k])
                     + (Y_before[b, offset+n] if add_inputs else 0)
```

Metadata `-1` skips the complete row. Every other logical Y value, every byte
outside the active output footprint, and the complete X/W/metadata backing
ranges are preserved. The existing Python wrapper returns `None` and retains
the supplied Y object and descriptor.

## Explicit domain changes

| Property | Historical checked source | This candidate |
| --- | --- | --- |
| Output/read isolation | Unequal StorageImpl `_cdata` | Disjoint complete untyped backing byte ranges from `data_ptr()` and `nbytes()`; empty ranges are disjoint |
| Read/read aliases | Allowed | Allowed, including distinct DLPack StorageImpls sharing physical bytes |
| Nonempty tile | No frontend-size guard | `256 * next_power_of_2(K) <= 2**20`, hence `K<=4096` |
| B=0 or N=0 | Returns before launch and metadata content read | Same early return; tile cap applies only after it |
| K=0 | Rejected | Rejected, including empty calls |
| Other wrapper behavior | Checked dtype/layout/device/shape/stride/lazy-view/alignment/scalar/offset domain | Unchanged |

Both W ranks (`[L,N,K]`, `[L,1,N,K]`), both signed index widths (INT32/INT64),
both addition modes and aligned offset views remain supported. Complete backing
range separation is deliberately stronger than separation of only the logical
view. It rejects physically overlapping output/read DLPack imports without
assuming that storage identity determines physical separation.

`B>0` and `N>0` still dispatch grid `(N,B)`, `BLOCK_N=256`, `SPLIT_N=N`,
`CAST_TYPE=True`; the signed source split length is one. The frontend guard
bounds the actual broadcast tile, including padded false-mask lanes. It does
not certify per-device registers, shared memory, launch delivery or all legal
compiler resource configurations.

## Proof endpoints and precision

`Kernels.source` names `Surface.source`, the complete signed quotation:
program-id and arange casts, signed metadata, early return, signed cdiv/range,
branches, masks, loads, reduction and stores are retained. `check_source.py`
compares all 25 non-docstring statement nodes after the checked primary
specialization. It enumerates the representation changes: signed cdiv's
arithmetic expansion, method spelling for a cast, specialized contiguous
strides, and explicit FP32 store conversion.

`TensorBridge.correct` constructively executes the complete shared typed source over
arbitrary Real inputs and proves the independent K-term target at actual
shared Torch FP32 Tensor logical addresses. This is a mathematical projection;
its arbitrary Real reader is not an encoder or finiteness assumption.

`TensorBridge.fp_equiv` executes that same source over a scalar expression graph. X is
narrowed FP32 to FP16; multiplication and every selected reduction-tree node
are FP16; the reduction is widened to FP32, followed by optional FP32 addition.
All K-padding zeros remain in the selected tree. Inputs come from physical
bytes, including signed metadata decoding and original output words. The
actual final source cells determine stored FP32 words and all four bytes.

Only five pointwise instruction conformance laws appear: FP32-to-FP16,
FP16-to-FP32, FP16 multiply, FP16 add and FP32 add. Their arguments have the
actual 16/32-bit widths, without finite-value restrictions. The reduction plan
is shape/dtype/axis/output-coordinate selected and independent of input values;
there is no whole-reduction, whole-output, execution or compiler-success
premise. Compiler realization of the selected tree and numerical realization
of the five local ports remain explicit external boundaries. No equality
between native arithmetic and the Real projection is asserted.

`Tensor` uses the repository's actual shared FP32 Tensor carrier and ordinary
API laws. W and integer metadata use a local descriptor/physical-index binding
because the shared library has no half/integer Tensor carrier. `Memory` derives
read-backing frames from the new checked physical ranges. `Safety` proves
active address bounds, all formed padded byte offsets, signed 64-bit offset
agreement, signed cdiv arithmetic, and the static tensor shapes. It does not
create a host/allocator/compiler model.

`Audit` checks standard axioms and noncircularity, including transitive proof
bodies. Pure dimensions, addresses and formula/fold lemmas are reused. The
public theorems are independent of the normalized source and normalized
execution proofs that remain in auxiliary imports; the historical
`CompilerBinding` and physical-separation implication premises are absent.

## Validation and retained evidence

`check_revision.py` passed 14 directed controls: four actual Torch DLPack
output/read aliases rejected before launch without mutation; K4096 admitted
and K4097 rejected for rank3/INT32/add0 and rank4/INT64/add1; both empty shapes
retain their K4097 early return and K0 rejection; two complete CPU source-AST
executions cover disjoint inputs and actual overlapping read backings with an
offset output. Eight controls reject and six admit. CUDA descriptors/context
use the existing CPU adapter; storage, DLPack and byte comparisons are real
Torch operations. These are not GPU or native numerical tests.

The old full suite and its successful K2/K7 SM80 offline compiles are retained,
not rerun. Unchanged kernel AST makes those two small compiler artifacts
relevant to this candidate, but neither they nor the K4096 admission control
certify all admitted compiler plans or hardware resources. No large K4096
resource compilation or broad historical rebuild is part of this candidate.

The evidence directory contains current-source build receipts, source/guard
controls, the preservation check for all 113 baseline files, and a hash-pinned
`delivery.json` for independent review. Reproduction uses:

```
python bench/repairs/tritonbench_g/bgmv_expand_slice_physical_tile_checked/check_revision.py
python bench/repairs/tritonbench_g/bgmv_expand_slice_physical_tile_checked/check_source.py
```

Lean modules are compiled in `final-build-order.json` dependency order with
`.veritile-cache/build-tools/workspace_lean_local_first_v2.py`. No shared core,
ledger, Lake or historical task files are modified.

## Public IO organization

`Correct.lean` now states `gemvIO.ImplementsOn PublicIO.Valid PublicMath.formula`;
`Misc/NativeCorrect.lean` states the corresponding shared `ProgramIO.ImplementsOn`
observation against `PublicNative.formula`. Neither headline exposes a raw
state, execution predicate, final-state existential or per-address memory
formula. `Targets.lean` defines independent logical-coordinate formulas.

`IO.lean` lists the complete batch/output-row source schedule, typed X/W/Y and
signed metadata channels, gathered W windows, optional old-Y input, active
write masks and exact output footprint. X has width BK in the unmasked EVEN_K
branch and K otherwise. The kernel-level Call domain still permits K=0,
including EVEN_K with BK>0, although the unchanged Python contract rejects K=0.
Signed metadata remains -1 or any nonnegative integer with backed gathered
reads; no adapter-count upper bound was added to the kernel theorem.

`Proofs/IOExecution.lean` connects actual source launches to arbitrary typed
initial cells. `AccessSafety.lean` and `SourceSafety.lean` prove bounds along
all actual prefix/guard/setup/range/body accesses. `IOGrid.lean` proves the
complete grid, initial-value behavior of the optional add, and exact exterior
cell frame. `IOBindings`, `IOValues`, `IOSafety`, `IOImplementation` and `IOReal`
compose these into the shared IO judgment. Untouched tags and skipped outputs
are preserved, without canonicalizing arbitrary exterior cells.

The old complete raw statements remain as ordinary theorems in
`Proofs/KernelRealResults.lean` and `Proofs/KernelNativeResults.lean`.
`TensorBridge.correct` and `TensorBridge.fp_equiv` retain their original actual
Tensor observations, all four stored bytes, physical/read-backing frames and
wrapper guards. The native public interface retains arbitrary Expr inputs,
full padded BK products/tree and the same five primitive conformance laws;
there is no added leaf/finite restriction or new experimental assumption.

The IO-format pass uses targeted current-source Lean builds and standard-base
axiom/noncircularity checks. Python, Kernels and Contract are unchanged. No GPU
run or independent native numerical replay was performed in this pass.


Public FP entry status, 2026-10-21: **comparison proof pending**.
`Misc/NativeCorrect.lean` retains the former single-source `fp_equiv`
result as an ordinary theorem under its original namespace, full input domain
and proof. `FPEquiv.lean` records the missing/pending comparison and declares no two-program
FP specification. Selected signed-int64/early-return/full-grid source is not the original candidate Program. The current Native.Expr/add32 instruction ports have no commutativity law; native source implementation alone cannot prove the candidate.
The independent `Correct.lean` mathematical theorem and the complete native
Tensor application are preserved. This gap is a missing program comparison,
not an unproved numerical law; no new numerical experiment is claimed.


可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/Audit.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。
