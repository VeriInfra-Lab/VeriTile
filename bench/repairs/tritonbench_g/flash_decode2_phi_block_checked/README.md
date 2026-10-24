# flash_decode2_phi_block_checked

Minimal input-contract candidate for the independently confirmed Phi `tl.arange` hard-limit defect. Selection and independent acceptance are owned by the root review; this sibling does not change the original source or its evidence packet.

The sole Python change rejects `BLOCK_DMODEL > 1048576` immediately after the original `batch == 0 or head_num == 0` return. Thus the exact admitted domain is the original contract and `(B = 0 or H = 0 or next_power_of_2(D) <= 2^20)`. Earlier original metadata limits, including `BLOCK_DMODEL <= 2^30`, still apply to empty calls. Both zero-batch and zero-head behavior are preserved, including admitted empty widths above the new active limit.

The kernel body, launch arguments, grid, operation precision and all other Python AST nodes are unchanged. This candidate addresses only the `tl.arange` element-count limit. It does not add a grid-product restriction or claim launch-resource, driver or runtime compatibility for every admitted shape. Capturing a width at the API upper bound is not a claim that such a kernel compiles or fits every GPU.

`Correct.lean` exposes the complete selected Phi kernel over the two-dimensional grid, with arbitrary traversal permutations, every logical output, the independent Real stage-2 formula, and the memory frame. For row `r = b*H+h`, `K = ceil(length[b]/block_seq)`, the target is

```
Out[r,d] = sum(j<K, exp(Log[r,j])*Mid[r,j,d]) / sum(j<K, exp(Log[r,j]))
```

The public `Correct.kernel_real` takes source geometry, a complete row schedule and a shared `ProgramIO.ImplementsOn` judgment over logical summary/length inputs. `IO.lean` lists actual S strides, dynamically used input masks, the masked source, complete output windows and empty scratch. Its separate `capacity` is a backing-window extent; it does not constrain counts to source S. `PhiIO.exists_capacity` proves every original finite-grid length assignment has a suitable extent, including zero lengths and Q=0. The Real domain retains positive reached counts and arbitrary finite Real summary operands; unused input payloads are unbound. The block-width bound covers every logical output coordinate. This is the stage-2 summary API and does not assume stage-1 provenance.

The optional `FPEquiv.kernel_native` theorem in `Misc/NativeCorrect.lean` uses the shared `ProgramIO.ObservesOn` judgment for complete native execution, every logical output word and the exterior typed-memory frame. Inputs retain arbitrary original symbolic terms with identity encoding; only outputs are evaluated under the original instruction model and compiler schedule. No positivity, Q>0, count≤S or leaf-only input restriction is added. `Misc/TensorBridge.lean` retains actual `Torch.FP32.logicalValue` output readback after the source's little-endian four-byte stores at the caller's output pointer, all bytes outside the output element span and all bytes of the admitted input backing storages; Mid/Log aliases remain allowed. The native target is independently defined by the source-ordered scalar primitive graph, with structural (payload-independent) compiler fusion and operand-order choices. All 32-bit FP32 payloads are admitted. The primitive operations and supported structural lowering relation are model/API boundaries, not extra numerical-experiment obligations.

The bridge premises are actual argument handles with ordinary Tensor API laws, live input assertions, caller memory and a complete row schedule. Its lengths come from actual signed INT32 caller bytes. There is no successful host invocation, returned-object, allocator, precomputed output or reference-equality premise. `KernelSafety` derives active and padded formed address bounds, signed length arithmetic, storage bounds and exact uint64 pointer addition. `Contract.arange_bounds` exports the new active hard limit.

Validation reuses all 42 original CPU source-interpreter cases and checks six real-wrapper boundary captures/rejections. It includes complete Python AST identity after deleting the sole guard and unchanged JIT AST identity. No GPU execution is claimed. Source/receipt pins and original-25-file preservation are recorded in the candidate evidence packet.

The prior raw `real`, `native`, `source` and `empty` statements keep their names in `Misc/RawResults.lean`. The complete Tensor interfaces remain unchanged in `TensorBridge`. `Proofs/PublicIO.lean` reuses complete grid results and the shared actual-source access-safety proof; only independent input bindings and logical formulas are connected. The primary modules do not import the bridge. `Misc/SourceAudit.lean` audits helper results directly; root `Misc/Audit.lean` includes the public and Tensor audits. This format update changes no Python, kernel source, contract, numerical law or experiment status.

## Actual source pair (2026-10-21)

`FPEquiv.kernel_equivalent g width order coverage` now compares the complete
original masked source and `Candidates.phiSharedBase`. No distinct historical
JIT candidate existed. The new directly related optimization computes
`offs_logic = batch*(H*S)+head*S` first and reuses that base in
`offs_v = offs_logic*D+offs_d`. It removes one scalar natural multiplication
and one addition; no measured speedup is claimed. Both complete sources are
ordinary readable quotations, with transparent reuse of the original source.

`Proofs/SourcePair.lean` proves exact natural address equality and complete
execution/safety preservation for arbitrary structural Algebra. The entire
dynamic floating loop, FP32 load/conversion boundaries, padded-lane masks and
output mask are unchanged. `Proofs/IOPair.lean` retains every original launch
setting and every permutation of the complete grid. Shared
`ProgramIO.EquivalentOn` supplies both safety conclusions, both successful
executions, active typed output equality and both exterior memory frames.

The pair's domain is the existing native `Fits` domain and `D≤BD`, with arbitrary
symbolic input terms/all FP32 words. Zero lengths, Q=0, dynamically used counts
above S, independent backing capacity, empty batch/head grids and unbound unused
input payloads remain allowed. No positivity, finiteness or leaf-only restriction
is added. The original primitive/compiler observer can be applied afterward;
no floating rewrite or experimental numerical rule is used. The independent
Real formula and its existing Positive premise remain unchanged. Single-source
native observations remain ordinary auxiliary results in `Misc/NativeCorrect.lean`.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
