# Kernel-only context_attn_mistral revision

This revision proves the complete shared FP32 source body. The selected Python is the existing `context_attn_mistral_repaired.py` (SHA256 `d22065e999a99d827fdcf62053df5b992a68ce6bdb7222bb61c28d3a5c9bf865`). No Python, guard, historical theorem, ledger, Lake, or shared core source is changed. The old conditional results remain available under their original hypotheses and reviewed domains.

The source and Real entries are `Kernels.lean`, `Contract.lean`, and `Correct.lean`, under namespace `VeriTile.Bench.Repairs.ContextAttnMistralKernel`. `Misc/NativeCorrect.lean` retains the complete native observation as an ordinary auxiliary theorem. `FPEquiv.lean` proves `original ≡[[]] blockStartReused` for the complete selected arbitrary-job interface, including aliases, duplicate writes, every output and both exterior frames. The existing `scaleFirst` candidate remains available historically; its floating reassociation is not needed for this exact index optimization.

- `RealIO.lean` declares the selected source, five named finite input channels, strides, masks, complete ordinary grid, and independent finite-sentinel formula. `Correct.kernel_real` uses the shared `AlgorithmProgramIO.ImplementsOn` judgment. Its proof connects actual source access safety, successful whole-grid execution, every active output and the exterior cell frame.
- `NativeArrays.lean` defines six named typed array views, including the initial output view, finite source-free read windows and alias-aware ordered scatter. `NativeIO.lean` declares the actual source invocation schedule and the fixed native observer. The ordinary `kernel_native` theorem in `Misc/NativeCorrect.lean` uses shared `ArrayProgramIO.ObservesOn` for arbitrary jobs and all owner observations; its original full name, statement and proof are preserved.
- `Proofs/RawRealResults.lean` and `Proofs/RawNativeResults.lean` retain the exact previous flat-memory and raw-state statements. `Proofs/RealOriginalBinding.lean` proves original admission supplies the new finite inputs and the new formula equals the previous logical Tensor formula. Native snapshot/grid transport and output tag preservation retain arbitrary input Exprs, aliases, repeated jobs and overlapping stores.
- `TensorBridge.correct` and `TensorBridge.fp_equiv` retain the complete checked Tensor applications, including physical output words/bytes, untouched input/metadata words, exterior-byte frame and native pointer correspondence. All existing source-address, coverage, empty-grid and window lemmas remain. Supporting modules do not import either public reader.

## Retained checked Tensor domain

Q/K/V/O are contiguous FP32 tensors with D in {16,32,64,128}. BM=BN=128; the selected launch uses four warps through D64 and eight at D128, with one stage. Grouped heads divide exactly. Starts and lengths are signed int32 metadata; nonempty intervals are disjoint and within the packed token storage. Output is separate from all input/metadata storage. Read/read aliasing remains permitted.

The checked contract intentionally permits underlaunch: only rows within `ceil(max_input_len/128)*128` are written. Gaps, underlaunched rows, inactive tile lanes, and all other bytes are framed. Arbitrary Python windows retain the existing clamp to [0,2^63-1], whose mask equivalence uses the checked bounds. The existing early-return domain for zero batches or zero blocks remains unchanged. The Lean empty-grid theorem is stronger and needs no data-allocation admission; this is not a claim to execute a host wrapper.

## Source and independent results

The AST audit compares the complete repaired and original kernel, erasing exactly eight signed-int64 widenings and normalizing one equivalent window predicate. The exact FP32 ABI scale bits are D16 `3e800000`, D32 `3e3504f3`, D64 `3e000000`, D128 `3db504f3`.

The source keeps finite -1e9 masks, both equality guards, the probability cast, and the live PV accumulator. The Real reference includes the finite weights of padded zero-V positions. It is not hard masking, an exact FP32 result, or an error bound. The focused all-masked controls (window 0 and a large negative window, one V=7 followed by 127 padded positions) have Real result 7/128 and FP32 interpreter result 0, due to the source center guard and exponential underflow. This is an interpretation distinction, not a source defect.

`Reference`, `Execution`, `Program`, and `Grid` prove initialization, each complete loop iteration, epilogue/scatter, successful source execution and memory-threaded whole-grid equality with the separately defined recurrence. The public theorem requires no `SourceGraph.Binding`, caller-supplied plan, source-success hypothesis, reduction/dot equality, or final-output premise. Ordinary Tensor laws, role/address bindings, initial signed metadata observation, the checked input contract, and one-word byte-decoder laws remain explicit interfaces.

## Fixed native profile and the window=1 finding

The native graph uses fixed SM90 K8 WGMMA chains, physical producer/consumer routes, exact per-consumer reduction trees, observed FMA/operand ordering, delayed raw-word equality guards, and structural clear/live selectors. No raw word or Real literal payload selects a plan, route, tree, or clear/live mode.

The real Triton binder specializes integer one to constexpr. For window=1, this changes the emitted sum parenthesization. The historical generic tree alone therefore does not describe that specialization. This revision adds 24 max descriptors and 20 complete sum templates, with fixed row/owner tables. A max descriptor stores the exact emitted code and a separate static annotation listing source lanes replaced by mask constants; Lean proves their combined source-lane provenance is a permutation. The native evaluator executes only the emitted code. Sum renaming is a proven permutation. The static audit checks the generated Lean trees and tables against every saved PTX replica. No floating addition reassociation is used.

The source initializer zero plus negative infinity is the sole scalar fold in `consume`. For window=1, the integer mask allows only the diagonal: every omitted source lane is statically masked, as checked by the descriptor audit and `window_one_offdiagonal`. The emitted maximum code retains every PTX max node, including runtime max(S,S), and its exact literal constant leaves. No max idempotence, reassociation, or runtime constant folding is assumed. Masked probability leaves in a sum template may be renamed only within the same producer, preserving their exact evaluation and addition order. These are explicit source/compiler model boundaries, not deferred numerical experiments.

Twenty actual-wrapper/native-binder offline compilations cover all four widths with constexpr-one, aligned and unaligned i32 runtime windows, and aligned and unaligned i64 runtime windows. All max/sum replicas, scalar operand orientations, K8 coverage and live PV seed sites are checked. Shared-memory requirements are 16384/32768/65536/131072 bytes for D16/32/64/128. No resource exclusion was observed in these SM90 artifacts. These finite artifacts support this selected profile; they do not prove all compiler implementations, all possible specialization dimensions, native device launch success, synchronization/runtime behavior, or compiler correctness.

## Numerical and execution boundary

`PrimitiveLaws` has only FP32 literal/add/sub/mul/div/max/exp2/FMA and single-output fixed m64n{16,32,64,128}k8 TF32 WGMMA ports, including the explicit clear/live accumulator selector. Instantiating these ports against hardware still needs the local numerical conformance evidence. No new primitive numerical experiment or GPU kernel execution is claimed.

The wrapper was executed with CPU storage and a CUDA metadata/device/submission facade to recover the unchanged actual launch arguments. JIT `_pack_args` and `native_specialize_impl` and the SM90 compiler were real. This is neither a full native launcher nor a CUDA execution. The runtime, concurrency, allocation implementation, and universal compiler mapping remain outside this kernel-only theorem, separately from numerical primitive laws.

The task-local evidence packet records source preservation, current build receipts, repository import/source/object pins and `.ilean` import consistency. Reused imported objects are explicitly distinguished from the freshly compiled new task modules; receipt and import audits are not a claim of rebuilding the entire compiler or Mathlib.

## Public IO and preserved raw scope

The Real headline has the original geometry/capacity conditions and finite metadata domain. It covers the complete selected ordinary grid, including underlaunch. The retained `raw_kernel_real` additionally exposes the original full flat allocation transport and entire bounded input-allocation preservation. Both use the same canonical quoted source and independent finite-sentinel formula; no second source is introduced.

The retained native auxiliary theorem permits arbitrary job lists, arbitrary Expr-valued decoded inputs, repeated jobs, output collisions and Q/K/V-output aliases. Only output separation from start/sequence metadata is required. Finite read windows constrain actual access safety; successful execution, output binding and the complete exterior frame hold independently of capacity premises. The retained `raw_kernel_native` keeps exact all-cell recurrence equality, final metadata and native observations at every region, offset and owner.

`Proofs/NativeSourceSafety`, `NativeLoopSafety` and `NativeProgramSafety` prove safety along the actual source states, including the signed loop and both equality guards. `NativeAccess` connects the finite windows to that proof for the actual arbitrary-job schedule. The existing scalar instruction laws and fixed native profile are unchanged. This format revision adds no numerical assumption, replay result or GPU measurement.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.

## Complete program pair (2026-10-21)

`Kernels.lean` contains two complete ordinary `triton` blocks. The candidate
uses the already computed `block_start_loc = BM * start_m` in `offs_m`, replacing
`start_m * BM`. This removes one repeated integer multiplication per CTA; no
performance measurement is claimed. The original is exactly the existing bounded
machine-index projection. Its accepted signed-width transport is retained, and
`SourcePair.original_preserved` proves equality with the full historical FP32
program. `Kernels.lean` does not import proof modules.

`PairIO.lean` binds both actual sources to the same ABI, regions, layout, jobs,
finite access windows and `NativeArrays.metadata` domain. The public
`block_start_reused_equiv` theorem retains arbitrary payloads and arbitrary ordered
job lists, repeated writes, output collisions and Q/K/V-output aliases; only
output/metadata separation is required. Exact reached-state equality transfers
source safety, success, every typed final output and both complete exterior frames.
The proof uses integer commutativity, no floating rewrite, experimental rule or
finite-word restriction. `Correct.kernel_real` retains its separate independent
finite-sentinel weighted-sum formula and original geometry/metadata conditions.
