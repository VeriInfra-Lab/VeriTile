# Independent cosine revision 1 review

Reviewer: `/root/mul2_independent_review`. Producer: `/root/matrix_transpose_acceptance`. Review and acceptance increment: **0**. This is peer evidence for root's integration decision, not root approval and not completion of the new Real / FP32 / BF16 checklist.

The frozen candidate is `/tmp/veritile-standardization-20261015/cosine-native-finite-acceptance-revision1-wip`, receipt SHA-256 `9b62ea3b598a6ac164f91333e2fa3e98cb634cf020e83247014687479bc0433a`. No substantive blocker was found within the reviewed original-source finite scope. Host and kernel numerical correspondence remains explicitly unvalidated.

## Source, host, and numerical boundaries

The original Python kernel has five statements. The task-local Lean syntax preserves program axis 0, dynamic BLOCK, signed comparison, the masked input load, explicit conversion to FP32 before cosine, and the masked output store. Independent replay of the source translator accepted the unchanged source and rejected five mutations. This is an audit of a Python AST translator, not a verified Python parser.

The wrapper allocates `empty_like(A)` before querying `numel`, evaluating host math, dividing for the grid, launching, and returning that exact allocation. There is no explicit CUDA, rank, contiguous-layout, or same-device guard. Eleven original-wrapper AST traces confirmed this order, including allocation failures and empty input. Ten real CPU metadata cases covered FP32/BF16 scalar, transpose, offset slice, broadcast, and empty offset inputs; these checks did not execute the GPU kernel.

Host BLOCK is based on integer-to-binary64 nearest-even conversion, binary64 sqrt, represented-value ceil, and Triton's six-shift next-power operation. The pure finite model does not replace this with Real.sqrt. At `n = 0`, BLOCK is zero and `cdiv(0, 0)` fails after allocation, before JIT setup. At `n = 2^52 + 1`, native rounded sqrt produces ceil `2^26`, although the mathematical sqrt ceiling is larger. At `2^40` the frontend admits BLOCK `2^20`; at `2^40 + 1` it rejects BLOCK `2^21` for tensor shape. At `2^63 - 1` it rejects BLOCK `2^32` at the i32 scalar conversion before the arange size check.

The concrete `HostMath.integer64` and `HostMath.sqrtSignificand` native-conformance obligations bind **all 246 post-allocation cases**, including zero and later binder, architecture, BLOCK, and setup rejections. They do not bind the 22 allocation-failure cases, where host math never runs. The 492 host bindings plus 42 kernel bindings produce 534 exact bindings in 17 families. Every binding is explicitly deferred/unvalidated. Independently removing each binding was rejected by the unchanged exact-link validator. The pinned floating environment is FE_TONEAREST; producer's 210943 comparisons and reviewer's 24 boundary samples are nonexhaustive evidence, not all-domain conformance proofs.

FP32 executes the actual loaded FP32 word through `cos32`. BF16 executes BF16-to-FP32, `cos32`, then FP32-to-BF16 storage conversion; the original source supports this path because its cast is explicit. FP64 also narrows to FP32 in the original source. This review did not add or replay the unrelated full dtype matrix; existing broader records remain preserved and were checked as records.

## Finite requests and memory

Lane arithmetic wraps i32, masks use the signed i32 bound, and pointer formation sign-extends the offset, scales by storage width, and wraps uint64. At `n = 2^31`, BLOCK is 65536 and grid is 32768; all masks are false and successful setup returns the exact allocation and unchanged allocated memory. A regression also checks a negative wrapped offset that re-enters allocation address 32.

The model separates Torch storage base, Torch data pointer, and the actual extracted launch pointer. Mapped allocation metadata, access permission, and owner device are distinct. Unused total affine witnesses on unentered or failed extraction paths do not assert successful extraction. No separate native-to-canonical relocation theorem is claimed.

`WordMemory` requires exact little-endian raw loads and stores, mandatory byte presence, and local frame laws. Missing bytes are not replaced with default values. BF16 decode/store and missing-byte regressions passed. Freshness comes from the ordinary `empty_like` allocation contract, so source-required input/output aliasing is not omitted. All active input words are loaded before the program's scalar/store phase. Every complete pid permutation is represented, and an independent initial-memory plan yields the full byte result and exterior frame; a reverse-schedule regression passed.

Invalid access, missing byte, undefined conversion, and ownership conflict are mathematical observations. The candidate does not claim native traps, rollback, race winners, or postfailure memory for them. The compiled C checks covered 3 exact signed-grid parser cases, 9 exact i32 extraction cases, and 174 pointer comparisons at BF16/FP32 widths. The grid-parser overflow tests are infrastructure evidence only: oversized cosine BLOCK compilation rejects first, so no reachable overflowing-grid cosine case is inferred.

## Independent verification and provenance

All nine candidate modules independently rebuilt with zero warnings: 48 clean axiom checks, 7 noncircular checks, and 9 module audits. All nine independent object hashes equal producer objects. Twelve independent Lean boundary regressions passed their axiom audits. Review dependencies were captured before the first independent build/import; 201 project source/object pairs and 37 external objects matched producer and remained unchanged. Producer's capture was before its final rebuild, **after iterative development checks**; this review does not misstate that timing.

Selective native replay used exactly FP32 and BF16, eight extents each, with two offline default-profile backend compiles. All ten raw TTIR / TTIR / TTGIR / LLVM / PTX comparisons matched after recorded path and PTX debug normalization. There were no GPU launches or large allocations.

The actual frozen receipt was used in isolated adapter fixtures. Twenty-one tamper checks rejected wrong pins, changed modules/source/import/raw IR/host probe/scope, conflicting prior pins, producer self-review, wrong review subject/status/path, omitted host/kernel bindings, and altered callback/rule. The synthetic root-review fixture is explicitly not approval. Whole-queue equality passed in the isolated additive fixture; actual repository integration remains root-owned.

Final verification rehashed all 651 frozen artifacts, all nine module sources, all nine objects, and dependencies. Nine currently retained old source/documentation files were snapshotted; the producer candidate is additive and contains no replacement for these paths. This is not a historical Git comparison. One reviewer final-integrity harness attempt failed because external dependency keys end in `.lean`, not `.olean`; its script/log remain preserved, and corrected attempt 2 passed. No producer or repository file was edited.

The broader original-source finite evidence is not the new three-mainline acceptance checklist. Real model coverage, exact three-mainline mapping, repository queue/scope integration, and final acceptance accounting remain root's responsibility. No new task was started after the pause clarification.
