# Persistent matmul signed-integer checkpoint

The integrated ten-module checkpoint proves source-expression integer agreement
under an explicit sufficient domain. It does not change the mathematical public
contract or certify native numerical GEMM. `Proofs/NativeAudit.lean` exposes all
76 checked axiom outputs (three axiom-free). Root reviewed the sources, domain,
reached-state induction and artifact hashes before integration.

## Source and width evidence

The unchanged upstream source is revision `603e28a`, SHA256
`4537e22452bcb9df7844521498b61232772d7ae44e9aa3220c38c2b341f2152c`.
The two existing original-AST compilation receipts use Triton 3.7.1, CUDA sm90,
warp size 32, BM128, BN256, GM8, SMS132 and eight warps. The fp16 profile uses
BK64/stages3; E4M3 uses BK128/stages4. Both signatures leave M/N/K and all six
strides dynamic signed i32. They do not specialize these runtime dimensions.
Their recorded shared-memory footprint is 49,152 bytes per CTA, with one CTA.
No GPU execution is involved.

`Proofs/native_width_evidence.json` pins the exact original TTIR hashes and records integer
operation lines, pointer operations, original LLVM hashes, signed-extension
evidence and the installed frontend/backend rule files. The two TTIR bodies
use signed i32 for every source integer scalar/vector computation and scf.for
induction value. LLVM global GEP indices sign-extend i32 to i64.

| Source family | Actual integer operations | Checked bridge |
|---|---|---|
| cdiv(M/BM), cdiv(N/BN), cdiv(K/BK) | i32 add of block-1, signed division | `NativePrefix.ceil_exact` |
| tile distribution, exact flat trip count | i32 products, signed quotient/remainder, comparison, conditional +1 | `NativePrefix.agrees` |
| tile_id=PID-SMS, ki=-1 | signed i32 subtraction and literal; negative values retained | `NativePrefix.agrees`, `initial_tile_step` |
| loop induction and K reset | i32 +1, count-1, equality, select; both eager arms fit | `NativePrefix.loop_induction_step`, `NativeControl.next_exact` |
| persistent tile update | conditional i32 +SMS; full original iteration count | `NativeLoop.agrees` |
| grouped schedule, including short last group | signed div/rem, products, subtraction, minimum, addition | `NativeGroup.agrees` |
| coordinate caching | recompute only at actual ki=0, retain otherwise | `NativeCoordinates.agrees` |
| row/column/K lanes | i32 tile-origin product, arange add, signed clamp comparison/select | `NativeAddresses.lane_exact`, `clamp_exact` |
| A/B offsets | two i32 products followed by their i32 sum | `NativeCoordinates.load_addresses` |
| K-tail mask | signed K-ki*BK subtraction and signed lane comparison | `NativeIteration.load_tail_mask` |
| C offsets | **two separate i32 products and two pointer advances** | `NativeIteration.store_products`, `NativeAddresses.two_addresses_exact` |
| C mask and last-K branch | signed comparisons, boolean and, equality | `NativeIteration.store_mask`, `last_k_branch` |
| global pointers | sign-extend32→64, element scaling and unsigned64 address addition | `NativeAddresses.address_exact`, `two_addresses_exact` |

The C expression is not modeled as an invented native i32 sum. Its two
products are checked separately and its two address advances use BitVec64.
The candidate's reordered output-row product also has a direct modular
BitVec commutativity proof. Division/remainder operate only on nonnegative
values; the negative initial tile is advanced before either operation.

## Explicit sufficient domain

Let L=2^31, R=ceil(M/BM), C=ceil(N/BN), T=R*C and Q=ceil(K/BK).
`NativeDomain.Admissible` requires positive M/N/BM/BN/BK/GM/SMS and the
following plain arithmetic conditions. K=0 remains allowed.

- Every dimension, block/group/SMS constant and nonnegative stride is <L.
- M+BM-1, N+BN-1 and K+BK-1 are <L, covering cdiv intermediates.
- T, GM*C and Q*(T/SMS+1) are <L, covering scheduling and the complete loop.
- R*BM, C*BN and Q*BK are ≤L, covering all raw lanes before masking.
- `(M-1)*sam + (Q*BK-1)*sak < L`.
- `(Q*BK-1)*sbk + (N-1)*sbn < L`.
- `(R*BM-1)*scm + (C*BN-1)*scn < L`.

The last three conditions also cover evaluated inactive tail addresses. The
C bound is deliberately sufficient rather than necessary. These are
shape/stride conditions, not an assumed execution, safety theorem, prepared
state or result invariant. Every source-i32 arithmetic agreement follows from
them and the inductively derived control/coordinate values.

The pointer lemmas additionally state the explicit physical arithmetic bound
`base + elementBytes*offset < 2^64`, or its two-advance C form. They prove
no unsigned64 wrap; they do not assert pointer allocation or provenance.

## Nonempty recorded-profile witnesses

Both fp16 and E4M3 witnesses have M=2176, N=2304, K=129; A row stride129,
B/C row stride2304 and inner strides1. Their exact wrapper configurations
are BM128/BN256/GM8/SMS132, BK64 or128, eight warps and stages3 or4.
All domain inequalities are discharged in Lean without native_decide.
The 17×9 output tiles give PID0 two persistent tiles, the final group has one
row, and K has a partial final block in both profiles. Host grid is132.

The installed frontend's numeric arange/power-of-two, tensor size≤2^20,
dot K minimum16/32, and warp checks are proved for these configurations.
The exact wrapper grid is positive and ≤132; 8×32=256 threads is within the
numeric 1024-thread cap. The witnesses also prove unclamped contiguous M/N
coordinate ranges with block-aligned starts. These facts are narrower than
a formal native-configuration admission theorem.

## Remaining work and boundaries

1. The modules prove a handwritten, source-mapped BitVec expression/control
   projection. They do not prove a refinement theorem for a formal Triton/TTIR
   interpreter, the entire typed Lean source, native memory effects or GPU
   execution. Source-to-projection correspondence is recorded/reviewed here,
   not supplied by a verified importer or compiler.
2. Native compilation/configuration admission beyond the two recorded sm90
   profiles and SMS132 is unproved. Numeric frontend checks do not encode all
   backend/resource limits, host/device agreement or resource availability.
3. The separately integrated [hint results](hint_scope.md) characterize the
   cached-coordinate promises and prove reached tail counterexamples. Formal
   AxisInfo/compiler soundness remains open. The arithmetic domain permits
   M/N tails; its hypotheses do not silently assume the promises.
4. Physical allocation, provenance, alignment and the mapping of actual
   device pointers to finite flat memory remain outside the arithmetic
   pointer lemmas. Negative strides and alternative runtime i64 ABIs are
   outside this signed-i32/nonnegative-stride domain.
5. Correctness of compiler transformations, software pipelining and additional
   LLVM/PTX intermediates is not established by source-expression agreement.
6. Native FP/IEEE numerical correctness, concurrent launch semantics,
   official independent replay and final corpus acceptance remain separate.

The existing exact-real/finite-flat and structural-FP public results remain
unchanged and retain their stated scope. This is a compiled integer-domain
checkpoint, not complete original-Triton-port acceptance.
