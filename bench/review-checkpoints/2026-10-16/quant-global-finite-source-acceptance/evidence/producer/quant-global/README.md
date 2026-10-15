# Quantize global: complete finite source acceptance candidate

This is a producer candidate for independent root review. It does not accept the
original program, establish native numerical correctness, or claim GPU execution.
The original repository and all earlier frozen producer packages were left
unchanged by this work. The original Python SHA256 is
`163f6e73234587c1d05763ed60ccc1ac4ca7d7d384f18b7af17399509205032f`.

## Public contract and execution

`QuantizeGlobalFinite.quantize_global_full_source_outcomes` is the sole new
public `specification`. Its API domain is ordinary dense strided Torch tensors
on CPU/CUDA, valid finite Torch shape/storage metadata with numel < 2^63, Torch
2.13.0+cu130 commit cf30153c4c131c8164ee7798e5022d810682e2cb, default floating
dtype **float32**, Triton 3.7.1, CUDA sm90. Other default floating dtypes, sparse
or quantized layouts, other devices/targets and arbitrary Python objects are not
claimed covered. No previously admitted dtype route has been removed.

The contract takes arbitrary explicit primitive outcomes and covers every
source-ordered rejection, an i32-offset or source-linear-view hazard, an actual
active undefined conversion, or actual returned execution with independent
logical and physical results. It has no Safe/defined-conversion premise, no
successful launch or final-output premise. Readiness is conditional: typed
input/metadata/allocation ownership is needed only when admission actually
reaches a safe-address kernel route. It never asserts that admission succeeds.

The producer executes abs on every logical strided-view cell, the invocation's
whole-tensor maximum, unsqueeze, reciprocal, output allocation and CUDA
assertion in source order. The kernel uses **linear** cells from x.data_ptr(),
which can differ from the logical view traversal. Every active load has its
actual dtype; unavailable or wrongly typed cells never default to zero. Every
program loads its scalar inverse; every active lane executes promotion, the two
parenthesized multiplications, llrint64 and low-eight-bit storage. Padded lanes
have no observable store. No inactive undefined arithmetic value is invented.

The actual source configurations are BLOCK_SIZE1024 / warps4 / stages3 / ctas1
and BLOCK_SIZE2048 / warps4 / stages1 / ctas1. The theorem executes an arbitrary
finite list of trials from these configurations followed by the final selected
launch. Each successful run consumes the evolving memory from previous trials;
this is not a collection of unrelated per-block facts. There is no output reset
in the original autotuner configuration. Failed primitive/launch outcomes have
no returned-memory or rollback claim. Installed Autotuner probes cover cache
miss/hit, repeated benchmark calls and first-config OutOfResources; they use
instrumented kernel invocations, not a GPU.

## Complete dtype and first-error inventory

All22 canonical ordinary dense scalar dtype families are inventoried, including
bool, standard integer/float, complex, five float8 families and packed float4.

* CUDA bool, uint8, int8, int16, int32, int64, float32 and float64 reach the typed
  kernel. Bool/integer inverse is fp32 under the pinned default. Integers promote
  to fp32; float32 stays fp32; float64 stays fp64. Both multiplications use that
  compute dtype. All successful stores narrow signed64 to int8.
* Bool uses actual Triton signature `*u1`: its loaded i8 byte is normalized by
  nonzero. All256 storage bytes are represented. CUDA bool abs is supported;
  CPU bool fails at abs.
* CUDA uint16/uint32/uint64 and all float8/packed-float4 families fail at abs.
  CPU abs supports the four e4m3fn/e5m2/e4m3fnuz/e5m2fnuz families, but their
  nonempty maximum is unsupported. Empty maximum is checked before that error.
* float16/bfloat16 host primitives succeed; the unchanged kernel compilation
  fails at llrint because only fp32/fp64 overloads exist. There is no fabricated
  promotion to fp32.
* complex32/64/128 abs produces fp16/fp32/fp64; those host stages succeed on
  CUDA, then Triton argument specialization rejects the complex x dtype.
* Otherwise-supported CPU tensors reach the CUDA assertion only after output
  allocation succeeds. Primitive allocation failure precedes that assertion.
* Empty reduction happens before unsqueeze/reciprocal/allocation. A rank-zero
  input has one element, but `torch.empty(*x.shape)` has no size arguments;
  the missing-size error happens **after** reciprocal.

The finite admission interpreter records first stage/reason, not exact native
error wording. Runtime errors are trusted primitive outcomes at each actually
reached source call. Earlier rejection cases make no equality claim about
discarded numeric values and therefore require no numeric value callback.

## Finite addressing, physical ownership and actual byte execution

N <= 2^31 guarantees every reached padded offset is signed32-representable for
both configs. At N=2^31 the Python n_elements specialization is signed64 while
all reached offsets still fit signed32. For N>2^31 the source reaches offset
2^31, whose signed32 value is negative; this is an explicit source hazard.
Logical view bounds and the later linear N-element source span are separate.
Transpose/strided-slice witnesses retain their different traversals; an
expanded scalar view can be valid for abs/max and unsafe for the linear kernel.

`PhysicalBinding` explicitly supplies the actual unsigned64 x.data_ptr(),
allocation bases, storage byte offset/extent, capacities for input/max/inverse/
output, base+capacity no-wrap, alignment, pairwise physical interval separation,
input word-to-byte relation, actual arbitrary torch.empty payload and device
read/write permissions. Input/max/inverse retain the input CUDA ordinal;
`torch.empty(device="cuda")` output retains the current default CUDA ordinal.
Cross-device inputs are not silently rewritten: successful access requires the
actual launch-device permissions; failure stays at the runtime boundary.

The returned output is contiguous int8 with x.shape and storage offset zero;
returned absmax has shape[1], stride[1], input dtype/device and aliases the
rank-zero maximum storage created before unsqueeze. The inverse is a distinct
fresh region of its actual compute dtype. Allocation mechanics are trusted;
source-visible metadata, ownership and capacity facts are explicit.

`output_store_commutes` proves each actual logical output write equals one
concrete byteWrite at output.base+index for arbitrary intermediate memory.
`lanes_physical`, `program_physical`, `grid_physical` and the all-trials theorem
construct a `PhysicalStores` trace from the actual source execution. The master
contract derives that trace from its own executed result. It is not an assumed
final replay. Its logical/physical frame and input-preservation lemmas apply to
intermediate traces. Concrete input and inverse load lemmas connect typed reads
to physical little-endian words under the same representation.

`host_physical_preparation` separately derives publication of the produced max
and reciprocal with explicit word writes. The output allocation already has its
bound arbitrary payload; the model does not manufacture initialized output
bytes. `PhysicalResult` states exact output/max/inverse bytes, original input
preservation and the raw byte frame outside the owned source write footprint.
The byte snapshot means source-visible allocation memory, not allocator caches,
scratch space, page-table state or a proof of CUDA's MMU.

## Arithmetic and undefined conversions

Numerics parameterizes exact host abs/whole-tensor max/reciprocal, integer casts
and fp32/fp64 multiplications. The maximum callback is bound to **this original
invocation's shape, strides and reduction runtime context**. It may close over
that metadata; it is not a fabricated associative left fold or a universal,
shape-independent NaN-payload result. Signed abs(min) is not idealized into an
unrepresentable positive mathematical integer.

Llrint is a concrete partial function: decode the finite float bit pattern,
round the exact rational to nearest integer with ties even, check signed64
range, and only then narrow to int8. NaN, infinity and rounded out-of-range
values yield no result. The source can generate such values (including an
all-zero input's zero-times-infinity NaN); these are explicit undefined source
observations, never a numerical assumption permitting an arbitrary integer.
No native trap, rollback or deterministic value is claimed on those operands.

`evidence/numerical-authority-delta.json` binds every numeric operation port to
its exact public statement hash, source hashes, dtype, block config, size class,
source operation and raw native IR/receipts. Both multiply sites have distinct
bindings. Concrete literal/bool/llrint/store conformance ports are registered
alongside parameterized host/cast/multiply callbacks. Native whole-kernel
validation remains false. Per-case callback requirements and deferrals form an
exact checked bijection. Earlier rejection, primitive failure and structural
hazard classifications assert no returned numeric equality; they have no
kernel numeric ports. The retained fp32 bridge is valid only on the concrete
defined llrint domain and preserves earlier selected-fp32 results without
totalizing their exceptional callback.

## Evidence and review status

* The final unchanged-kernel probe contains52 exact signature/config/index-width
  attempts:32 successful eight-route compiles and20 expected fp16/bf16/complex
  rejections. It saves TTIR, TTGIR, LLVM IR and PTX plus raw errors and hashes.
  No GPU is available, and no GPU output or performance claim is made.
* CPU primitive probes cover22 dtypes at nonempty, empty and rank-zero shapes.
  They are explicitly CPU evidence. CUDA dispatch facts are reviewed against
  downloaded exact installed-commit Torch CUDA source files with URL/hash
  receipts; CPU results are not relabeled CUDA results.
* The unchanged production-host AST has360 passing instrumented source-wiring
  probes, including all dtype first errors and injected primitive failures.
  Six installed-Autotuner schedules check source configurations, argument/grid
  identity, repetitions and evolving state. These are boundary wiring probes.
* Every Lean attempt saves exact source, raw stdout/stderr, actual command and
  actual exit code. An expensive admission proof-search child was explicitly
  terminated and recorded as exit-15; it was not relabeled successful.
* Final mapped modules are freshly compiled and all pass axiom auditing with
  only propext, Classical.choice and Quot.sound. Existing imported dependencies
  are reused and fingerprinted; this is not a trust-zero rebuild.
* Each final module also exists at its exact repo-relative `.lean.txt` path,
  with an explicit source mapping in the frozen receipt. The overlay changes
  local import names only. Intermediate specifications were explicitly changed
  to theorem helpers before freezing; only the master is public.

Schema2 proposals retain exact complex32/64/128 and fp8e8m0fnu dtype labels.
Those four labels need a reviewed additive enum extension; packed float4 uses
the existing packed_fp4e2m1 label. Producer validation runs the unchanged
validator functions with only those proposed labels added in memory and says
so explicitly. It does not claim canonical acceptance. Root must substantively
review the original-source correspondence, public contract, physical primitive
binding, cases, retained results, numerical delta and dependency receipts, then
integrate the catalog/scope/numerical authority. No repository checker was
modified by this producer.
