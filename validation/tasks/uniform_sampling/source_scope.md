# Source and theorem scope

Complete real execution, finite memory correctness, typed source/candidate
equivalence, and successful complete typed execution are proved. Public real
and FP nonempty witnesses passed the final audit, including all 29 axiom gates
and zero floating rewrite assumptions. Corpus acceptance remains separate.

The complete pinned Python file, including the host generator update and all
tests, is imported unchanged from commit `603e28a`. Its SHA256 is
`d2046880647babeb532c22d3fe39e5f9662c52de36ac9a465b6405aab352a4c1`.

The strict `direct_uniform_identity_v1` source comparison passed on the actual
port for all six explicit storage ASTs. The declared fp32 comparison also
passed all 24 production audit gates. This retains the original try/except
helper binding, decorators, all twenty kernel statements, and all four store
arguments. Three checked identity assignments materialize runtime seed,
runtime offset, and the constexpr block adapter; the `to` parameter is named
`upper` in Lean. Final store casts make the native destination conversion
explicit. These checks establish source comparison, not execution correctness
or native numerical refinement.

The source first converts seed and offset to signed int64. It extracts the
low and high uint32 offset words, adds the lane counter index to the low word
with uint32 wraparound, and leaves the high word unchanged. It invokes the
four-counter, ten-round Philox generator. All four outputs are converted by
the original helper and stored at four consecutive B-lane blocks per program,
each with its own `offset < N` mask. The counter index advances B per program,
while the output base advances 4B.

The public source execution contract requires both original runtime seed and
original runtime offset to lie in the signed64 range. Validation of the
generated uint32 counters does not establish the original offset's ABI domain.

`ScalarSpec` gives an input-only four-word specification. Its uniform decoder
preserves both fp32 rounding stages. `SourceDomain` proves complete output
coverage, unique program/field/lane ownership, and signed i32 address bounds
for all launched lanes under `N < 2^31` and legal power-of-two B up to 2^20.
The exact wrapper block sizes are 512 and 1024. `WordArithmetic` connects
actual fixed-width index operations and the fallback helper's modular
negation/subtraction to these formulas, including signed64 boundary inputs.
`Formula` maps every output address back to its proper counter index and one
of the four Philox words. None of these facts assumes successful execution.

`CounterOperands` relates the actual signed64 AND and arithmetic-shift
expressions to the counter words. `RandomDomain` derives each actual Philox
field's successful evaluation from evaluated child operands and the seed
domain; it does not assume a complete kernel execution. `Memory` proves the
exact cell values and frame for the mathematical four-store sequence.
`Algorithm.source_eq` and `Execution.source_exec` connect all 23 expanded
source statements to that independent specification, through the actual
counter preparation, five snapshots, four Philox results, four affine
computations, four offsets, and four masked stores.
`TypedRandom` separately proves the same five-operand Philox evaluation and
canonical uniform fp32 payload in the floating structural evaluator.

The Python helper binding is selected by the original try/except. The installed
Triton 3.8.0 helper uses bitwise complement in the negative branch; the fallback
uses `-x-1`. The formal machine-word identity covers both, including INT32_MIN.
The kernel reaches the 32-bit helper branch. The source file's 64-bit helper
branch remains imported, but is not a second execution path for this kernel.

The offline native probe preserves the decorated kernel AST and actual helper
selection. Its primary matrix uses signed64 seed/offset, signed32 N, fp32 bounds,
six floating destination types, and all three wrapper block/warp choices.
Separately labelled fallback probes use B512 and four warps. This matrix is
an explicit specialization, not an exhaustive characterization of every host
argument or destination type accepted by Python. No GPU kernel is launched.

Native compilation with FP fusion enabled contracts the outer multiply/add
to `fma.rn.f32`; the decoder's integer conversion and scale multiplication
remain separate. The real formula erases the outer arithmetic precision.
`ScalarSpec.roundedValue` explicitly describes an unfused interpretation and
must not be cited as the exact floating result of the default native binary.
The separate `ScalarSpec.fusedValue` records that affine contraction without
claiming a hardware-conformance proof.
`FPProof.whole_state` establishes complete typed source/candidate execution
equality for all six storage profiles, all initial states and all floating
algebras, including equal failure results. It uses only integer multiplication
commutativity; the full random and floating suffix is unchanged. It is not
an assertion that unfused arithmetic equals the observed native FMA result.
`Erasure.complete_projection` proves successful floating-dtype erasure of the
complete source for all profiles. `RealSurface.realizes_frame` provides the
standard tagged-cell realization and full frame; `RealIO.source_correct`
provides successful finite flat-memory execution under explicit allocation
bounds and zero undef (the latter is required by the general flatten bridge).
`Correct.lean` publishes both properties for the original and candidate, with
the original signed64 seed/offset domains and launched signed-i32 index domain.
The outer real arithmetic erases precision; this specification retains the
generator's canonical rounded fp32 uniform decoder. It proves each launched
program separately, not the Python generator-state mutation or a device run.

`TypedExecution.source_executes` proves successful typed execution of all 23
source statements for every floating algebra, every initial state and all six
storage profiles under the original input domain. Local typed proofs cover
the signed64 split, uint32 wrap, five snapshots, all four Philox words, actual
uniform fp32 payloads, fp32 subtraction/multiplication/addition, four offsets,
and all four original destination conversions/stores.
`TypedExecution.nonempty_pair` gives a common successful state for source and
candidate in the public default-fp32 engine from empty registers. It uses the
wrapper-compatible B1024/N4095 and seed=offset=-1, with all four output streams
active, the last lane masked, and counter wrap. The consistency model uses
real arithmetic and the canonical finite fp32 bit decoder; identity casts
make it an abstract consistency witness, not a device rounding model. No
random output is precomputed, and no successful execution is assumed.

`Witness.real_nonempty` instantiates the full finite flat-memory I/O contract
with one output allocation of extent 4095 at base 19 and empty registers. It
provides all active tagged-cell values and the complete frame, for either
source variant and all storage profiles. `Correctness.nonempty_instances`
and `Equivalence.nonempty_pair` expose ordinary successful-execution theorems
for the public real and FP programs. `Proofs/Audit.lean` checks their complete
axiom dependencies along with the public specifications.

The unchanged native receipt is retained as `native_receipt.json`, including
the selected and fallback helper source and all 24 compilation records. The
original `/tmp` artifact locations remain provenance fields; hashes and the
recorded source/precision evidence are stored in the receipt itself.
