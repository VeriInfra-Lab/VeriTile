# FP32 primitive and compiler model boundaries

The optional single-source native results use `Native.Instructions` over `BitVec 32`.
Its six ports name the selected native operations: `add.f32`, `mul.f32`,
`fma.rn.f32`, `sqrt.approx.ftz.f32`, `div.full.f32`, and `neg.f32`. They are
primitive semantics, not algebraic transformation laws or deferred experiments.
No FP32=Real equation, reassociation, split/FMA identity, commutation law, or
whole-output equality is assumed.

`Native.algebra` is the explicit compiler-profile interpretation of the
canonical typed source. It retains product provenance to represent the actual
local fusion choices. `Native.row` is a separately written word-level graph;
the complete program theorem derives source success and its output from the
local interpretation, including all stores and memory frames.

For block 1, square-plus-epsilon is an FMA. For blocks 2 through 128, the first
shuffle fuses the self product with the peer's rounded product. Above 128,
each thread starts with rounded slot 1, fuses slot 0, then slots 2 onward.
Shuffle distances are 16, 8, 4, 2, 1 within warps and 2, 1 across warp groups.
The dot product multiplies X by DY. Dot-times-reciprocal order depends on
whether the block exceeds 32. The final expression explicitly negates the
correction and uses `fma.rn.f32(rstd, dy, negativeCorrection)`.

The selected profile is Triton 3.7.1/CUDA SM90, four warps, one CTA, three stages,
default fusion, and power-of-two blocks 1 through 16384. Runtime value and
alignment specialization are disabled. The source pointers and epsilon are
FP32; strides and N are signed64. The source literal one and masked positive
zero have exact FP32 words. Epsilon is the actual runtime FP32 ABI value,
including conversion of Python's default `1e-5` by the binder.

The proof covers all input/gradient/epsilon words. Basic and approximate
instruction policies, including special values and FTZ, are selected by the
primitive implementation. The model does not identify distinct NaN payloads or
silently replace approximate square root/full division with ideal operations.
The input negative flags are checked in Python; this is independent of the
native `neg.f32` instruction's semantics.

`Words.Codec` is ordinary little-endian storage binding. It turns the proved
output words into all four actual output bytes; the physical-grid theorem
preserves every exterior byte. Source execution and memory preservation are
proved, not included as model-boundary premises.

`Native.Obligations`, retained for optional historical interpretation transport,
contains only equalities of the six primitive functions. Current public kernel
results do not require it. It is a model-binding interface, not a list of
numerical experiments. There are no deferred experiments for this revision.
Offline compilation checks pin the profile and instructions; they do not
establish general compiler correctness or native numerical conformance.

The current main `FPEquiv.grid_equiv` compares the repaired source with its
row-index-commuted candidate through complete safe successful row-grid IO.
Both direct source quotations retain all FP32 arithmetic, reduction and fusion
inputs. The external FP32 epsilon is declared with `tl.param` and supplied by
the invocation, without a source literal or conversion. Only the three signed
integer row/stride products commute; the proof
uses integer multiplication commutativity, with no floating rewrite law,
finite-value premise, experimental rule or deferred experiment. The numerical
rule set and additional logical rewrite domain are empty. Ordinary output
geometry and finite backed IO windows are retained from the Real/source contract.

`Misc/NativeCorrect.lean` retains the independent native graph theorem. It may
instantiate the same primitive interpretation used by the main pair, but is not
needed to prove the index rewrite. The pair does not assert FP32-to-Real equality,
compiler correctness, hardware conformance, or a different reduction schedule.
