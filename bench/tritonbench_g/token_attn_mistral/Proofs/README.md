# token_attn_mistral

The selected repair [token_attn_mistral.py](../token_attn_mistral.py) and its existing
integer-address candidate [token_attn_mistral.py](../token_attn_mistral.py)
are copied as complete function bodies in [Kernels.lean](../Kernels.lean). Caller
storage types and scalar bindings precede the quotations.
[source correspondence inventory](../../../../validation/source-correspondence.json) checks both 20-statement bodies,
including the sliding maximum, loop hint, masks and explicit output conversion.

The probability mask uses the attention length. The page-table and value masks
use the sequence length plus the sliding origin. Tail lanes can therefore read
values after the attention-length boundary; the proof preserves those accesses
and every padded FP32 product/reduction/update in source order.

[Ports.lean](Support/Ports.lean) retains the actual signed sequence, start, request,
attention and token words, including inactive entries. `Admissible` states the
inherited nonnegative input domain explicitly: attention words match arbitrary
logical counts; sequence/start/request words are nonnegative; token words are
nonnegative only when read under the actual visited-block and value masks.
Named natural address projections are connected to those signed reads by the
[bindings](ReferenceBinding.lean). Port extents are derived from the work
list and counts; no fixed sequence cap is introduced.

[Correct.lean](../RealCorrect.lean) proves the complete submitted work list's gathered
weighted sum, with the sliding-window condition visible in the formula. Positive
block size, read/output separation and unique output addresses remain explicit.
The proof includes actual signed-loop execution, access safety, all output lanes
and the exterior frame. [FPEquiv.lean](../FPEquiv.lean) compares both complete grids,
including zero-block behavior. It changes only integer probability-address
arithmetic and assumes no numerical rewrite rule.

The optional [integer min/max frontend](../../../../VeriTile/Triton/DSL/IntegerMinMaxLiterals.lean)
keeps weak zero literals integral on known signed metadata in straight-line
assignments. Independent complete typed-program checks cover that integer path
and preserve the floating path after name rebinding. A semantic regression also
checks the actual sliding origin for arbitrary signed length words.

[IO.lean](../IO.lean), [Contract.lean](../Contract.lean) and [Proofs/](.) contain
the complete interfaces, existing input checks and proof connections.
[Validation status](../../../../validation/body-migration-progress.json) distinguishes
local verification from independent replay; compilation alone is not completion.
