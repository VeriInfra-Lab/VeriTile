# Vector addition: task-local fp64 typed memory and full program grid

Four modules extend the independently reviewed binary64 core to actual logical
memory execution for all 29 successful fp64 operand profiles. A memory address
contains an optional complete tagged cell. Every fp64 value has a real 64-bit
payload. Missing cells and mismatched dtype tags produce unavailable typed loads;
they never supply a default word.

`runLanes` computes and writes one lane at a time. All reads use a fixed snapshot
of the memory at program entry; stores update a separate working memory. This
snapshot-based model does **not** reproduce the source's vector load-all,
compute-all and store instruction trace. It supports the proved successful final
memory result; a trace/compiler bridge remains open. Masked lanes preserve memory
without requiring input cells or inventing a masked-load `other` value.

`runGrid` executes every selected program against the current memory produced by
the preceding program. The theorem covers every complete permutation of
`range(ceil(N/1024))`. Under the explicit input view, output separation and
conversion-domain premises, the final memory equals an independent reference
computed from the initial typed input words. Every region/address outside the
active output remains exactly as it was, including absent cells and arbitrary
output-tail contents. Zero-length grids and completely masked programs are
handled explicitly.

The initial `InputWords` relation requires full tags and payloads. Output and
input region names must be distinct as an explicit selected-domain assumption;
this is not a proof of wrapper allocation freshness or physical nonaliasing.
The two input names may agree only when their complete tagged input cells agree.
`OutputDefined` restricts only active integer-output conversions. All 20 floating
output profiles are automatically defined for arbitrary words under the chosen
total callbacks; the other 9 profiles retain the explicit integer-conversion
domain. No finiteness restriction is added to floating outputs or masked lanes.

Unavailable active typed loads and undefined integer conversions are different
error constructors. An error returns no final memory. This mathematical exception
policy does not establish native poison, traps, partial writes or rollback, and
it must not be presented as native error execution.

The 15 numerical callback families are inherited unchanged. The concrete queue
candidates in `evidence/numerical-bindings.json` contain 73 dtype/port instances
and 120 applicable declaration/port bindings. Every instance records the exact
source and raw TTIR hashes, callback/type/operation, frontend configuration and
public declaration statement hash. No numerical experiment or accepted queue
result is claimed. The full captured sm90 configuration, including Triton/Torch
versions, block size and options, is retained in `frontend-configuration.json`.

The two public specifications state success of this **task-local typed AST and
memory interpreter**. They do not state success of the shared `FP.Structural`
interpreter or of native Python/Triton execution. The preceding 160 logical
value/grid profiles use the shared interpreter and concrete software operations;
these additional 29 use the separate typed fp64 interpreter and unverified
numerical callbacks. The combined inventory is 189/217 across two models, not
189 profiles proved in one interpreter. The 28 fnuz frontend rejection pairs
remain outside successful logical-grid execution. Whole-original-program
acceptance remains 0/183.

Still required structurally: source/extraction/compiler correspondence, physical
bytes and pointer bounds/layout/admission, finite-width native offset/mask
behavior, output allocation/freshness and device dispatch, inactive-lane and
error behavior, host/module/harness composition, other source dtype/error
branches, and native instruction scheduling. These are not numerical deferrals.

All compilation attempts retain actual exits, logs and exact source snapshots.
The final four-module run binds source and imported olean fingerprints before
and after, records the actual Lean search path, audits axioms and noncircularity,
and is followed by an actual disposable Git staged whitespace check. The core
package and its independent review remain unchanged.
