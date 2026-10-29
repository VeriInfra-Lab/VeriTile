# Partial byte access and complete ReLU program orders

This package adds four Lean modules and one public specification for the same
nine integer/bool ReLU profiles. It adds no dtype profiles and no numerical
rules. Whole-original-program acceptance remains 0/183.

`AccessMemory.Memory` contains the existing total byte payload together with
allocation, read permission and write permission predicates. `partialContents`
is an allocated-content representation view: unallocated addresses are `none`,
and allocated addresses expose their stored byte without asserting readability.
The access operations are separate. `readByte` succeeds only when allocated
and readable; `writeByte` succeeds only when allocated and writable. Their
explicit failures are `loadUnavailable address` and `storeUnavailable address`.
Write failures retain the state reached by preceding successful byte writes.

`AccessMemory.loadLE` reads each constituent byte through `Except` bind, then
assembles it in the same little-endian order as the frozen byte codec.
`loadWords` reads every active lane in increasing lane order. A program matches
the result of all active loads before reaching its pid-zero store gate. Thus
nonzero pids cannot skip access checks. Inactive masked lanes perform no byte
access in this model. The `nonzero_program_unavailable` theorem shows the
distinction explicitly: the old total byte projection has identity effect,
while the new execution fails when the first active input byte is unavailable.
The zero entries returned by `loadWords`/`loadedPrefix` outside the loaded
prefix are placeholders in this mathematical container. They are never used
by active stores and do not assert that the original no-`other` masked load
returns zero, or any other particular inactive value.

On pid zero, all source words are collected before any output store. The
returned word values supply four-byte output stores in lane/byte order. Each
byte checks its own allocation and write permission. `write_prefix_unavailable`
proves that a later unavailable store returns the exact preceding write state.
The original integer comparison/conversion callbacks are total and pure;
there is no separate failing scalar-computation operation in this model.
Nonzero-pid scalar computations are not separately evaluated by this access
interpreter. The callbacks are total and pure, so this model covers load/store
success without asserting an instruction trace for those computations.

Full success requires only that all N source words are allocated/readable and
the first `min(N, 1024)` fp32 outputs are allocated/writable. No output read
permission, tail allocation or tail write permission is assumed. Allocation
and both permission maps are unchanged by execution. The byte payload and
allocated-content view have a complete frame outside the written output
prefix. The arbitrary latent payload of an unavailable tail is preserved;
the old complete `LogicalView` is a payload relation, not a tail-access claim.

The ascending interpreter and every complete permutation of original program
ids succeed and produce exactly the frozen `byteGrid`. Actual accesses occur
on the preceding program's current state. This is proved by first connecting
each successful partial program to its byte effect and then proving complete
whole-program permutations preserve that effect. Only pid zero writes.
Programs are atomic at this schedule layer; no native instruction-interleaving
or race theorem is claimed.

The public specification composes these success results with the frozen
`relu_byte_grid_correctness`: actual old typed-grid execution, independent
word and byte specifications, final `LogicalView`, frame, unchanged access
metadata, finite-address input access and inherited address certificate.
It retains distinct logical names, full input/output interval separation,
B=1024, N<2^31 and 64-bit base/end domains as explicit theorem premises.
The stronger full-N output address bound is inherited from that certificate;
it does not strengthen output allocation/permission requirements.
`constructed_access_grid` removes the initial view premise using the existing
state construction for arbitrary byte payloads.

This is a custom mathematical interpreter with partial accesses. Native
allocation, access rights, endianness, provenance, alignment, layout and errors
have not been related to it. Its failures are deterministic first-unavailable
byte results in the chosen lane/byte order, not GPU trap/poison/sanitizer
semantics. There is no TTIR/compiler simulation, host execution, native launch
or native instruction-interleaving result. Remaining dtypes and error branches
are unchanged. The three previous integer-to-fp32/zero callbacks retain their
two existing numerical followups; no rule is duplicated.

The final evidence records the actual four compiler exits and full axiom
audits. Every real earlier attempt, including the initial missing-import
attempt and subsequent proof errors, is retained with its exact source
snapshot and log. Failed attempts are not successful verification evidence.
Source/config dependencies and imported project oleans are fingerprinted
before and after the final batch. All four predecessor ReLU packages remain
unchanged, and the 13 inherited TTIR fingerprints are rechecked. This package
performs no new frontend/backend compilation, GPU execution, comparator replay
or external-library rebuild.
