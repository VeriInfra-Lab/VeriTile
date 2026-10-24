# Vector addition: arbitrary sequential program order

Two new modules extend the existing 25 floating-pair execution contract to
any permutation of the complete ceil-divided program grid. No input/output
region separation is required: distinct natural flat tile offsets ensure
earlier programs cannot modify the next program's active input addresses,
even when the logical region names coincide. Both loads precede the store.

The runner executes the supplied list in forward order, gives every program
fresh registers and the original launch metadata, and passes actual preceding
memory to the existing typed source execution. A duplicate-free partial list
has an independent initial-memory formula: an offset is updated exactly when
its quotient by block size belongs to the completed list and it is below n.
The public complete-permutation result equals the existing TypedValueCorrect
specification, including all untouched cells. Positive block size is required;
zero-length inputs, partial final tiles and arbitrary starting registers are
included. Repeated program IDs are deliberately excluded because in-place
logical aliasing could otherwise apply addition to already-updated inputs.

The five storage profiles and promotion routes are unchanged. Floating add
(including fp32) and distinct-format conversions remain arbitrary callbacks;
same-format casts remain identity. No new numeric law or numerical followup is
needed. Logical tags/default wrong-tag reads, physical byte layouts and views,
finite-width pointer arithmetic, instruction interleaving, native compiler/
runtime behavior and remaining original dtype/host/harness coverage remain
outside this result. Original-program acceptance stays 0/183.

Final actual verification: both modules compiled with exit0, eight named axiom
checks and one noncircularity check passed. All 156 project source/configuration
fingerprints matched before and after compilation. Toolchain and external
Mathlib imports follow the unchanged pinned repository configuration. These
are Lean proof results; no official comparator or GPU result is asserted.

Only the two .lean files, this README, verifier/freezer scripts and evidence
are review material. Dependency symlinks and .olean outputs are temporary and
must not be copied into the repository. Final compiler logs and receipts are
frozen. Full repository integration checks and independent review are pending.
