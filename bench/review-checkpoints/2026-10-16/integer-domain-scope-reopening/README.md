# Integer-domain scope reopening

Current source acceptance is 2/183, with 181 pending. Vector addition and ReLU's earlier whole-source acceptance is withdrawn; matrix transpose and quantize_global retain their current accepted statuses. The vector review first reduced the count to3; the subsequently confirmed ReLU omission reduced it to2.

The unchanged vector_addition, vector_addition_custom and add_value kernels accept a Python runtime extent of 2^31 through their annotated i32 binder. The installed driver stores -2^31 without rejection. Their original legal grids run only false signed masks, hence no input loads or output stores. Under ordinary successful allocation/setup, the wrappers return the unchanged allocated output. Existing finite proofs require extent<2^31 and omit that route.

The checkpoint retains original-source compilation, exact C extractor evidence, all-lane Lean proofs, the unsuccessful proof attempt, candidate scope snapshots and independent root review. No GPU execution or large tensor allocation was performed. Custom addition also has an actual zero-storage Torch shape [2^31,0] example.

Original proofs, frozen reviews, catalogs and exact numerical bindings remain unchanged. Historical acceptance is preserved in the ledger's accepted_scope_history; current production coverage is partial pending repair. Custom-vector and add-value candidates are not integrated or accepted. Their prior candidate recommendations are superseded for whole-source acceptance.

The repair must model actual narrowing, signed masks and uint64 addresses and cover returned/undefined/access outcomes. No checker was weakened and no numerical assumption can discharge this gap.

ReLU uses constexpr N, not a runtime annotated i32 bound. Five exact-original offline compilation profiles include N=2^31+1 with legal grid2^21+1. The current SourceDomain excludes this path, while the existing proof already exhibits negative wrapped offsets and an active mask. Its full-source claim is withdrawn separately in relu-root-review.json. Bounded proofs and all numerical registrations remain unchanged.
