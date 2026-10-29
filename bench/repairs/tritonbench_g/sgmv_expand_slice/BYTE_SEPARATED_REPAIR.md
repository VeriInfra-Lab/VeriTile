# Independent SGMV byte-separation candidate

Candidate: `sgmv_expand_slice_byte_separated.py`; public proof namespace and
files: `ByteSeparated/{Kernels,Contract,Correct,FPEquiv}.lean`.
The canonical Python, existing proofs, source pins, and frozen alias defect
remain unchanged. Selection and ledger acceptance belong to the root reviewer.

The only behavior change replaces Y/read StorageImpl identity comparison with
the actual complete untyped backing ranges observed through `data_ptr()` and
`nbytes()`. For Y=(q,m) and each X/W/starts/lengths/indices=(p,n), accept exactly
`m=0 or n=0 or q+m<=p or p+n<=q`. Python endpoint arithmetic is unbounded.
This is precisely absence of a shared byte, including distinct DLPack imports.
Adjacent imported backings and empty shared backings are admitted. Nonempty
shared backing remains rejected even for disjoint views; an empty view of a
nonempty backing still checks that whole backing. Full backing includes any
untyped trailing bytes that are not complete FP32 elements. Read/read aliasing
is unrestricted as before. Same CUDA device checks precede this comparison.

All other checks and the exact kernel AST are retained: FP32 X/Y, FP16 W,
independent INT32/INT64 starts/lengths/indices, both add modes, K=0 and tails,
3D/4D singleton weights, contiguous resolved aligned storage-offset views,
zero batch/column behavior, all signed limits, and ordered overlapping segment
semantics. No tensor copies, output allocation, finite-value guard, dtype
expansion, or new host scheduling/allocator mechanism is introduced.

The new binding uses actual core FP32 X/Y Tensor objects with ordinary
TensorLaws and exact metadata/storage observations. It contains no separation,
source success, or numerical-result field. Source and native proofs read the
actual pointer bytes; the byte frame is derived from this guard. Complete
ordered-segment source execution and the independent Real/native references
reuse the retained generic proof chain. Native statements retain only local
conversion/add/m16n8k16 realization and two zero-conversion laws. CPU controls
check actual source AST behavior; no new GPU conformance claim is made.
