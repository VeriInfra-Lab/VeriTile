# Source binding

Selected candidate: `flash_decode2_phi_block_checked.py`.

Candidate SHA256: `e40385d5a9c3c674443f90d14f539298e36bc890ba32229b8c31101e14334653`.
Original selected source SHA256: `6eec28122238f0f4ea7352b8b8f52ab2a37aae7823c828c974ba392c465d61e1`.
JIT AST SHA256, shared unchanged with the original: `643dfb229a4f769a91dc36fd0815a8ff92f6d2e2671aa5b7627b9ec4aad1f922`.

`check_candidate.py` removes precisely the new guard after the original empty return and checks equality of the complete module AST. It also runs the original source interpreter on the candidate path without modifying the original checker. This ties the local `Kernels.kernel` alias to the original `FlashDecode2.phi` quotation and original source-connection proof.

The source promotes PID/offset and length arithmetic to INT64, reads signed INT32 lengths, iterates summaries in increasing order, loads only live `d<D` Mid lanes, and writes only live `d<D` output lanes. Log, maximum, exponential, multiplication, addition and final division retain their original FP32 operation structure. The native target preserves explicit input-load boundaries and masked zero loads. Its schedule choices depend on expression shapes, not word payloads.

The original defect packet pins the installed Triton frontend and documents the rejected `D=1048577`, `BLOCK_DMODEL=2097152`, single-grid-block counterexample. The new limit follows the pinned `tl.arange` maximum of 1048576 elements. The successful D=3 compiler control is reused through exact JIT AST identity. No broad compiler/resource or runtime guarantee is inferred from that control.

The candidate imports `square_matrix.Contract` locally before the frozen original Phi contract to resolve an existing import drift. The Llama owner refreshed unchanged shared Phi helper build objects to the current Algebra layout. No original source bytes, shared proof core, canonical selection or ledger entries are edited by this candidate.
