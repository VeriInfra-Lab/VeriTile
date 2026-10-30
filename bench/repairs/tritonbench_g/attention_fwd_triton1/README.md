# Checked chunked linear attention

`Whole001/attention_fwd_triton1_repaired.py` preserves the accepted FP32 source graph, six casts, `allow_tf32=False`, default fusion, and public output-only return. This is chunked linear attention: row `r` sees tokens before `min(T, (r // 32 + 1) * 32)`, including later tokens in its own chunk. It has no softmax. The optional internal history stores the KᵀV state before each chunk.

The public caller supports ordinary CUDA strided FP32 Q/K/V tensors with the same rank-four shape and device, resolved contiguous storage, and power-of-two D ≥ 16. Read-only Q/K/V aliases are allowed. Boolean STORE/IFCOND flags preserve their source behavior. Empty B/H/T returns fresh O/H without a launch. Inputs are never materialized or padded; boundary-checked zero loads and masked O stores handle tails.

`Contract.lean` connects the executed metadata, coordinate, actual data-pointer alignment, and absolute address checks to the complete machine domain. Canonical launch strides avoid oversized singleton strides. Ordinary tensor/allocation bindings supply live storage and fresh outputs; they do not assume the checked pointer alignment or padded address bounds.

`Correct.checked_real` proves the independent input-only Real chunk formula, pre-update H values, disabled-H preservation, read-only inputs, and the complete memory frame. Its scale is `1 / sqrt(D)` without asserting exact representation in a finite word. `FPEquiv.checked_native` proves the complete source-ordered FP32 output/history graph and public return from the actual host scale word. Local mulRN/fmaRN bit laws and the documented ordinary compiler/runtime graph binding remain deferred, as described in `Whole001/README.md`.

Validation: the accepted 16-module closure and four checked entry modules compile with clean standard-base axiom audits. `python bench/repairs/tritonbench_g/attention_fwd_triton1/test_checked_contract.py` executes the actual kernel AST on CPU for seven lengths and both flags, checks full history/output writes against independent dyadic formulas, exercises real host guards through FakeTensor, and checks the exact original-to-repair AST delta. It does not claim GPU execution or native numerical conformance.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `921575e2b476c9033d5042927038075a7c058b7ae174fc0beccabe28440abeef`.
The installed Triton JIT chooses the current CUDA device and stream; equal
input devices alone did not select the owning tensor device. The actual source
submission now enters `torch.cuda.device(q.device)` and restores the
ambient caller device. Formal host device/stream observations refer to this
effective context. CPU controls check a cuda:1 launch from an ambient cuda:0
caller and restoration on success and submission exception. Kernel AST and
numerical semantics are unchanged. The prior hash is retained as acceptance
correction evidence; no native GPU/runtime conformance is claimed.
