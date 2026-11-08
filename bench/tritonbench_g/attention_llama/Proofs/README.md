# attention_llama

Current repaired Python implementation: [attention_llama.py](../attention_llama.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Finite-N exp-attention weighted value average over keys satisfying causal=false or key+SP≤query, with runtime scale and full D feature sum.

[Sources/Body.lean](../Kernels.lean) contains the complete original and carry-first candidate function bodies for both FP16 and signed-byte E5M2 storage. The four pairs are recorded in [source correspondence inventory](../../../../validation/source-correspondence.json). Pointer/scalar types and the closed storage flag are supplied outside the pasted bodies; `IS_CAUSAL` remains an ordinary source condition. All original bitcasts, conversions, positional load fallbacks, masks, loop bounds and the final implicit FP16 store are retained.

The repaired original Python is unchanged. [candidate-provenance.json](../../../../validation/tasks/attention_llama/candidate-provenance.json) records a complete Python source for the existing carry-first normalizer candidate: only `l_curr` addition order and the matching kernel name differ. The runtime scalar ABI, both actual storage paths and their existing numerical assumptions are preserved.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
