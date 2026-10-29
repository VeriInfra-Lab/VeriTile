# attention_fwd_triton1

Chunk-causal O(h,row,col)=1/sqrt(D)*sum u<min((row/32+1)*32,T) sum e<D Q*K*V; pre-update H(h,chunk,row,col)=sum u<min(chunk*32,T) K*V.

- [attention_fwd_triton1_tile_checked.py](attention_fwd_triton1_tile_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

All heads/chunks of complete shared grid, output O and optional pre-update H history. Fixed temporal block32, full logical T and D; empty grid, STORE and IFCOND both branches preserved.

## Conditions

- Shape and public finite KernelIO ports retain source configuration/launch geometry and all accesses.
- STORE masks H channel; IFCOND branch remains in actual source and theorem quantification. Contract retains positive supported power-of-two D and live tile formation limit.

Exact FP program pairs and numerical dependencies are recorded in the [attention_fwd_triton1 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
