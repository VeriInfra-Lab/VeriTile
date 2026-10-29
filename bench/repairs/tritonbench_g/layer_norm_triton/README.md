# layer_norm_triton

Forward:

Named output {y[j] = ((x[j]-mean(x))*inverse(x,eps))*weight[j]+bias[j], mean = sum_{j<N} x[j]/N, inverse = 1/sqrt(sum_{j<N}(x[j]-mean(x))^2/N+eps)}.

Grouped backward:

For supplied saved mean/inverse, q[j]=dy[j]*weight[j], z[j]=(x[j]-mean)*inverse, DX[j]=(q[j]-sum_{k<N}q[k]/N-z[j]*sum_{k<N}q[k]*z[k]/N)*inverse. DW[j]=sum_{row in groupRows}dy[row,j]*z[row,j]; DB[j]=sum_{row in groupRows}dy[row,j].

Final backward:

Named outputs DW[col]=sum_{row:Fin M}xs.dw[row,col] and DB[col]=sum_{row:Fin M}xs.db[row,col].

- [layer_norm_triton_repaired.py](layer_norm_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

All three genuine original source stages remain separate. Forward is one source row; grouped exposes every visited DX plus both partial sums; final uses the existing BM=32 Real source scope. No new full host composition claim.

## Conditions

- Forward retains SourceForwardReal.Laws m, 0<N and N≤B.
- Grouped retains the same Laws, 0<G, N≤B and explicit row*stride=row*N for every row<M.
- Final retains SourceFinalReal.Laws m; grouped keeps the accurate ReadProgramIO judgment and its declared read view, masks and output scope.

Exact FP program pairs and numerical dependencies are recorded in the [layer_norm_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
