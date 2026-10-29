import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Prefix
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Memory
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false


def suffix (c : Config) : List ComputeStmt := (activeBody c).drop 24


theorem suffix_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s t : State α) (a : Accumulator α c.BM c.BD) (h : Ready M c s t a) :
    ∃ final, run (backend M D) (suffix c) t = some final ∧
      final.mem = (outputMemory M D c s a).mem := by
  rcases c with ⟨storage,q,k,v,out,seq,blocks,offsets,counts,columns,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svn,svk,soz,soh,som,sok,Z,H,T,NR,NS,NV,BM,BN,BD,scale⟩
  cases storage <;> simp [suffix,activeBody,parsedActiveBody,parsedKernel,TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.Storage.tile,TypedSource.Storage.float,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,numericEq,backend,Region.cast,ofFloat,toFloat,store,
    h.output,h.normalizer,h.outptr,h.mask,outputPointer,outputMemory,show (0.0 : ℝ)=0 by norm_num]
  all_goals apply maskedMemory_congr; exact h.mem

#axiomsClean suffix_run
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
