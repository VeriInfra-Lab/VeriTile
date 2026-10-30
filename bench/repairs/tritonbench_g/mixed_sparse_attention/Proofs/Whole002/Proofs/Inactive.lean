import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Memory
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- The actual source early return reads sequence length and performs no store. -/
theorem inactive_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (c : Config) (s : State α)
    (h : sequenceLength c s≤rowStart c s) :
    ∃ t, FP.Structural.exec (backend M D) (kernel c) s=some t ∧ t.mem=s.mem := by
  simp only [sequenceLength,rowStart,Region.cast] at h
  rcases c with ⟨storage,q,k,v,out,seq,blocks,offsets,counts,columns,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svn,svk,soz,soh,som,sok,Z,H,T,NR,NS,NV,BM,BN,BD,scale⟩
  cases storage <;> dsimp only at h <;> simp [kernel,parsedKernel,TypedSource.FP16.source,TypedSource.BF16.source,
    ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,integralFloorDiv,integralMod,
    backend,Region.cast,State.setReg,show ¬s.pids 0*BM<(s.mem {name:=seq.name} (s.pids 1/H)).read .nat by exact Nat.not_lt.mpr h]

theorem inactive_outputMemory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (c : Config)
    (s : State α) (a : Accumulator α c.BM c.BD) (h : sequenceLength c s≤rowStart c s) :
    outputMemory M D c s a=s := by
  have hi : ∀ i : TileIndex [c.BM,c.BD],¬rowStart c s+i.1.val<sequenceLength c s := by
    intro i
    omega
  simp [outputMemory,hi]

#axiomsClean inactive_run
#axiomsClean inactive_outputMemory
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
