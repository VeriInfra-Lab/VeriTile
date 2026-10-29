import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawBody

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    let t := (bodyExecution M D p st initial current k a ctx ha).val
    Context p initial t ∧ HasAcc p (update M D p st initial k a) t := by
  cases st <;> constructor
  all_goals try (constructor <;> simp [bodyExecution,ctx.memory,ctx.dimM,ctx.dimN,ctx.dimK,
    ctx.strideAM,ctx.strideAK,ctx.strideBK,ctx.strideBN,ctx.strideCM,ctx.strideCN,
    ctx.blocks,ctx.offk,ctx.am,ctx.bn,ctx.mm,ctx.nn,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex])
  all_goals simp [HasAcc,bodyExecution,RawSource.update,RawSource.loadA,RawSource.loadB,RawSource.addressA,RawSource.addressB,RawSource.innerOffset,RawSource.innerMask,
    dtype,FPValue.read,backend,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals apply congrArg₂ (fun (av : AIndex p → α) (bv : BIndex p → α) =>
    D.apply (some .fp32) _ _ .real {} (batch := []) av bv a)
  all_goals funext i
  all_goals simp [RawSource.loadA,RawSource.loadB,RawSource.addressA,RawSource.addressB,
    RawSource.innerOffset,RawSource.innerMask,FPValue.read,dtype]

  all_goals rfl

#axiomsClean bodyContext
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
