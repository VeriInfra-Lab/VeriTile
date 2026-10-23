import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportLoop

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup FPBody
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The original computed accumulator is retained. Only the backing memory
changes; the source's atomic reads observe its actual current output cells. -/
def finish_rebased (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (s : State α)
    (ready : Invariant h data batch split slot (accumulator M P h data batch slot split h.cycles) s)
    (memory : RegionName → Nat → Cell α) :
    {t : State α // run (machine M P) (tail h) (rebase s memory) = some t} := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage <;> cases outputType
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [tail,active,code,Surface.source,
      ComputeKernel.surfaceBody,List.drop,Source.conditional?,Source.intRange?,
      Option.map_some,Option.getD_some,beq_self_eq_true]
    simp (config := {maxSteps := 300000}) [Source.intRange?,Option.map,Option.getD,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,FP.Structural.bop,
      State.setReg,ofFloat,toFloat,TileShape.axisDim,Region.cast,dtype,rebase,
      ready.scaleReg,ready.batchReg,ready.acc,FP.Structural.atomicAdd,atomicAddition]
    try rfl

theorem finish_rebased_memory (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (s : State α)
    (ready : Invariant h data batch split slot (accumulator M P h data batch slot split h.cycles) s)
    (memory : RegionName → Nat → Cell α) :
    (finish_rebased M P h batch split slot data s ready memory).val.mem =
      (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile h P batch
        (payload M P h data batch slot split) (rebase s memory)).mem := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage <;> cases outputType
  all_goals
    simp (config := {maxSteps := 500000}) only [finish_rebased,
      VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile,payload,
      State.setReg,machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,FP.Structural.bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,dtype,rebase,
      ready.scaleReg,ready.batchReg,ready.acc,
      Broadcast.leftIndex,Broadcast.rightIndex,outputOffset,Nat.mul_one]
    simp only [Nat.zero_add,Nat.mul_one,FloatDType.toTileDType,id_eq]
    apply VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.fold_memory
    · rfl
    · intro i u v same
      have castAddress : ((batch:Int)*columns+i.1.val)=((batch*columns+i.1.val:Nat):Int) := by
        simp only [Int.natCast_add,Int.natCast_mul]
      have natAddress : Int.toNat ((batch:Int)*columns+i.1.val)=batch*columns+i.1.val := by
        rw [castAddress,Int.toNat_natCast]
      by_cases active : i.1.val < N
      · simp only [active,if_true]
        funext r off
        simp [State.write,same,natAddress]
      · simp only [active,if_false,same]

#axiomsClean finish_rebased
#axiomsClean finish_rebased_memory
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
