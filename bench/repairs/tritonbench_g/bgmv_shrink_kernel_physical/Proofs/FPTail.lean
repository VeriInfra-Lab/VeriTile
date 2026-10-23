import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPLoop
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.AtomicScatter
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPTail
open VeriTile.Triton FP.Structural FPProgram VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues FPSetup FPBody
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Constructive execution of source scaling, output conversion, address/mask
setup and the computed SPLIT_K==1 branch (false for this actual launch). -/
def finish (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (s : State α)
    (ready : Invariant h data batch split slot (accumulator M P h data batch slot split h.cycles) s) :
    {t : State α // run (machine M P) (tail h) s=some t} := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage <;> cases outputType
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [tail,active,code,Surface.source,
      ComputeKernel.surfaceBody,List.drop,Source.conditional?,Source.intRange?,
      Option.map_some,Option.getD_some,beq_self_eq_true]
    simp (config := {maxSteps := 300000}) [Source.intRange?,Option.map,Option.getD,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,FP.Structural.bop,
      State.setReg,ofFloat,toFloat,TileShape.axisDim,Region.cast,Data.cells,dtype,
      ready.memory,ready.scaleReg,ready.batchReg,ready.acc,FP.Structural.atomicAdd,atomicAddition]
    try rfl

theorem finish_memory (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (s : State α)
    (ready : Invariant h data batch split slot (accumulator M P h data batch slot split h.cycles) s) :
    (finish M P h batch split slot data s ready).val.mem=
      (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile h P batch (payload M P h data batch slot split) s).mem := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage <;> cases outputType
  all_goals
    simp (config := {maxSteps := 500000}) only [finish,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile,payload,
      State.setReg,machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,FP.Structural.bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,Data.cells,dtype,
      ready.memory,ready.scaleReg,ready.batchReg,ready.acc,
      Broadcast.leftIndex,Broadcast.rightIndex,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.outputOffset,Nat.mul_one]
    simp only [Nat.zero_add,Nat.mul_one,FloatDType.toTileDType,id_eq]
    apply VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.fold_memory
    · exact ready.memory.symm
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

#axiomsClean finish_memory
#axiomsClean finish
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPTail
