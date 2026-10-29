import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPBody
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPInvariant
open VeriTile.Triton FP.Structural FPProgram VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues FPSetup FPBody
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem setup_ready (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) :
    Invariant h data batch split slot (accumulator M P h data batch slot split 0)
      (preloop M P h batch split slot data metadata).val := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    constructor
    all_goals
      simp (config := {maxSteps := 300000}) [preloop,preguard,State.setReg,entry,
        machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
        numeric,integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,FP.Structural.bop,
        Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,Data.cells,metadata,accumulator,
        Broadcast.leftIndex,Broadcast.rightIndex,Int.toNat_mul,Int.toNat_natCast]
      try rfl
      all_goals
        funext i
        simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Int.toNat_mul,Int.toNat_natCast]
        all_goals rw [←Int.natCast_mul,←Int.natCast_mul,Int.toNat_natCast]

theorem iteration_ready (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot c : Nat) (data : Data α) (a : Fin h.BN → α) (s : State α)
    (ready : Invariant h data batch split slot a s) :
    Invariant h data batch split slot
      (fun n => M.binary (some .fp32) .real .add (a n)
        (M.cast (some .fp32) (dtype h.storage) .real (sums M P h data batch slot split c n)))
      (iteration M P h batch split slot c data a s ready).val := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    constructor
    all_goals
      simp (config := {maxSteps := 400000}) [iteration,State.setReg,
        machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
        numeric,integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,FP.Structural.bop,
        Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,Data.cells,dtype,
        ready.memory,ready.scaleReg,ready.batchReg,ready.offsetN,ready.offsetK,ready.inputPtr,ready.weightPtr,ready.acc,
        sums,VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.products,loaded,weight,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.inputOffset,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.weightOffset,
        Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm,Nat.add_assoc]
      try rfl
      all_goals
        funext i
        simp [FP.Structural.bop,VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.products,loaded,weight,dtype,ofFloat,toFloat,
          Broadcast.leftIndex,Broadcast.rightIndex,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.inputOffset,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.weightOffset,Nat.mul_comm,Nat.add_assoc]
        congr 3
        funext j
        simp [FP.Structural.bop,VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.products,loaded,weight,dtype,ofFloat,toFloat,
          Broadcast.leftIndex,Broadcast.rightIndex,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.inputOffset,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.weightOffset,Nat.mul_comm,Nat.add_assoc]

  all_goals
    norm_cast
    simp only [Int.toNat_natCast,Int.ofNat_lt]

#axiomsClean setup_ready
#axiomsClean iteration_ready
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPInvariant
