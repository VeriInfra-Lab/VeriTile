import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPExecution
import VeriTile.Triton.Float.StructuralSafety
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.MemoryTransport

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RebasedSafety
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup FPBody MemoryTransport
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem prefix_safe (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split : Nat) (data : Data α) (memory : RegionName → Nat → Cell α) (bounds : RegionName → Nat)
    (indexBound : batch < bounds "Indices") :
    Safety.traceSafeList (machine M P) bounds (FPProgram.prelude h) (rebase (entry h batch split data) memory) := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    simpa [FPProgram.prelude,FPProgram.code,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,Source.intRange?, Option.map, Option.getD, Source.conditional?,
      ComputeKernel.surfaceBody, List.drop, beq_self_eq_true,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      machine, Algebra.withDefaultPrecision, resolvePrecision, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      rebase, State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, Data.cells, dtype,entry] using indexBound

theorem setup_safe (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int))
    (memory : RegionName → Nat → Cell α) (bounds : RegionName → Nat) :
    Safety.traceSafeList (machine M P) bounds (FPProgram.setup h) (rebase (preguard M P h batch split data).val memory) := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    simp (config := {maxSteps := 300000}) [FPProgram.setup,FPProgram.active,FPProgram.code,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,Source.intRange?, Option.map, Option.getD, Source.conditional?,
      ComputeKernel.surfaceBody, List.drop, beq_self_eq_true,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      machine, Algebra.withDefaultPrecision, resolvePrecision, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      rebase, State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, Data.cells, dtype,preguard,entry,metadata]

theorem body_safe (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot c : Nat) (data : Data α) (a : Fin h.BN → α) (s : State α)
    (ready : Invariant h data batch split slot a s) (memory : RegionName → Nat → Cell α) (bounds : RegionName → Nat)
    (inputBounds : ∀ k, k < h.K → inputOffset h batch k < bounds "X")
    (weightBounds : ∀ n k, n < h.N → k < h.K → weightOffset h slot n k < bounds "W") :
    Safety.traceSafeList (machine M P) bounds (FPProgram.body h)
      ((rebase s memory).setReg "k" .int [] (fun _ => ((c*16384:Nat):Int))) := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    simp (config := {maxSteps := 400000}) [FPProgram.body,FPProgram.active,FPProgram.code,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,Source.intRange?, Option.map, Option.getD, Source.conditional?,
      ComputeKernel.surfaceBody, List.drop, beq_self_eq_true,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      machine, Algebra.withDefaultPrecision, resolvePrecision, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      rebase, State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, Data.cells, dtype,ready.memory,ready.scaleReg,ready.batchReg,ready.offsetN,ready.offsetK,ready.inputPtr,ready.weightPtr,ready.acc, Broadcast.leftIndex, Broadcast.rightIndex]
    constructor
    · intro e FPProgram.active
      norm_cast at FPProgram.active
      simpa only [inputOffset,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,FPInvariant.negative_nat,FPInvariant.negative_sum,Nat.sub_zero] using inputBounds (c*16384+(e.val+split*256)) (by change _ < K; exact FPProgram.active)
    · intro n e hn hk
      norm_cast at hk
      simpa only [weightOffset,Nat.mul_comm,Nat.add_assoc,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,FPInvariant.negative_nat,FPInvariant.negative_sum,Nat.sub_zero] using
        weightBounds n.val (c*16384+(e.val+split*256)) hn (by change _ < K; exact hk)

theorem tail_safe (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (a : Fin h.BN → α) (s : State α)
    (ready : Invariant h data batch split slot a s) (memory : RegionName → Nat → Cell α) (bounds : RegionName → Nat)
    (outputBounds : ∀ n, n < h.N → outputOffset h batch n < bounds "Y") :
    Safety.traceSafeList (machine M P) bounds (FPProgram.tail h) (rebase s memory) := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage <;> cases outputType
  all_goals
    simp (config := {maxSteps := 400000}) [FPProgram.tail,FPProgram.active,FPProgram.code,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,Source.intRange?, Option.map, Option.getD, Source.conditional?,
      ComputeKernel.surfaceBody, List.drop, beq_self_eq_true,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      machine, Algebra.withDefaultPrecision, resolvePrecision, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      rebase, State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, Data.cells, dtype,ready.memory,ready.scaleReg,ready.batchReg,ready.offsetN,ready.offsetK,ready.inputPtr,ready.weightPtr,ready.acc,FP.Structural.atomicAdd,atomicAddition,FloatDType.numericDType,
      Broadcast.leftIndex, Broadcast.rightIndex]
    intro n hn
    simpa only [outputOffset,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast] using outputBounds n.val hn

end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RebasedSafety
