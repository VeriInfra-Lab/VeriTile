import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.MemoryTransport

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup FPBody
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The existing iteration is insensitive to every cell outside its masked
read windows. Its arithmetic state and ordered recurrence are reused exactly. -/
theorem body_transport (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot c : Nat) (data : Data α) (a : Fin h.BN → α) (s : State α)
    (ready : Invariant h data batch split slot a s) (memory : RegionName → Nat → Cell α)
    (inputRead : ∀ k, k < h.K → memory "X" (inputOffset h batch k) =
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.input (inputOffset h batch k))))
    (weightRead : ∀ n k, n < h.N → k < h.K → memory "W" (weightOffset h slot n k) =
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.weights (weightOffset h slot n k)))) :
    run (machine M P) (body h)
      ((rebase s memory).setReg "k" .int [] (fun _ => ((c*16384:Nat):Int))) =
      some (rebase (iteration M P h batch split slot c data a s ready).val memory) := by
  suffices transport : run (machine M P) (body h)
      ((rebase s memory).setReg "k" .int [] (fun _ => ((c*16384:Nat):Int))) =
      (run (machine M P) (body h)
        (s.setReg "k" .int [] (fun _ => ((c*16384:Nat):Int)))).map
          (fun t => rebase t memory) by
    rw [(iteration M P h batch split slot c data a s ready).property] at transport
    exact transport
  have inputLoad (k : Nat) (fallback : Value α (dtype h.storage).toTileDType) :
      (if k < h.K then (memory "X" (inputOffset h batch k)).read (dtype h.storage).toTileDType else fallback) =
      (if k < h.K then ofFloat (dtype h.storage) (data.input (inputOffset h batch k)) else fallback) := by
    by_cases active : k < h.K
    · simp only [active, if_true, inputRead k active, Cell.read_same]
    · simp only [active, if_false]
  have weightLoad (n k : Nat) (fallback : Value α (dtype h.storage).toTileDType) :
      (if n < h.N ∧ k < h.K then
        (memory "W" (weightOffset h slot n k)).read (dtype h.storage).toTileDType else fallback) =
      (if n < h.N ∧ k < h.K then
        ofFloat (dtype h.storage) (data.weights (weightOffset h slot n k)) else fallback) := by
    by_cases active : n < h.N ∧ k < h.K
    · simp only [active, if_true, weightRead n k active.1 active.2, Cell.read_same]
    · simp only [active, if_false]
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    simp only [inputOffset,weightOffset,dtype,FloatDType.toTileDType,ofFloat,
      Nat.mul_comm slot (N*K)] at inputLoad weightLoad
    simp only [body,active,code,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,
      ComputeKernel.surfaceBody,List.drop,Source.conditional?,Source.intRange?,
      Option.map_some,Option.getD_some,beq_self_eq_true]
    simp (config := {maxSteps := 100000}) [
      Source.intRange?,Source.conditional?,ComputeKernel.surfaceBody,List.drop,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,ofFloat,toFloat,TileShape.axisDim,Region.cast,Data.cells,dtype,
      ready.memory,ready.scaleReg,ready.batchReg,ready.offsetN,ready.offsetK,ready.inputPtr,ready.weightPtr,ready.acc,
      Broadcast.leftIndex,Broadcast.rightIndex,rebase]
    have negativeRank (e : Nat) :
        Int.toNat (-((split:Int)*256) + -(e:Int) + -((c:Int)*16384)) = 0 := by omega
    have rankCast (e : Nat) : (c:Int)*16384 + ((e:Int)+(split:Int)*256) =
        ((c*16384+(e+split*256):Nat):Int) := by norm_cast
    simp only [negativeRank,rankCast,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt,
      FPInvariant.negative_nat,FPInvariant.negative_sum,Nat.sub_zero]
    simp only [inputLoad,weightLoad]
    rfl

#axiomsClean body_transport
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
