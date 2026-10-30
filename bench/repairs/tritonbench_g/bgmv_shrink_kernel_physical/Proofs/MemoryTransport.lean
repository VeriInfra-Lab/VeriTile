import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPExecution

/-! Transport existing constructive source states to an arbitrary backing
memory. Only actually read cells are compared; exterior tags stay unchanged. -/
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup FPBody
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def rebase (s : State α) (memory : RegionName → Nat → Cell α) : State α :=
  {s with mem := memory}

/-- The source metadata load observes exactly the supplied INT64 cell. -/
theorem prefix_transport (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split : Nat) (data : Data α) (memory : RegionName → Nat → Cell α)
    (index : memory "Indices" batch = .mk .int (data.indices batch)) :
    run (machine M P) (prelude h) (rebase (entry h batch split data) memory) =
      some (rebase (preguard M P h batch split data).val memory) := by
  have indexRead : (memory "Indices" batch).read .int = data.indices batch := by rw [index]; rfl
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    simp [prelude,code,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,ComputeKernel.surfaceBody,List.drop,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      preguard,rebase,entry,Data.cells,State.setReg,Region.cast,indexRead]
    <;> simp only [Cell.read]

theorem setup (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int))
    (memory : RegionName → Nat → Cell α) :
    run (machine M P) (FPProgram.setup h) (rebase (preguard M P h batch split data).val memory) =
      some (rebase (preloop M P h batch split slot data metadata).val memory) := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    simp (config := {maxSteps := 400000}) [FPProgram.setup,active,code,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,
      Source.intRange?,Source.conditional?,ComputeKernel.surfaceBody,List.drop,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      preloop,preguard,rebase,entry,Data.cells,State.setReg,Cell.read,Region.cast,metadata,
      machine,Algebra.withDefaultPrecision,resolvePrecision,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,ofFloat,toFloat]

#axiomsClean prefix_transport
#axiomsClean setup
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
