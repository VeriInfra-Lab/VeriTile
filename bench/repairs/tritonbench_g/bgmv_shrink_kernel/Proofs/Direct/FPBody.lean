import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.FPSetup

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.FPBody
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Invariant (h : Call) (data : Data α) (batch split slot : Nat)
    (a : Fin h.BN → α) (s : State α) : Prop where
  memory : s.mem=data.cells h
  batchReg : s.regs .nat [] "cur_batch"=some (fun _ => batch)
  offsetN : s.regs .nat [h.BN] "offset_n"=some (fun i => i.1.val)
  offsetK : s.regs .nat [256] "offset_k"=some (fun i => i.1.val+split*256)
  inputPtr : s.regs .ptr [] "a_ptr"=some (fun _ => (("X":RegionName),batch*h.K))
  weightPtr : s.regs .ptr [] "b_ptr"=some (fun _ => (("W":RegionName),(h.N*h.K)*slot))
  acc : s.regs .real [h.BN] "accumulator"=some (fun i => a i.1)

/-- A complete actual typed iteration from only entry memory/register facts.
The result is constructed by the interpreter; execution is not a premise. -/
def iteration (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot c : Nat) (data : Data α) (a : Fin h.BN → α) (s : State α)
    (ready : Invariant h data batch split slot a s) :
    {t : State α // run (machine M P) (body h)
      (s.setReg "k" .nat [] (fun _ => c*16384))=some t} := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [body,active,code,Kernels.source,Kernels.original,
      VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedSource.original,
      VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedSource.specialize,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,
      Option.map_some,Option.getD_some,beq_self_eq_true]
    simp (config := {maxSteps := 300000}) [Source.dynamicRange?,Option.map,Option.getD,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,Data.cells,dtype,
      ready.memory,ready.batchReg,ready.offsetN,ready.offsetK,ready.inputPtr,ready.weightPtr,ready.acc]
    try rfl

#axiomsClean iteration
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.FPBody
