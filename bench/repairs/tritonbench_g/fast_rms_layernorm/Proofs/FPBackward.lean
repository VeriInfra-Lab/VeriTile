import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPSource
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPBackwardReference

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def expectedHeap (M : Algebra α) (h : Call) (u : State α) : RegionName → Nat → Cell α :=
  let x:=fun j : Fin h.block => (u.mem "x" (u.pids 0*h.cols+j.val)).read .real
  let g:=fun j : Fin h.block => (u.mem "dx" (u.pids 0*h.cols+j.val)).read .real
  let w:=fun j : Fin h.block => (u.mem "w" j.val).read .real
  let saved:=(u.mem "r" (u.pids 0)).read .real
  commit "dx" (fun i : TileIndex [h.block] => u.pids 0*h.cols+i.1.val)
    (fun i => FPBackwardReference.output M h x w g saved i.1) (fun i => i.1.val<h.cols)
    (TileShape.allIndices [h.block]) u.mem

theorem source_memory (M : Algebra α) (h : Call) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.backward h) u=some v ∧ v.mem=expectedHeap M h u := by
  cases mode : h.gemma
  all_goals
    simp [mode,ofFloat,toFloat,FP.Structural.exec,Kernels.backward,Kernels.inputGradientSource,Kernels.inputGradientBody,Original.backward,
      VeriTile.Bench.TritonBenchG.FastRmsLayernorm.TypedSource.backward,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      expectedHeap,FPBackwardReference.output,FPBackwardReference.rowSum,
      FPBackwardReference.normed,FPBackwardReference.weighted,
      FPReference.reduce,FPReference.input,FPReference.padded,FPReference.literal,FPReference.count,FPReference.cast,
      FPReference.add,FPReference.mul,FPReference.sub,FPReference.div,fold_mem]
    <;> (try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,State.write,FPSource.update])
    <;> norm_num
    <;> rfl

#axiomsClean source_memory
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPBackward
