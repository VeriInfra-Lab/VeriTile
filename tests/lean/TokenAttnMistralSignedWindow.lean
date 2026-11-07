import bench.tritonbench_g.token_attn_mistral.Proofs.Execution
namespace VeriTile.Tests.TokenAttnMistralSignedWindow
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnMistral
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

private theorem integer_max (a : Int) (w : Nat) :
    (if (w : Int)<a then a-w else 0)=max (a-w) 0 := by
  rw [max_def]
  split <;> split <;> omega

/-- The actual prelude computes the integer sliding origin for arbitrary
signed length words, before any mathematical nonnegativity assumption. -/
theorem integer_window {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    (run (withSum M R) (prelude p) s).map (fun t=>t.regs .int [] "cur_batch_start_index")=
      some (some (fun _=>max
        ((s.mem (Region.cast p.length) (s.pids 0*p.sequenceStride)).read .int-(p.window : Int)) 0)) := by
  simp [prelude,kernel,sourceKernel,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,integralFloorDiv,
    numericLt,withSum,Region.cast,State.setReg]
  funext i
  exact integer_max _ _

#axiomsClean integer_window
end
end VeriTile.Tests.TokenAttnMistralSignedWindow
