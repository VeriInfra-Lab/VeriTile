import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.Reference

namespace VeriTile.Bench.Repairs.ContextAttnMistral.KernelReference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral Formula
open VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program (Regions)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- All loop-carried values, including the complete floating graph, follow
one independently defined recurrence step. -/
theorem body (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (initialState s : State α) (old : Acc p α)
    (live : Matches p r o (inputs p r initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) :
    ∃ t,run (engine o.algebra) (parts p r .fp32 scale).body
        (s.setReg "start_n" .int [] (fun _ => cur))=some t ∧
      Matches p r o (inputs p r initialState) (advance p o scale (inputs p r initialState) cur old) t ∧
      t.mem=initialState.mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13⟩
  simp (config := {maxSteps := 1000000}) [parts,program,
    VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,
    resolvePrecision,numeric,numericLt,numericLe,numericEq,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,
    TileShape.axisDim,hBN,Matches,advance,keyTile,valueTile,keyPointer,valuePointer,
    memory,inputs,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,keyTile,valueTile,Function.comp_def]
  refine ⟨?_,?_,?_⟩ <;> intros
  all_goals congr 100

end
end VeriTile.Bench.Repairs.ContextAttnMistral.KernelReference
