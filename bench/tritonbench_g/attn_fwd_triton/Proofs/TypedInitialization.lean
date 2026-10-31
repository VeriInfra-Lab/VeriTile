import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedPointers
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def initializationCode (p : Config) (BD : Nat) := ((code p BD).drop 22).take 5

theorem initialization_success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (s : State α) (hs : PointerReady BD s) :
    ∃ t, run (engine A) (initializationCode p BD) s=some t ∧ Ready BD t := by
  rcases hs with ⟨⟨x0,h0⟩,⟨x1,h1⟩,⟨x2,h2⟩,⟨x3,h3⟩,⟨x4,h4⟩,⟨x5,h5⟩,⟨x6,h6⟩,⟨x7,h7⟩,⟨x8,h8⟩,⟨x9,h9⟩,⟨x10,h10⟩,⟨x11,h11⟩⟩
  simp only [initializationCode,code,source,Bool.false_eq_true,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,ControlView.compute,List.drop_succ_cons,List.drop_zero,
    List.take_succ_cons,List.take_zero]
  simp [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,numericLt,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,Ready,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution

