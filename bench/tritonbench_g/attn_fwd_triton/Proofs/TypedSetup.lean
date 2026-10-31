import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedInitialization
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def setupCode (p : Config) (BD : Nat) (second : Bool) :=
  ((code p BD).drop (if second then 34 else 27)).take 6

theorem setup_success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (second : Bool) (s : State α) (hs : Ready BD s) :
    ∃ t, run (engine A) (setupCode p BD second) s=some t ∧ Ready BD t ∧ LoopReady p BD second t := by
  rcases hs with ⟨⟨x0,h0⟩,⟨x1,h1⟩,⟨x2,h2⟩,⟨x3,h3⟩,⟨x4,h4⟩,⟨x5,h5⟩,⟨x6,h6⟩,⟨x7,h7⟩,⟨x8,h8⟩,⟨x9,h9⟩,⟨x10,h10⟩,⟨x11,h11⟩,⟨x12,h12⟩,⟨x13,h13⟩,⟨x14,h14⟩⟩
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [setupCode,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,ControlView.compute,List.drop_succ_cons,List.drop_zero,
    List.take_succ_cons,List.take_zero]
  all_goals simp [hd,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,numericLt,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,Ready,
    LoopReady,lowerValue,keyName,scaleName,valueName,lowerName,upperName,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution

