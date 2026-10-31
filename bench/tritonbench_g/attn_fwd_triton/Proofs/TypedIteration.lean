import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedSetup
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Availability of the two actual dot operations. No scalar formula,
rounding rule, cast law or output equality is imposed. -/
structure DotSupport {α : Type} (A : Algebra α) (BD : Nat) where
  score : Values α .real [128,BD] → Values α .real [BD,64] → Values α .real [128,64]
  score_available : ∀ q k, (engine A).dot (some .fp32) (batch:=[]) q k=some (score q k)
  value : Values α .fp16 [128,64] → Values α .fp16 [64,BD] → Values α .fp16 [128,BD]
  value_available : ∀ q k, (engine A).dotTyped (some .fp32) .fp16 .fp16 .fp16 {outDType:=some .fp16}
    (batch:=[]) q k=some (value q k)

theorem iteration_success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (support : DotSupport A BD) (second : Bool)
    (s : State α) (hs : Ready BD s) (hl : LoopReady p BD second s) (cursor : Int) :
    ∃ t, run (engine A) (body p BD second) (s.setReg "start_n" .int [] (fun _ => cursor))=some t ∧
      Ready BD t ∧ LoopReady p BD second t := by
  rcases hs with ⟨⟨x0,h0⟩,⟨x1,h1⟩,⟨x2,h2⟩,⟨x3,h3⟩,⟨x4,h4⟩,⟨x5,h5⟩,⟨x6,h6⟩,⟨x7,h7⟩,⟨x8,h8⟩,⟨x9,h9⟩,⟨x10,h10⟩,⟨x11,h11⟩,⟨x12,h12⟩,⟨x13,h13⟩,⟨x14,h14⟩⟩
  rcases hl with ⟨⟨kp,hkp⟩,⟨ksp,hksp⟩,⟨vp,hvp⟩,⟨lo,hlo⟩,⟨hi,hhi⟩⟩
  cases second <;> by_cases hd : p.stage=2
  all_goals simp [keyName,scaleName,valueName,lowerName,upperName,lowerValue,hd] at hkp hksp hvp hlo hhi
  all_goals simp only [body,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,ControlView.compute,Source.intRange?,
    Option.map_some,Option.getD_some,beq_self_eq_true,List.drop_succ_cons,List.drop_zero]
  all_goals simp [hd,Source.intRange?,ControlView.compute,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,Ready,
    LoopReady,lowerValue,keyName,scaleName,valueName,lowerName,upperName,TileShape.axisDim,
    support.score_available,support.value_available,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,hkp,hksp,hvp,hlo,hhi]

  all_goals rcases hlo with ⟨a,ha,_⟩; exact ⟨a,ha⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution

