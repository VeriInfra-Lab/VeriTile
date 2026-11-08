import bench.tritonbench_g.rms_matmul_rbe.Proofs.Source

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

theorem prelude_format (fp8 outputFP8 : Bool) (p : Arguments) :
    prelude fp8 outputFP8 p=prelude Bool.false Bool.false p := by
  cases fp8 <;> cases outputFP8 <;> rfl

/-- Compute the format-independent integer/pointer prologue once. -/
def prefixBaseExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Arguments) (s : State α) :
    {t : State α // run M (prelude Bool.false Bool.false p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.rmsKernel,ComputeKernel.surfaceBody,run,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
      Region.cast,Option.bind,integralFloorDiv,integralMod]
  rfl

/-- Successful execution is constructed from the caller's state. -/
def prefixExecution {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    {t : State α // run M (prelude fp8 outputFP8 p) s=some t} :=
  ⟨(prefixBaseExecution M p s).val,by
    rw [prelude_format]
    exact (prefixBaseExecution M p s).property⟩

def isAssignment : ComputeStmt → Bool
  | .assign _ _ _ _=>Bool.true
  | _=>Bool.false

theorem assignment_frame {α : Type} [Inhabited α] (M : Algebra α)
    (statement : ComputeStmt) (pure : isAssignment statement=Bool.true)
    (s t : State α) (ran : step M statement s=some t) : t.mem=s.mem := by
  cases statement <;> simp only [isAssignment] at pure
  all_goals try contradiction
  case assign d sh name value =>
    cases h : evalExpr M value s <;> simp only [step,h,Option.bind_none,Option.bind_some] at ran
    · contradiction
    · cases ran
      rfl

theorem assignments_frame {α : Type} [Inhabited α] (M : Algebra α)
    (code : List ComputeStmt) (pure : code.all isAssignment=Bool.true)
    (s t : State α) (ran : run M code s=some t) : t.mem=s.mem := by
  induction code generalizing s with
  | nil=>simp only [run,Option.some.injEq] at ran; cases ran; rfl
  | cons statement rest ih=>
    have facts : isAssignment statement=Bool.true ∧ rest.all isAssignment=Bool.true := by
      simpa only [List.all_cons,Bool.and_eq_true] using pure
    cases h : step M statement s with
    | none=>simp only [run,h,Option.bind_none] at ran; contradiction
    | some mid=>
      have tail : run M rest mid=some t := by
        simpa only [run,h,Option.bind_some] using ran
      exact (ih facts.2 mid tail).trans (assignment_frame M statement facts.1 s mid h)

theorem prefix_memory {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    (prefixExecution M fp8 outputFP8 p s).val.mem=s.mem := by
  apply assignments_frame M (prelude fp8 outputFP8 p) _ s _
    (prefixExecution M fp8 outputFP8 p s).property
  cases fp8 <;> cases outputFP8 <;> rfl

def assignmentAway (name : RegName) : ComputeStmt → Bool
  | .assign _ _ target _=>decide (target≠name)
  | _=>Bool.false

theorem assignment_register {α : Type} [Inhabited α] (M : Algebra α)
    (statement : ComputeStmt) (name : RegName) (pure : assignmentAway name statement=Bool.true)
    (s t : State α) (ran : step M statement s=some t) (d : TileDType) (sh : TileShape) :
    t.regs d sh name=s.regs d sh name := by
  cases statement <;> simp only [assignmentAway] at pure
  all_goals try contradiction
  case assign dtype shape target value=>
    have different : target≠name := of_decide_eq_true pure
    cases h : evalExpr M value s <;> simp only [step,h,Option.bind_none,Option.bind_some] at ran
    · contradiction
    · cases ran
      simp only [State.setReg,if_neg different.symm]

theorem assignments_register {α : Type} [Inhabited α] (M : Algebra α)
    (code : List ComputeStmt) (name : RegName) (pure : code.all (assignmentAway name)=Bool.true)
    (s t : State α) (ran : run M code s=some t) (d : TileDType) (sh : TileShape) :
    t.regs d sh name=s.regs d sh name := by
  induction code generalizing s with
  | nil=>simp only [run,Option.some.injEq] at ran; cases ran; rfl
  | cons statement rest ih=>
    have facts : assignmentAway name statement=Bool.true ∧ rest.all (assignmentAway name)=Bool.true := by
      simpa only [List.all_cons,Bool.and_eq_true] using pure
    cases h : step M statement s with
    | none=>simp only [run,h,Option.bind_none] at ran; contradiction
    | some mid=>
      have tail : run M rest mid=some t := by simpa only [run,h,Option.bind_some] using ran
      exact (ih facts.2 mid tail).trans (assignment_register M statement name facts.1 s mid h d sh)

theorem assignment_pids {α : Type} [Inhabited α] (M : Algebra α)
    (statement : ComputeStmt) (pure : isAssignment statement=Bool.true)
    (s t : State α) (ran : step M statement s=some t) : t.pids=s.pids := by
  cases statement <;> simp only [isAssignment] at pure
  all_goals try contradiction
  case assign d sh name value=>
    cases h : evalExpr M value s <;> simp only [step,h,Option.bind_none,Option.bind_some] at ran
    · contradiction
    · cases ran
      rfl

theorem assignments_pids {α : Type} [Inhabited α] (M : Algebra α)
    (code : List ComputeStmt) (pure : code.all isAssignment=Bool.true)
    (s t : State α) (ran : run M code s=some t) : t.pids=s.pids := by
  induction code generalizing s with
  | nil=>simp only [run,Option.some.injEq] at ran; cases ran; rfl
  | cons statement rest ih=>
    have facts : isAssignment statement=Bool.true ∧ rest.all isAssignment=Bool.true := by
      simpa only [List.all_cons,Bool.and_eq_true] using pure
    cases h : step M statement s with
    | none=>simp only [run,h,Option.bind_none] at ran; contradiction
    | some mid=>
      have tail : run M rest mid=some t := by simpa only [run,h,Option.bind_some] using ran
      exact (ih facts.2 mid tail).trans (assignment_pids M statement facts.1 s mid h)

theorem prefix_pids {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    (prefixExecution M fp8 outputFP8 p s).val.pids=s.pids := by
  apply assignments_pids M (prelude fp8 outputFP8 p) _ s _
    (prefixExecution M fp8 outputFP8 p s).property
  cases fp8 <;> cases outputFP8 <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
