import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedIteration
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def loopCode (p : Config) (BD : Nat) (second : Bool) : List ComputeStmt :=
  [.forRangeInt "start_n" (lowerOp p second)
    (.ref .int [] (upperName second)) (.constNat 64) (body p BD second)]

theorem loop_success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (support : DotSupport A BD) (second : Bool)
    (s : State α) (hs : Ready BD s) (hl : LoopReady p BD second s) :
    ∃ t, run (engine A) (loopCode p BD second) s=some t ∧ Ready BD t := by
  have hsave := hl
  obtain ⟨_,_,_,⟨lo,hlo⟩,⟨hi,hhi⟩⟩ := hl
  obtain ⟨last,t,ht,_,hr,hl'⟩ := ControlView.signed_range_invariant (engine A)
    (idx:="start_n") (stop:=hi PUnit.unit) (stride:=64) (body:=body p BD second)
    (P:=fun _ t => Ready BD t ∧ LoopReady p BD second t) (by decide)
    (fun cur t _ h => iteration_success A p BD support second t h.1 h.2 cur)
    (lo PUnit.unit) s ⟨hs,hsave⟩
  refine ⟨t,?_ ,hr⟩
  simp only [loopCode,run,step]
  rw [eval_lower,hlo]
  simp only [evalOp_unfold,hhi,bind,Option.bind]
  rw [ht]

theorem tail_success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (r : Bool) (s : State α) (hs : Ready BD s) :
    ∃ t, run (engine A) (tail p BD r) s=some t := by
  rcases hs with ⟨⟨x0,h0⟩,⟨x1,h1⟩,⟨x2,h2⟩,⟨x3,h3⟩,⟨x4,h4⟩,⟨x5,h5⟩,⟨x6,h6⟩,⟨x7,h7⟩,⟨x8,h8⟩,⟨x9,h9⟩,⟨x10,h10⟩,⟨x11,h11⟩,⟨x12,h12⟩,⟨x13,h13⟩,⟨x14,h14⟩⟩
  cases r <;> by_cases hd : p.stage=2
  all_goals simp only [hd,tail,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,ControlView.compute,List.drop_succ_cons,List.drop_zero]
  all_goals simp [hd,ControlView.compute,run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    Cell.read,ofFloat,toFloat,Region.cast,State.setReg,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14]

theorem assembled (p : Config) (BD : Nat) (r : Bool) :
    code p BD =
      coordinateCode p BD ++ pointerCode p BD ++ initializationCode p BD ++
      setupCode p BD Bool.false ++ loopCode p BD Bool.false ++
      setupCode p BD Bool.true ++ loopCode p BD Bool.true ++ tail p BD r := by
  cases hd : decide (p.stage=2)
  all_goals simp only [coordinateCode,pointerCode,initializationCode,setupCode,loopCode,body,tail,
    lowerOp,lowerName,upperName,code,source,TypedSource.repaired,ComputeKernel.surfaceBody,hd,
    Bool.false_or,Bool.true_or,Bool.false_eq_true,Bool.true_eq,↓reduceIte,ControlView.compute,
    List.cons_append,List.take_succ_cons,List.take_zero,List.drop_succ_cons,List.drop_zero]
  all_goals rfl

/-- Every actual source instruction succeeds for arbitrary input cells and
opaque arithmetic, provided the two exact dot backends are implemented.
No Real identities, finite inputs, output law or successful-run premise. -/
theorem success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (support : DotSupport A BD) (r : Bool) (s : State α) :
    ∃ t, exec (engine A) (source p BD r) s=some t := by
  obtain ⟨s1,h1,hr1⟩ := coordinates_success A p BD s
  obtain ⟨s2,h2,hr2⟩ := pointers_success A p BD s1 hr1
  obtain ⟨s3,h3,hr3⟩ := initialization_success A p BD s2 hr2
  obtain ⟨s4,h4,hr4,hl4⟩ := setup_success A p BD Bool.false s3 hr3
  obtain ⟨s5,h5,hr5⟩ := loop_success A p BD support Bool.false s4 hr4 hl4
  obtain ⟨s6,h6,hr6,hl6⟩ := setup_success A p BD Bool.true s5 hr5
  obtain ⟨s7,h7,hr7⟩ := loop_success A p BD support Bool.true s6 hr6 hl6
  obtain ⟨t,ht⟩ := tail_success A p BD r s7 hr7
  refine ⟨t,?_⟩
  change run (engine A) (source p BD r).surfaceBody s=some t
  rw [←code_run A p BD r s, assembled p BD r]
  simp only [run_append,h1,h2,h3,h4,h5,h6,h7,ht,Option.bind_some]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
