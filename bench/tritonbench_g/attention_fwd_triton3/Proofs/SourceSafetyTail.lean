import bench.tritonbench_g.attention_fwd_triton3.Proofs.SourceSafetySteps

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
open VeriTile Triton FP.Structural Safety FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
open Reference
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
attribute [local simp 1100] FP.Structural.eval_truncDiv

theorem if_else_safe (A : Algebra α) (bounds : RegionName → Nat) (s : State α)
    (condition : Op .bool []) (yes no : List ComputeStmt)
    (checked : safeOp A none bounds s condition)
    (hy : traceSafeList A bounds yes s) (hn : traceSafeList A bounds no s) :
    stepSafe A bounds (.ifThenElse condition yes no) s := by
  rw [stepSafe_unfold]
  refine ⟨checked,?_⟩
  cases evalOp A none condition s with
  | none => trivial
  | some value => dsimp; split <;> assumption

theorem if_safe (A : Algebra α) (bounds : RegionName → Nat) (s : State α)
    (condition : Op .bool []) (yes : List ComputeStmt)
    (checked : safeOp A none bounds s condition)
    (hy : traceSafeList A bounds yes s) : stepSafe A bounds (.ifThen condition yes) s := by
  rw [stepSafe_unfold]
  refine ⟨checked,?_⟩
  cases evalOp A none condition s with
  | none => trivial
  | some value => dsimp; split <;> trivial

theorem safeAdvance2 (A : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (ptr : Op .blockPtr shape)
    (a b : Op .int []) :
    safeOp A precision bounds s (.advanceBlockPtrDyn ptr [a,b]) ↔
      safeOp A precision bounds s ptr ∧ safeOp A precision bounds s a ∧ safeOp A precision bounds s b := by
  rw [safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

syntax "prove_memory_free" : tactic
macro_rules
| `(tactic| prove_memory_free) => `(tactic|
  first
  | exact traceSafeList_nil _ _ _
  | (apply traceSafeList_cons_intro
     · prove_memory_free
     · intro next _; prove_memory_free)
  | (apply if_else_safe
     · prove_memory_free
     · prove_memory_free
     · prove_memory_free)
  | (apply if_safe
     · prove_memory_free
     · prove_memory_free)
  | (try simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
     try rw [safeAdvance2]
     simp only [safeOp_unfold,true_and,and_true]))

theorem window_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat)
    (initialState s : State α) (_live : Started c o (inputs initialState) s) :
    traceSafeList (engine o.algebra) bounds ((parts c).before.drop 20) s := by
  prove_memory_free

theorem epilogue_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat)
    (a : Inputs α) (old : Carry c α) (s : State α) (live : Matches c o a old s)
    (backed : PreludeBounds c a bounds) :
    traceSafeList (engine o.algebra) bounds (parts c).after s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12⟩
  rcases backed with ⟨query,center,denominator,output⟩
  have ho : ∀ (r : Fin c.BM) (d : Fin c.D),
      (optr c a).address (TileShape.indexToList [c.BM,c.D] (r,d,PUnit.unit))<bounds (optr c a).region := by
    simpa only [block_bounds_2d,BlockPtr.inBounds_nil_checkedAxes,forall_true_left] using output
  by_cases hf : c.finish=0
  all_goals
    simp (config := {maxSteps := 500000}) [parts,Kernels.source,Quoted.kernel,
      ComputeKernel.surfaceBody,Source.intRange?,traceSafeList_unfold,stepSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,
      addressSafe_unfold,maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,
      FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
      natLt,natLe,ofFloat,toFloat,Region.cast,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,hf,
      store,bop,Broadcast.leftIndex,Broadcast.rightIndex,State.scatter_regs]
  · exact ⟨denominator (Or.inr hf),center,ho⟩
  · exact ⟨center,ho⟩

theorem append_of_run {M : Algebra α} {bounds : RegionName → Nat}
    {before after : List ComputeStmt} {s t : State α}
    (safe : traceSafeList M bounds before s) (ran : run M before s=some t)
    (tail : traceSafeList M bounds after t) : traceSafeList M bounds (before++after) s := by
  apply (traceSafeList_append M bounds before after s).mpr
  refine ⟨safe,?_⟩
  intro u hu
  have eq := Option.some.inj (ran.symm.trans hu)
  simpa only [←eq] using tail

theorem prelude_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat) (s : State α)
    (backed : PreludeBounds c (inputs s) bounds) :
    traceSafeList (engine o.algebra) bounds (parts c).before s := by
  obtain ⟨headerState,hheader,headerLive,headerMemory⟩ := header c o s
  obtain ⟨startedState,hstarted,startedLive,startedMemory⟩ :=
    initialization c o s headerState headerLive headerMemory
  have split : (parts c).before=(parts c).before.take 16 ++
      (((parts c).before.drop 16).take 4 ++ (parts c).before.drop 20) := rfl
  rw [split]
  exact append_of_run (header_safe c o bounds s) hheader
    (append_of_run (initialization_safe c o bounds s headerState headerLive backed) hstarted
      (window_safe c o bounds s startedState startedLive))

#axiomsClean prelude_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
