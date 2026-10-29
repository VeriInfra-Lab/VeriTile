import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.SourceStoreSafety

namespace VeriTile.Bench.Repairs.RotaryTransformOps.SourceSafetyTools
open VeriTile.Triton FP.Structural TypedExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

/-- Compose through a known actual successor without reducing either program. -/
theorem append_safe {M : Algebra α} {bounds : RegionName → Nat}
    {before after : List ComputeStmt} {initial middle : State α}
    (first : Safety.traceSafeList M bounds before initial)
    (ran : run M before initial=some middle)
    (last : Safety.traceSafeList M bounds after middle) :
    Safety.traceSafeList M bounds (before++after) initial := by
  apply (Safety.traceSafeList_append M bounds before after initial).mpr
  refine ⟨first,?_⟩
  intro next executed
  have eq : next=middle := Option.some.inj (executed.symm.trans ran)
  subst next
  exact last

theorem if_not_safe (M : Algebra α) (bounds : RegionName → Nat) (flag : Bool)
    (yes no : List ComputeStmt) (s : State α)
    (body : if flag then Safety.traceSafeList M bounds no s else Safety.traceSafeList M bounds yes s) :
    Safety.traceSafeList M bounds [.ifThenElse (.boolNot (.constBool flag)) yes no] s := by
  apply Safety.traceSafeList_cons_intro
  · cases flag <;> simpa [Safety.stepSafe,Safety.safeOp,evalOp_unfold] using body
  · intro t _
    exact Safety.traceSafeList_nil M bounds t

theorem if_safe (M : Algebra α) (bounds : RegionName → Nat) (cond : Op .bool [])
    (body : List ComputeStmt) (s : State α)
    (condition : Safety.safeOp M none bounds s cond)
    (safe : Safety.traceSafeList M bounds body s) :
    Safety.traceSafeList M bounds [.ifThen cond body] s := by
  apply Safety.traceSafeList_cons_intro
  · rw [Safety.stepSafe_unfold]
    refine ⟨condition,?_⟩
    cases evaluated : evalOp M none cond s with
    | none => trivial
    | some value =>
      by_cases active : value PUnit.unit=Bool.true
      · simp only [active,if_true]
        exact safe
      · simp only [active,Bool.false_eq_true,if_false]
  · intro t _
    exact Safety.traceSafeList_nil M bounds t

/-- Each slice is from the selected quotation and uses its execution witness. -/
theorem payload_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat)
    (initial : State α) (bound : AccessBounds s initial bounds) :
    Safety.traceSafeList M bounds (payload s) (indices M s initial).val := by
  have joined := append_safe (loads_safe M s bounds initial bound)
    (loads M s initial).property (stores_safe M s bounds initial bound)
  exact Eq.mp (congrArg (fun code => Safety.traceSafeList M bounds code (indices M s initial).val)
    (List.take_append_drop (loadsCount s) (payload s))) joined

#axiomsClean payload_safe

theorem continuation_tail_code (s : Shape) :
    (continuation s).drop 4 =
      [.ifThenElse (.boolNot (.constBool s.interleaved))
        (splitBody s) (interleavedBody s)] := by
  cases hv : s.varlen <;> cases ht : s.offsetTensor <;>
    simp [continuation,sourceBody,SourceBlocks.body,SourceBlocks.select,SourceBlocks.setupCount,
      splitBody,interleavedBody,branchBody,ifBody,kernel,KernelSource.source,KernelSource.source64_64,
      KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap,hv,ht]

theorem continuation_tail_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat)
    (u : State α) (hp : Safety.traceSafeList M bounds (payload s) u) :
    Safety.traceSafeList M bounds ((continuation s).drop 4) u := by
  have selected : if s.interleaved then
      Safety.traceSafeList M bounds (interleavedBody s) u else
      Safety.traceSafeList M bounds (splitBody s) u := by
    cases hi : s.interleaved
    · change Safety.traceSafeList M bounds (splitBody s) u
      have eq : payload s=splitBody s := by simp only [payload,splitBody,hi]
      exact Eq.mp (congrArg (fun code => Safety.traceSafeList M bounds code u) eq) hp
    · change Safety.traceSafeList M bounds (interleavedBody s) u
      have eq : payload s=interleavedBody s := by simp only [payload,interleavedBody,hi]
      exact Eq.mp (congrArg (fun code => Safety.traceSafeList M bounds code u) eq) hp
  have branchRaw : Safety.traceSafeList M bounds
      [.ifThenElse (.boolNot (.constBool s.interleaved)) (splitBody s) (interleavedBody s)] u :=
    if_not_safe M bounds s.interleaved (splitBody s) (interleavedBody s) u selected
  exact Eq.mpr (congrArg (fun code => Safety.traceSafeList M bounds code u)
    (continuation_tail_code s)) branchRaw

theorem continuation_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat)
    (initial : State α) (bound : AccessBounds s initial bounds)
    (offset : s.offsetTensor = Bool.true → initial.pids 1<bounds "offsets") :
    Safety.traceSafeList M bounds (continuation s) (setup M s Bool.true Bool.true initial).val := by
  have branch := continuation_tail_safe M s bounds (indices M s initial).val
    (payload_safe M s bounds initial bound)
  have joined := append_safe (indices_safe M s bounds initial offset)
    (indices M s initial).property branch
  exact Eq.mp (congrArg (fun code => Safety.traceSafeList M bounds code (setup M s Bool.true Bool.true initial).val)
    (List.take_append_drop 4 (continuation s))) joined

#axiomsClean continuation_safe

theorem guard_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat) (u : State α) :
    Safety.safeOp M none bounds u (guard s) := by
  unfold TypedExecution.guard
  cases hv : s.varlen <;> cases ht : s.offsetTensor
  all_goals
    simp only [guardOf,sourceBody,SourceBlocks.body,SourceBlocks.select,
      SourceBlocks.setupCount,kernel,KernelSource.source,KernelSource.source64_64,
      KernelSource.source64_64_tensor,KernelSource.source64_32_scalar,ComputeKernel.surfaceBody,
      List.take,List.drop,List.flatMap_cons,List.flatMap_nil,List.append_nil,
      List.cons_append,List.nil_append,hv,ht,Bool.false_eq_true,if_false,if_true]
    simp [Safety.safeOp,Safety.safeOp_unfold]

theorem source_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat)
    (initial : State α) (bound : AccessBounds s initial bounds)
    (cu : s.varlen = Bool.true → initial.pids 1+1<bounds "cu")
    (offset : s.offsetTensor = Bool.true → initial.pids 1<bounds "offsets")
    (cumulative64 offsets64 : Bool) :
    TraceSafe M bounds (kernel s cumulative64 offsets64) initial := by
  have tail : Safety.traceSafeList M bounds ((sourceBody s).drop (SourceBlocks.setupCount s.varlen))
      (setup M s Bool.true Bool.true initial).val :=
    Eq.mpr (congrArg (fun code => Safety.traceSafeList M bounds code (setup M s Bool.true Bool.true initial).val)
      (source_tail s))
      (if_safe M bounds (guard s) (continuation s) _ (guard_safe M s bounds _)
        (continuation_safe M s bounds initial bound offset))
  have joined := append_safe (setup_safe M s bounds initial cu)
    (setup M s Bool.true Bool.true initial).property tail
  have selected : Safety.traceSafeList M bounds (sourceBody s) initial :=
    Eq.mp (congrArg (fun code => Safety.traceSafeList M bounds code initial)
      (List.take_append_drop (SourceBlocks.setupCount s.varlen) (sourceBody s))) joined
  have canonical : TraceSafe M bounds (kernel s Bool.true Bool.true) initial :=
    (source_body_safe_iff M s bounds initial).mpr selected
  exact Eq.mpr (congrArg (fun selected => TraceSafe M bounds selected initial)
    (kernel_carrier s cumulative64 offsets64)) canonical

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.RotaryTransformOps.SourceSafetyTools
