import bench.tritonbench_g.int8_quantization.Proofs.SourceExecution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.SourceSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem trace_from_steps (M : Algebra α) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (safe : ∀ st ∈ code, ∀ s, stepSafe M bounds st s) (s : State α) :
    traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun cmd member => safe cmd (by simp [member])) t

theorem prefix_safe (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i, Model.active p s i → Model.offset p s i < bounds xRegion) :
    traceSafeList (Model.backend M ops) bounds (segment p 0 8) s := by
  simp [segment,source,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Model.backend,Option.bind,
    numericLt,numericLe,numericEq,natLt,natLe,bop,numeric,ofFloat,toFloat,
    Region.cast,ComputeDType.eraseDType,integralMod,integralFloorDiv]
  intro i first second
  exact input (i,PUnit.unit) ⟨first,second⟩

/-- The complete scalar/reduction middle has no memory effects. This proof
checks every actual statement, including its conditional branch. -/
theorem arithmetic_safe (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    traceSafeList (Model.backend M ops) bounds (segment p 8 13) s := by
  apply trace_from_steps
  intro st member t
  simp only [segment,source,ComputeKernel.surfaceBody,List.drop,List.take,
    List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals
    cases query : p.query <;>
      simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
        traceSafeList_unfold,step,evalOp_unfold,evalExpr,Option.bind,query]
  all_goals split <;> trivial

theorem stores_safe (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0 < p.width) (s : State α) (bounds : RegionName → Nat)
    (output : ∀ i, Model.active p s i → Model.offset p s i < bounds outRegion)
    (scale : Model.group p s < bounds scaleRegion) :
    traceSafeList (Model.backend M ops) bounds (segment p 21 2) (registers M ops p positive s) := by
  simp [segment,source,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Model.backend,Option.bind,numericEq,bop,numeric,
    FP.Structural.store,Region.cast,ComputeDType.eraseDType,
    VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_regs]
  exact ⟨fun i u active => output (i,u) active,scale⟩

theorem source_safe (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0 < p.width) (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i, Model.active p s i → Model.offset p s i < bounds xRegion)
    (output : ∀ i, Model.active p s i → Model.offset p s i < bounds outRegion)
    (scale : Model.group p s < bounds scaleRegion) :
    TraceSafe (Model.backend M ops) bounds (source p) s := by
  have split : (source p).surfaceBody = segment p 0 8 ++ segment p 8 13 ++ segment p 21 2 := rfl
  have middle : run (Model.backend M ops) (segment p 8 13) (addressed M p s) =
      some (registers M ops p positive s) := by
    have parts : segment p 8 13 = segment p 8 1 ++ segment p 9 3 ++ segment p 12 4 ++ segment p 16 5 := rfl
    rw [parts]
    simp only [run_append,prescaled_run,classified_run M ops p positive,reduced_run,registers_run,Option.bind_some]
  unfold TraceSafe
  rw [split,List.append_assoc]
  apply (traceSafeList_append _ _ _ _ _).mpr
  constructor
  · exact prefix_safe M ops p s bounds input
  · intro t ht
    have eq : t = addressed M p s := Option.some.inj (ht.symm.trans (addressed_run M ops p s))
    subst t
    apply (traceSafeList_append _ _ _ _ _).mpr
    constructor
    · exact arithmetic_safe M ops p _ bounds
    · intro t ht
      have eq : t = registers M ops p positive s := Option.some.inj (ht.symm.trans middle)
      subst t
      exact stores_safe M ops p positive s bounds output scale

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.SourceSafety
