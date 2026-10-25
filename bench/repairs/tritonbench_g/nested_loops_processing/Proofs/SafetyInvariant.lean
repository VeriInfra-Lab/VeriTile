import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SignedLoop
import VeriTile.Triton.Float.StructuralSafety

/-! Proof combinators that carry the actual initialized integer arguments
through successful instruction and loop successors. No execution or safety
property is assumed for the kernel. -/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedSafety
open VeriTile.Triton FP.Structural SignedExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

structure Atom (M : Algebra α) (bounds : RegionName → Nat) (count : Nat)
    (stmt : ComputeStmt) : Prop where
  safe : ∀ s, Context count s → Safety.stepSafe M bounds stmt s
  keeps : ∀ s t, Context count s → step M stmt s = some t → Context count t

structure Code (M : Algebra α) (bounds : RegionName → Nat) (count : Nat)
    (code : List ComputeStmt) : Prop where
  safe : ∀ s, Context count s → Safety.traceSafeList M bounds code s
  keeps : ∀ s t, Context count s → run M code s = some t → Context count t

theorem nil (M : Algebra α) (bounds : RegionName → Nat) (count : Nat) : Code M bounds count [] := by
  constructor
  · intro s _; exact Safety.traceSafeList_nil M bounds s
  · intro s t hs h
    have eq : s = t := by simpa [run] using h
    subst t
    exact hs

theorem cons {M : Algebra α} {bounds : RegionName → Nat} {count : Nat}
    {st : ComputeStmt} {rest : List ComputeStmt}
    (head : Atom M bounds count st) (tail : Code M bounds count rest) :
    Code M bounds count (st::rest) := by
  constructor
  · intro s hs
    apply Safety.traceSafeList_cons_intro (head.safe s hs)
    intro t h
    exact tail.safe t (head.keeps s t hs h)
  · intro s t hs h
    cases he : step M st s with
    | none => simp [run,he] at h
    | some u => exact tail.keeps u t (head.keeps s u hs he) (by simpa [run,he] using h)

theorem append {M : Algebra α} {bounds : RegionName → Nat} {count : Nat}
    {before after : List ComputeStmt}
    (head : Code M bounds count before) (tail : Code M bounds count after) :
    Code M bounds count (before++after) := by
  constructor
  · intro s hs
    rw [Safety.traceSafeList_append]
    exact ⟨head.safe s hs,fun t h => tail.safe t (head.keeps s t hs h)⟩
  · intro s t hs h
    rw [run_append] at h
    cases he : run M before s with
    | none => simp [he] at h
    | some u => exact tail.keeps u t (head.keeps s u hs he) (by simpa [he] using h)

theorem context_setReg (count : Nat) (s : State α) (name : RegName)
    (n : name ≠ "n_elements") (minor : name ≠ "stride_n")
    (d : TileDType) (sh : TileShape) (v : Values α d sh) (hs : Context count s) :
    Context count (s.setReg name d sh v) := by
  simpa [Context,n,minor,Ne.symm n,Ne.symm minor] using hs

private theorem fold_preserves {ι : Type} (l : List ι) (f : State α → ι → State α)
    (hf : ∀ s i, (f s i).regs=s.regs) (s : State α) : (l.foldl f s).regs=s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih => exact (ih (f s i)).trans (hf s i)

/-- Same structural store lemma used by runtime-register transport proofs. -/
private theorem store_regs (M : Algebra α) (mem : MemAccess d sh) (mask : MaskOpt d sh)
    (values : Values α d sh) (s t : State α) (h : FP.Structural.store M mem mask values s=some t) : t.regs=s.regs := by
  cases mem <;> cases mask
  all_goals
    simp only [FP.Structural.store,bind,pure,Option.bind,Option.map] at h
    repeat' first | (simp only [bind,pure,Option.bind,Option.map] at h) | split at h | contradiction
    all_goals cases h
    all_goals apply fold_preserves
    all_goals intro u i; first | rfl | (split <;> rfl)

def DataOnly : ComputeStmt → Prop
  | .assign _ _ name _ => name ≠ "n_elements" ∧ name ≠ "stride_n"
  | .store .. => True
  | .effectMarker name => name = "tl.debug_barrier"
  | _ => False

theorem data_keeps (M : Algebra α) (count : Nat) (st : ComputeStmt) (data : DataOnly st)
    (s t : State α) (hs : Context count s) (h : step M st s = some t) : Context count t := by
  cases st <;> simp only [DataOnly] at data
  case assign d sh name e =>
    cases he : evalExpr M e s with
    | none => simp [step,he] at h
    | some v =>
      have eq : s.setReg name d sh v = t := by simpa [step,he] using h
      subst t
      exact context_setReg count s name data.1 data.2 d sh v hs
  case store d sh mem e mask =>
    cases he : evalExpr M e s with
    | none => simp [step,he] at h
    | some v =>
      have eq : FP.Structural.store M mem mask v s = some t := by simpa [step,he] using h
      unfold Context
      rw [store_regs M mem mask v s t eq]
      exact hs
  case effectMarker name =>
    subst name
    simp only [step,Option.some.injEq] at h
    subst t
    exact hs

theorem of_flat (M : Algebra α) (bounds : RegionName → Nat) (count : Nat)
    (code : List ComputeStmt) (data : ∀ st ∈ code, DataOnly st)
    (safe : ∀ st ∈ code, ∀ s, Context count s → Safety.stepSafe M bounds st s) :
    Code M bounds count code := by
  induction code with
  | nil => exact nil M bounds count
  | cons st rest ih =>
    apply cons
    · exact ⟨safe st (by simp),data_keeps M count st (data st (by simp))⟩
    · exact ih (fun st h => data st (by simp [h])) (fun st h => safe st (by simp [h]))

theorem range_two (M : Algebra α) (bounds : RegionName → Nat) (count : Nat)
    (idx : RegName) (n : idx ≠ "n_elements") (minor : idx ≠ "stride_n")
    (body : List ComputeStmt) (checked : Code M bounds count body) :
    Atom M bounds count (.forRangeDyn idx (.constNat 0) (.constNat 2) (.constNat 1) body) := by
  constructor
  · intro s hs
    simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,Option.bind,
      true_and,and_true]
    rw [Safety.rangeSafe_unfold]
    simp only [Nat.one_ne_zero,Nat.reduceLT,↓reduceIte]
    have ctx0 := context_setReg count s idx n minor .nat [] (fun _ => 0) hs
    refine ⟨checked.safe _ ctx0,?_⟩
    cases h0 : run M body (s.setReg idx .nat [] (fun _ => 0)) with
    | none => trivial
    | some t =>
      have ctx1 := context_setReg count t idx n minor .nat [] (fun _ => 1)
        (checked.keeps _ _ ctx0 h0)
      dsimp only
      rw [Safety.rangeSafe_unfold]
      simp only [Nat.one_ne_zero,Nat.reduceLT,Nat.reduceAdd,↓reduceIte]
      refine ⟨checked.safe _ ctx1,?_⟩
      cases h1 : run M body (t.setReg idx .nat [] (fun _ => 1)) with
      | none => trivial
      | some u => simp [Safety.rangeSafe_unfold]
  · intro s t hs h
    have ran : FP.Structural.range M idx 0 2 1 body s = some t := by
      simpa [step,evalOp_unfold] using h
    rw [SignedExecution.range_two] at ran
    have ctx0 := context_setReg count s idx n minor .nat [] (fun _ => 0) hs
    cases h0 : run M body (s.setReg idx .nat [] (fun _ => 0)) with
    | none => simp [h0] at ran
    | some u =>
      have ctx1 := context_setReg count u idx n minor .nat [] (fun _ => 1)
        (checked.keeps _ _ ctx0 h0)
      exact checked.keeps _ _ ctx1 (by simpa [h0] using ran)
#axiomsClean data_keeps
#axiomsClean range_two
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedSafety
