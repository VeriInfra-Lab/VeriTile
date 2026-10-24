import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatSyntax

/-! Successful execution transport assembled from local child obligations.
These lemmas do not assume a whole-source commutation theorem. Sequence
composition quantifies only over actual successful predecessor steps; range
composition supplies the actual loop-index assignment at each iteration.
Memory bounds and unspecified-padding compatibility belong to the local
load/store obligations and are never replaced by Real erasure. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)

def StepTransport (st : ComputeStmt) (s : State α) : Prop :=
  ∀ t, FP.Structural.step M st s = some t →
    FP.Structural.step M (flattenComputeStmt A st)
      (flattenState A background flatUndef s) = some (flattenState A background flatUndef t)

def RunTransport (body : List ComputeStmt) (s : State α) : Prop :=
  ∀ t, FP.Structural.run M body s = some t →
    FP.Structural.run M (flattenComputeStmts A body)
      (flattenState A background flatUndef s) = some (flattenState A background flatUndef t)

theorem flatten_stmts_append (first second : List ComputeStmt) :
    flattenComputeStmts A (first ++ second) =
      flattenComputeStmts A first ++ flattenComputeStmts A second := by
  induction first with
  | nil => rfl
  | cons st rest ih =>
      simp only [List.cons_append, flattenComputeStmts, ih]

theorem run_nil_transport (s : State α) :
    RunTransport A M background flatUndef [] s := by
  intro t ht
  rw [FP.Structural.run] at ht
  have he : s = t := Option.some.inj ht
  subst t
  rw [flattenComputeStmts, FP.Structural.run]

/-- The tail obligation concerns only a state produced by this actual head.
No prepared register-state hypothesis is introduced by sequencing. -/
theorem run_cons_transport (st : ComputeStmt) (rest : List ComputeStmt) (s : State α)
    (hhead : StepTransport A M background flatUndef st s)
    (htail : ∀ u, FP.Structural.step M st s = some u →
      RunTransport A M background flatUndef rest u) :
    RunTransport A M background flatUndef (st :: rest) s := by
  intro t ht
  cases he : FP.Structural.step M st s with
  | none =>
      simp only [FP.Structural.run, he, bind, Option.bind_none] at ht
      cases ht
  | some u =>
      have hrest : FP.Structural.run M rest u = some t := by
        simpa only [FP.Structural.run, he, bind, Option.bind_some] using ht
      rw [flattenComputeStmts, FP.Structural.run]
      change (FP.Structural.step M (flattenComputeStmt A st)
        (flattenState A background flatUndef s)).bind
        (FP.Structural.run M (flattenComputeStmts A rest)) =
          some (flattenState A background flatUndef t)
      rw [hhead u he]
      exact htail u he t hrest

theorem run_append_transport (first second : List ComputeStmt) (s : State α)
    (hfirst : RunTransport A M background flatUndef first s)
    (hsecond : ∀ u, FP.Structural.run M first s = some u →
      RunTransport A M background flatUndef second u) :
    RunTransport A M background flatUndef (first ++ second) s := by
  intro t ht
  rw [FP.Structural.run_append] at ht
  cases he : FP.Structural.run M first s with
  | none =>
      simp only [he, Option.bind_none] at ht
      cases ht
  | some u =>
      have hrest : FP.Structural.run M second u = some t := by
        simpa only [he, Option.bind_some] using ht
      rw [flatten_stmts_append, FP.Structural.run_append, hfirst u he]
      exact hsecond u he t hrest

/-- A recursive collection of immediate-statement obligations, restricted
to the actual incoming state and actual successful predecessor steps. -/
def LocalTransport (A : FlatAlloc) (M : Algebra α)
    (background : RegionName → Nat → Cell α)
    (flatUndef : (d : TileDType) → RegionName → Nat → Value α d) :
    List ComputeStmt → State α → Prop
  | [], _ => True
  | st :: rest, s => StepTransport A M background flatUndef st s ∧
      ∀ u, FP.Structural.step M st s = some u →
        LocalTransport A M background flatUndef rest u

theorem run_of_local_transport (body : List ComputeStmt) (s : State α)
    (hlocal : LocalTransport A M background flatUndef body s) :
    RunTransport A M background flatUndef body s := by
  induction body generalizing s with
  | nil => exact run_nil_transport A M background flatUndef s
  | cons st rest ih =>
      exact run_cons_transport A M background flatUndef st rest s hlocal.1
        (fun u hu => ih u (hlocal.2 u hu))

theorem flatten_nat_index (s : State α) (idx : RegName) (j : Nat) :
    flattenState A background flatUndef (s.setReg idx .nat [] (fun _ => j)) =
      (flattenState A background flatUndef s).setReg idx .nat [] (fun _ => j) := by
  rw [flatten_setReg]
  rfl

/-- Fixed unit-step ranges, including empty ranges, from child body transport
and an invariant preserved by each actual successful body run. -/
theorem range_transport (idx : RegName) (body : List ComputeStmt)
    (ready : State α → Prop)
    (hbody : ∀ j s, ready s →
      RunTransport A M background flatUndef body (s.setReg idx .nat [] (fun _ => j)))
    (hpreserve : ∀ j s t, ready s →
      FP.Structural.run M body (s.setReg idx .nat [] (fun _ => j)) = some t → ready t)
    (n cur : Nat) (s : State α) (hs : ready s) :
    ∀ t, FP.Structural.range M idx cur (cur + n) 1 body s = some t →
      FP.Structural.range M idx cur (cur + n) 1 (flattenComputeStmts A body)
        (flattenState A background flatUndef s) = some (flattenState A background flatUndef t) := by
  induction n generalizing cur s with
  | zero =>
      intro t ht
      rw [FP.Structural.range] at ht
      have he : s = t := by
        simpa only [Nat.one_ne_zero, if_false,
          Nat.add_zero, lt_self_iff_false, Option.some.injEq] using ht
      subst t
      rw [FP.Structural.range]
      simp only [Nat.one_ne_zero, if_false, Nat.add_zero, lt_self_iff_false]
  | succ n ih =>
      intro t ht
      have hc : cur < cur + (n + 1) := by omega
      have hstop : cur + (n + 1) = (cur + 1) + n := by omega
      rw [FP.Structural.range] at ht
      simp only [Nat.one_ne_zero, if_false, hc, if_true] at ht
      cases he : FP.Structural.run M body (s.setReg idx .nat [] (fun _ => cur)) with
      | none =>
          simp only [he, bind, Option.bind_none] at ht
          cases ht
      | some u =>
          have hu : ready u := hpreserve cur s u hs he
          have hrest : FP.Structural.range M idx (cur + 1) ((cur + 1) + n) 1 body u = some t := by
            simpa only [he, bind, Option.bind_some, hstop] using ht
          have hflat : FP.Structural.run M (flattenComputeStmts A body)
              ((flattenState A background flatUndef s).setReg idx .nat [] (fun _ => cur)) =
                some (flattenState A background flatUndef u) := by
            simpa only [flatten_nat_index] using hbody cur s hs u he
          rw [FP.Structural.range]
          simp only [Nat.one_ne_zero, if_false, hc, if_true, hflat,
            bind, Option.bind_some]
          rw [hstop]
          exact ih (cur + 1) u hu t hrest

theorem step_const_if (b : Bool) (yes no : List ComputeStmt) (s : State α) :
    FP.Structural.step M (.ifThenElse (.constBool b) yes no) s =
      if b then FP.Structural.run M yes s else FP.Structural.run M no s := by
  conv_lhs =>
    rw [FP.Structural.step]
    simp only [FP.Structural.evalOp_unfold, bind, Option.bind, Option.pure_def]

theorem const_if_transport (b : Bool) (yes no : List ComputeStmt) (s : State α)
    (hyes : RunTransport A M background flatUndef yes s)
    (hno : RunTransport A M background flatUndef no s) :
    StepTransport A M background flatUndef (.ifThenElse (.constBool b) yes no) s := by
  intro t ht
  rw [step_const_if] at ht
  rw [flattenComputeStmt]
  simp only [FlatAlloc.flattenOp]
  change FP.Structural.step M (.ifThenElse (.constBool b)
    (flattenComputeStmts A yes) (flattenComputeStmts A no))
      (flattenState A background flatUndef s) = some (flattenState A background flatUndef t)
  rw [step_const_if]
  cases b with
  | «false» =>
      simp only [Bool.false_eq_true, if_false] at ht ⊢
      exact hno t ht
  | «true» =>
      simp only [if_true] at ht ⊢
      exact hyes t ht

theorem step_const_range (idx : RegName) (start stop : Nat)
    (body : List ComputeStmt) (s : State α) :
    FP.Structural.step M (.forRangeDyn idx (.constNat start) (.constNat stop) (.constNat 1) body) s =
      FP.Structural.range M idx start stop 1 body s := by
  conv_lhs =>
    rw [FP.Structural.step]
    simp only [FP.Structural.evalOp_unfold, bind, Option.bind, Option.pure_def]

theorem const_range_transport (idx : RegName) (body : List ComputeStmt)
    (ready : State α → Prop)
    (hbody : ∀ j s, ready s →
      RunTransport A M background flatUndef body (s.setReg idx .nat [] (fun _ => j)))
    (hpreserve : ∀ j s t, ready s →
      FP.Structural.run M body (s.setReg idx .nat [] (fun _ => j)) = some t → ready t)
    (n cur : Nat) (s : State α) (hs : ready s) :
    StepTransport A M background flatUndef
      (.forRangeDyn idx (.constNat cur) (.constNat (cur + n)) (.constNat 1) body) s := by
  intro t ht
  rw [step_const_range] at ht
  rw [flattenComputeStmt]
  simp only [FlatAlloc.flattenOp]
  change FP.Structural.step M
    (.forRangeDyn idx (.constNat cur) (.constNat (cur + n)) (.constNat 1)
      (flattenComputeStmts A body)) (flattenState A background flatUndef s) =
        some (flattenState A background flatUndef t)
  rw [step_const_range]
  exact range_transport A M background flatUndef idx body ready hbody hpreserve n cur s hs t ht

#axiomsClean flatten_stmts_append
#axiomsClean run_nil_transport
#axiomsClean run_cons_transport
#axiomsClean run_append_transport
#axiomsClean run_of_local_transport
#axiomsClean flatten_nat_index
#axiomsClean range_transport
#axiomsClean step_const_if
#axiomsClean const_if_transport
#axiomsClean step_const_range
#axiomsClean const_range_transport

end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
