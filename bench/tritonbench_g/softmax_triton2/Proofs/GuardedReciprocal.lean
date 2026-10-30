import bench.tritonbench_g.softmax_triton2.Kernels
import bench.tritonbench_g.softmax_triton1.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
open VeriTile.Triton FP.Structural FP.Guarded
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Exact bindings for the source's runtime classifiers. These describe the
finitef/FP32 order-comparison primitive implementation, not conditions on input words
and not a whole-kernel numerical assumption. -/
structure Ports {α : Type} (M : Algebra α) (D : Domain α) where
  finite : α → Bool
  atLeast : α → α → Bool
  finite_bind : M.isFinite .fp32 = some finite
  order_bind : M.compareLe (some .fp32) .real = some atLeast
  finite_sound : ∀ a, finite a = Bool.true → D .finite a
  nonzero_sound : ∀ b, atLeast (M.literal (some .fp32) .real 1) b = Bool.true → D .nonzero b

/-- No input-domain field: all states and all floating words are retained. -/
structure Program where
  kernel : ComputeKernel

def Equivalent (R : Spec.Assumptions FP.GuardedFragment) (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    Ports M D → ∀ s, ∃ a b,
      exec (FP.ProfiledRewrite.engine M) lhs.kernel s = some a ∧
      exec (FP.ProfiledRewrite.engine M) rhs.kernel s = some b ∧ a.mem = b.mem

instance : Spec.ProgramSyntax Program where
  Statement := FP.GuardedFragment
  theory := FP.Guarded.fp32Theory
  Signature := List RegionName × List RegionName
  signature p := match p.kernel with | .mk inputs outputs _ => (inputs,outputs)
  body p := [⟨[],p.kernel.surfaceBody,.real,none⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  sameContext := fun a b => a = b

/-- A true-count cannot equal the entire tile width unless every flag is true.
The admitted launch bound prevents uint32 count wraparound. -/
theorem count_all {n : Nat} (flags : Fin n → Bool) (hn : n < 2^32)
    (h : TileShape.boolCount32 flags = n) : ∀ i, flags i = Bool.true := by
  have bound : (∑ i, if flags i then (1 : Nat) else 0) ≤ n := by
    calc
      _ ≤ ∑ _ : Fin n, (1 : Nat) := Finset.sum_le_sum (fun i _ => by split <;> omega)
      _ = n := by simp
  have total : (∑ i, if flags i then (1 : Nat) else 0) = n := by
    unfold TileShape.boolCount32 at h
    rw [Nat.mod_eq_of_lt (lt_of_le_of_lt bound hn)] at h
    exact h
  intro i
  by_contra hi
  have flag : flags i = Bool.false := Bool.eq_false_iff.mpr hi
  have small : (∑ j, if flags j then (1 : Nat) else 0) < ∑ _ : Fin n, (1 : Nat) := by
    apply Finset.sum_lt_sum
    · intro j _
      split <;> omega
    · exact ⟨i,Finset.mem_univ _,by simp [flag]⟩
  simpa [total] using small


def originalExecution {α : Type} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0 < B) (s : State α) :
    {t : State α // exec (FP.ProfiledRewrite.engine M)
      (softmax_kernel out inp ib ob ir ic orow oc n B) s = some t} :=
  SoftmaxTriton1.Rewrite.originalExecution M out inp ib ob ir ic orow oc n B hB s

def enabled {α : Type} (M : Algebra α) (D : Domain α) (ports : Ports M D)
    (B : Nat) (num : Values α .real [B]) (den : α) : Prop :=
  TileShape.discreteBoolSum ⟨0,by simp⟩ Bool.false (fun i => ports.finite (num i))
      PUnit.unit = B ∧ ports.finite den = Bool.true ∧
    ports.atLeast (M.literal (some .fp32) .real 1) den = Bool.true

def candidateExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (ports : Ports M D) (out inp : RegionName)
    (ib ob ir ic orow oc n B : Nat) (hB : 0 < B) (s : State α) :
    {t : State α // exec (FP.ProfiledRewrite.engine M)
      (guarded_reciprocal_kernel out inp ib ob ir ic orow oc n B) s = some t} := by
  classical
  let num := SoftmaxTriton1.Rewrite.numeratorValue M inp ib ir ic n B s
  let den := SoftmaxTriton1.Rewrite.denominatorValue M inp ib ir ic n B s
  by_cases hc : TileShape.discreteBoolSum ⟨0,by simp⟩ Bool.false
      (fun i => ports.finite (num i)) PUnit.unit = B
  all_goals by_cases hf : ports.finite den = Bool.true
  all_goals by_cases hz : ports.atLeast (M.literal (some .fp32) .real 1) den = Bool.true
  all_goals
    refine ⟨?_,?_⟩
    swap
    dsimp [num,den,SoftmaxTriton1.Rewrite.numeratorValue,
      SoftmaxTriton1.Rewrite.denominatorValue] at hc hf hz
    simp [FP.Structural.exec,guarded_reciprocal_kernel,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,store,State.setReg,numeric,natLt,bop,
      TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeDType.eraseDType,
      ports.finite_bind,ports.order_bind,numericEq,numericLe,FP.ProfiledRewrite.engine,
      Algebra.withDefaultPrecision,resolvePrecision,hc,hf,hz]
    rfl

#axiomsClean count_all
#axiomsClean candidateExecution
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
