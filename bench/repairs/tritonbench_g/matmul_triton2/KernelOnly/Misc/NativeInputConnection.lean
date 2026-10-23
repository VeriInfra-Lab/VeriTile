import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Misc.NativeReference

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.KernelInputs
open VeriTile.Triton FP.Structural Revision001
noncomputable section
variable {α : Type} [Inhabited α]

theorem logical_reference (P : Core.Primitives α) (p : Core.Parameters)
    (initial : State α) (a b : Nat → Nat → α)
    (ha : ∀ r k, r<p.rows → k<p.innerSize →
      (initial.mem p.input (r*p.inputRow+k*p.inputInner)).read .real=a r k)
    (hb : ∀ k c, k<p.innerSize → c<p.columns →
      (initial.mem p.weight (k*p.weightInner+c*p.weightColumn)).read .real=b k c)
    (r c : Nat) (hr : r<p.rows) (hc : c<p.columns) :
    Core.logical P p initial r c=
      Reference.accumulation P p.innerSize p.blockInner a b r c (Core.steps p) := by
  have aux : ∀ q, Core.logicalAccum P p initial r c q=
      Reference.accumulation P p.innerSize p.blockInner a b r c q := by
    intro q
    induction q with
    | zero => rfl
    | succ q ih =>
      simp only [Core.logicalAccum,Reference.accumulation,ih]
      congr 1
      · funext k
        unfold Core.logicalA Reference.left
        have tail : k.val<Core.inner p-q*p.blockInner ↔ q*p.blockInner+k.val<p.innerSize := by
          dsimp only [Core.inner]; omega
        simp only [tail]
        split
        · rename_i hk
          exact ha r _ hr hk
        · rfl
      · funext k
        unfold Core.logicalB Reference.right
        have tail : k.val<Core.inner p-q*p.blockInner ↔ q*p.blockInner+k.val<p.innerSize := by
          dsimp only [Core.inner]; omega
        simp only [tail]
        split
        · rename_i hk
          exact hb _ c hk hc
        · rfl
  exact aux _

#axiomsClean logical_reference
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.KernelInputs
