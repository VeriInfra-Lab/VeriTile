import bench.repairs.tritonbench_g.l2_norm_triton1.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.L2NormTriton1
open VeriTile.Triton
/-- The readable source retains the scalar ABI register at its sole use site. -/
theorem epsilon_site (N B : Nat) :
    (source N B).surfaceBody[7]? = some
      (.assign .real [] "rstd" (.compute (.alg .fp32 (.div .real .nil (.const 1)
        (.sqrt (.add .real .nil (.ref .real [] "var")
          (.ref .real [] "eps"))))))) := by rfl

theorem full_body_length (N B : Nat) : (source N B).surfaceBody.length=11 := by rfl

end VeriTile.Bench.Repairs.L2NormTriton1

namespace VeriTile.Bench.Repairs.L2NormTriton1.Tile
open VeriTile.Triton FP.Structural
noncomputable section

/-- Every padded lane remains explicit. Sum order belongs to the selected
backend; no associative, zero-removal, or finite-input rule is imposed. -/
def loaded {α : Type} (M : Algebra α) (N B : Nat) (xs : Nat → α)
    (i : Fin B) : α :=
  M.cast (some .fp32) .real .real
    (if i.val<N then xs i.val else M.literal (some .fp32) .real 0.0)

def masked {α : Type} (M : Algebra α) (N B : Nat) (xs : Nat → α) (i : Fin B) : α :=
  if i.val<N then loaded M N B xs i else M.literal (some .fp32) .real 0.0

def variance {α : Type} (M : Algebra α) (N B : Nat) (xs : Nat → α) : α :=
  (M.reduceSum (some .fp32) (shape := [B]) ⟨0,by simp⟩ Bool.false
    (FP.Structural.bop (M.binary (some .fp32) .real .mul)
      (Broadcast.consSame Broadcast.nil)
      (fun i => masked M N B xs i.1) (fun i => masked M N B xs i.1))) PUnit.unit

def scalar {α : Type} (M : Algebra α) (N B : Nat) (eps : α) (xs : Nat → α)
    (i : Nat) : α :=
  let x := M.cast (some .fp32) .real .real (xs i)
  let root := M.unary (some .fp32) .sqrt
    (M.binary (some .fp32) .real .add (variance M N B xs) eps)
  let reciprocal := M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1) root
  M.binary (some .fp32) .real .mul x reciprocal

theorem variance_congr {α : Type} (M : Algebra α) (N B : Nat) (xs ys : Nat → α)
    (same : ∀ i,i<N → xs i=ys i) : variance M N B xs=variance M N B ys := by
  unfold variance
  congr 1
  funext i
  simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  by_cases hi : i.1.val<N
  · simp [masked,loaded,hi,same i.1.val hi]
  · simp [masked,hi]

theorem scalar_congr {α : Type} (M : Algebra α) (N B : Nat) (eps : α) (xs ys : Nat → α)
    (same : ∀ i,i<N → xs i=ys i) (i : Nat) (hi : i<N) :
    scalar M N B eps xs i=scalar M N B eps ys i := by
  unfold scalar
  rw [variance_congr M N B xs ys same,same i hi]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Tile
