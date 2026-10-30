import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.l2_norm_bwd.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.L2NormBwd.Tile
theorem full_body_length (s : Strides) (N B : Nat) :
    (source s N B).surfaceBody.length=14 := by rfl

#axiomsClean full_body_length
#guard_msgs (drop info) in
#auditModuleAxioms

open VeriTile.Triton FP.Structural
noncomputable section

def loaded {α : Type} (M : Algebra α) (N B : Nat) (xs : Nat → α) (i : Fin B) : α :=
  M.cast (some .fp32) .real .real
    (if i.val<N then xs i.val else M.literal (some .fp32) .real 0)

def masked {α : Type} (M : Algebra α) (N B : Nat) (xs : Nat → α) (i : Fin B) : α :=
  if i.val<N then loaded M N B xs i else M.literal (some .fp32) .real 0

def productSum {α : Type} (M : Algebra α) (N B : Nat) (xs ys : Nat → α) : α :=
  (M.reduceSum (some .fp32) (shape := [B]) ⟨0,by simp⟩ Bool.false
    (FP.Structural.bop (M.binary (some .fp32) .real .mul)
      (Broadcast.consSame Broadcast.nil)
      (fun i => masked M N B xs i.1) (fun i => masked M N B ys i.1))) PUnit.unit

/-- Source recurrence only: every padding lane, both reductions, epsilon uses,
reciprocals and operand order stay explicit. This is not the Real target. -/
def scalar {α : Type} (M : Algebra α) (N B : Nat) (eps : α) (xs dys : Nat → α) (i : Nat) : α :=
  let x := M.cast (some .fp32) .real .real (xs i)
  let dy := M.cast (some .fp32) .real .real (dys i)
  let q := M.binary (some .fp32) .real .add (productSum M N B xs xs) eps
  let rstd := M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1)
    (M.unary (some .fp32) .sqrt q)
  M.binary (some .fp32) .real .sub
    (M.binary (some .fp32) .real .mul dy rstd)
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .mul (productSum M N B dys xs)
          (M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1) q)) rstd) x)

theorem productSum_congr {α : Type} (M : Algebra α) (N B : Nat) (xs ys us vs : Nat → α)
    (hx : ∀ i,i<N → xs i=us i) (hy : ∀ i,i<N → ys i=vs i) :
    productSum M N B xs ys=productSum M N B us vs := by
  unfold productSum
  congr 1
  funext i
  simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  by_cases hi : i.1.val<N
  · simp [masked,loaded,hi,hx i.1.val hi,hy i.1.val hi]
  · simp [masked,hi]

theorem scalar_congr {α : Type} (M : Algebra α) (N B : Nat) (eps : α)
    (xs dys us vs : Nat → α) (hx : ∀ i,i<N → xs i=us i) (hy : ∀ i,i<N → dys i=vs i)
    (i : Nat) (hi : i<N) : scalar M N B eps xs dys i=scalar M N B eps us vs i := by
  unfold scalar
  rw [productSum_congr M N B xs xs us us hx hx,productSum_congr M N B dys xs vs us hy hx,hx i hi,hy i hi]

#axiomsClean scalar_congr
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Tile
