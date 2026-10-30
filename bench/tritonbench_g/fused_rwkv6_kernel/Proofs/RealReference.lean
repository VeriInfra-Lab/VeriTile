import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Reference

namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
/-- Dense RWKV6 matrix recurrence, independent of tiling and store execution. -/
def realState (c : Shape) (x : Input ℝ) (z e f : Nat) : Nat → ℝ
  | 0 => if c.initial then x.h0 z e f else 0
  | n+1 => realState c x z e f n * Real.exp (x.w z (visit c n) e) +
      x.k z (visit c n) e * x.v z (visit c n) f

def realOutput (c : Shape) (x : Input ℝ) (scale : ℝ) (z n f : Nat) : ℝ :=
  ∑ e ∈ Finset.range c.K,
    (realState c x z e f n + x.k z (visit c n) e * x.v z (visit c n) f * x.u z e) *
      (x.q z (visit c n) e * scale)

def realOps : Ops ℝ := ⟨0,id,id,(·+·),(·*·),Real.exp⟩

theorem scalar_real_state (c : Shape) (x : Input ℝ) (z e f n : Nat)
    (he : e<c.K) (hf : f<c.V) :
    scalarState realOps c x z e f n=realState c x z e f n := by
  induction n with
  | zero => simp [scalarState,scalarSeed,realState,realOps,he,hf]
  | succ n ih =>
    simp only [scalarState,realState]
    rw [ih]
    simp [scalarOuter,realOps,he,hf]

theorem scalar_real_term (c : Shape) (x : Input ℝ) (scale : ℝ) (z n e f : Nat) (hf : f<c.V) :
    scalarTerm realOps c x scale z n e f =
      if e<c.K then (realState c x z e f n + x.k z (visit c n) e *
        x.v z (visit c n) f * x.u z e) * (x.q z (visit c n) e * scale) else 0 := by
  by_cases he : e<c.K
  · unfold scalarTerm
    rw [scalar_real_state c x z e f n he hf]
    simp [scalarOuter,realOps,he,hf]
  · simp [scalarTerm,scalarOuter,realOps,he]

end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
