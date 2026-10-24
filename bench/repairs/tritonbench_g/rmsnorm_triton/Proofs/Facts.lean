import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.KernelPublic

namespace VeriTile.Bench.Repairs.RmsnormTriton.Facts
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 4000000

/-- Concrete source defect: the last increment overflows despite K<2^31. -/
theorem final_increment_overflow :
    (2147482625 : Nat)<2^31 ∧
    (BitVec.ofNat 32 (((2147482625+1023)/1024)*1024)).toInt= -2147483648 := by
  decide

/-- The independent mathematical target has a genuine nonzero denominator. -/
theorem denominator_positive (p : Parameters) (positive : 0<p.eps)
    (x : VeriTile.Torch.FP32.Tensor) (before : Memory ℝ) (b m : Nat) :
    0<Real.sqrt ((∑ j : Fin p.width,(tensorRead before x (flatIndex p b m j.val))^2)/
      (p.width : ℝ)+p.eps) :=
  mathematical_denominator_pos p.width p.eps positive (fun j => tensorRead before x (flatIndex p b m j))

/-- With no programs, source execution is identity independently of strides,
unused pointers, undefined words and every primitive numerical policy. -/
theorem empty_grid (Q : Numerics α) (red : Reduction) (p : Parameters)
    (u : Nat → α) (before : Memory α) (empty : p.batches=0 ∨ p.rows=0) :
    executeGrid Q red p u (programGrid p) before=before := by
  have grid : programGrid p=[] := by
    rcases empty with h|h
    · simp [programGrid,h]
    · simp only [programGrid,h,List.range_zero,List.map_nil]
      exact List.flatMap_eq_nil_iff.mpr (by intros; rfl)
  rw [grid]
  rfl

/-- Injectivity constrains active coordinates only. Zero stride is legal when
there is a single column; no positive-stride restriction is introduced. -/
theorem singleton_injective (base : Nat) :
    Function.Injective (fun k : Fin 1 => base+k.val*0) := by
  intro i j _
  exact Subsingleton.elim _ _

#axiomsClean final_increment_overflow
#axiomsClean denominator_positive
#axiomsClean empty_grid
#axiomsClean singleton_injective
end
end VeriTile.Bench.Repairs.RmsnormTriton.Facts
