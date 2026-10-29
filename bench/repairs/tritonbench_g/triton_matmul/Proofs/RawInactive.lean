import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawWritten

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

theorem outside_group_masks {α : Type} (p : Parameters) (s : State α)
    (hm : signedCeil p.M (BM p) = (blocksM p : Int))
    (hn : signedCeil p.N (BN p) = (blocksN p : Int))
    (hcover : p.M ≤ blocksM p * BM p) (hf : blocksM p ≤ firstM p s) :
    (∀ i, maskM p s i = Bool.false) ∧ (∀ i : CIndex p, ¬FPValue.active p s i) := by
  have hfirst : (control p s).first = (firstM p s : Int) := by
    simp only [control,hn]
    simp [firstM,perGroup]
  have hpm : (firstM p s : Int) ≤ (control p s).pidM := by
    have hmod := Int.tmod_nonneg (control p s).size (Int.natCast_nonneg (s.pids 0))
    change (firstM p s : Int) ≤ (control p s).first + (s.pids 0 : Int).tmod (control p s).size
    rw [hfirst]
    omega
  have hb : (blocksM p : Int) ≤ (control p s).pidM := by exact le_trans (by exact_mod_cast hf) hpm
  have hc : (p.M : Int) ≤ (blocksM p : Int) * (BM p : Int) := by exact_mod_cast hcover
  have hs : (p.M : Int) ≤ (control p s).startM := by
    change (p.M : Int) ≤ (control p s).pidM * (BM p : Int)
    exact le_trans hc (mul_le_mul_of_nonneg_right hb (Int.natCast_nonneg _))
  constructor
  · intro i
    have h : ¬(i.val : Int) < (p.M : Int) - (control p s).startM := by omega
    simp [maskM,h]
  · intro i hi
    have hrow : blocksM p * BM p ≤ row p s i.1 := by
      have hpid : blocksM p ≤ pidM p s := by unfold pidM; omega
      exact le_trans (Nat.mul_le_mul_right (BM p) hpid) (Nat.le_add_right _ _)
    exact (Nat.not_lt_of_ge (le_trans hcover hrow)) hi.1

/-- An id past the last row group writes nothing in either presentation.
This preserves generic host success helpers without restricting their ids. -/
theorem written_outside {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α)
    (hm : signedCeil p.M (BM p) = (blocksM p : Int))
    (hn : signedCeil p.N (BN p) = (blocksN p : Int))
    (hcover : p.M ≤ blocksM p * BM p) (hf : blocksM p ≤ firstM p s) :
    RawSource.written M D p st s = FPValue.written M D p st s := by
  obtain ⟨hr,hm⟩ := outside_group_masks p s hm hn hcover hf
  simp [RawSource.written,FPValue.written,RawSource.active,hr,hm]

#axiomsClean outside_group_masks
#axiomsClean written_outside
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
