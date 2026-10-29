import bench.repairs.tritonbench_g.softmax_triton1.Proofs.FlatNaturalAddressView
import bench.repairs.tritonbench_g.softmax_triton1.IO
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.RealIO
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- Successful mathematical execution, access bounds, each active output,
and preservation of every exterior flat-memory cell. -/
theorem original_correct (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) :
    Spec.Real (softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0) := by
  apply (MaskedKernelIO₁.Implements.congr_memory _
    (NaturalAddressView.view Bool.false out inp ib ob n 1 n 1 n B) (by rfl)
    (fun A s => NaturalAddressView.flat_memory Bool.false out inp ib ob n 1 n 1 n B hB A (A.flattenState s)) _).mpr
  refine MaskedKernelIO₁.Implements.intro _ (NaturalAddressView.flattenOk Bool.false out inp ib ob n 1 n 1 n B) ?_ ?_
  · intro bounds s hin hout _
    exact NaturalAddressView.traceSafe Bool.false out inp ib ob n 1 n 1 n B hB bounds s
      (by simpa [softmaxIO] using hin) (by simpa [softmaxIO] using hout)
  · intro s xs input
    apply (exists_memory_congr
      (NaturalAddressView.regional_memory Bool.false out inp ib ob n 1 n 1 n B hB s)
      _ (by intro u v hm; simp only [BlockState.readMem, hm])).mp
    obtain ⟨t,he⟩ := row_exec out inp ib ob n 1 n 1 n B hB s
    refine ⟨t,he,?_,?_⟩
    · intro i hi
      have value := row_value out inp ib ob n 1 n 1 n B hB
        (by intro i j _ _ eq; apply Fin.ext; simpa using eq) s t he i hi
      have same : ∀ j : Fin B, j.val < n →
          s.readMem inp (ib+s.pid*n+j.val*1) = xs j := by
        simpa [softmaxIO] using input
      rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_congr n B _ xs same i,
        VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_eq_formula n B hB xs i hi] at value
      simpa [softmaxIO] using value
    · intro r a miss _
      apply row_frame out inp ib ob n 1 n 1 n B hB s t he r a
      intro i hi eq
      rcases miss with hr | ha
      · exact hr eq.1.symm
      · exact ha i hi (by simpa [softmaxIO] using eq.2.symm)


theorem reciprocal_correct (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) :
    Spec.Real (reciprocalIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0) := by
  apply (MaskedKernelIO₁.Implements.congr_memory _
    (NaturalAddressView.view Bool.true out inp ib ob n 1 n 1 n B) (by rfl)
    (fun A s => NaturalAddressView.flat_memory Bool.true out inp ib ob n 1 n 1 n B hB A (A.flattenState s)) _).mpr
  refine MaskedKernelIO₁.Implements.intro _ (NaturalAddressView.flattenOk Bool.true out inp ib ob n 1 n 1 n B) ?_ ?_
  · intro bounds s hin hout _
    exact NaturalAddressView.traceSafe Bool.true out inp ib ob n 1 n 1 n B hB bounds s
      (by simpa [reciprocalIO,softmaxIO] using hin)
      (by simpa [reciprocalIO,softmaxIO] using hout)
  · intro s xs input
    apply (exists_memory_congr
      (NaturalAddressView.regional_memory Bool.true out inp ib ob n 1 n 1 n B hB s)
      _ (by intro u v hm; simp only [BlockState.readMem, hm])).mp
    obtain ⟨t,he⟩ := row_exec out inp ib ob n 1 n 1 n B hB s
    refine ⟨t,(reciprocal_exec out inp ib ob n 1 n 1 n B hB s).trans he,?_,?_⟩
    · intro i hi
      have value := row_value out inp ib ob n 1 n 1 n B hB
        (by intro i j _ _ eq; apply Fin.ext; simpa using eq) s t he i hi
      have same : ∀ j : Fin B, j.val < n →
          s.readMem inp (ib+s.pid*n+j.val*1) = xs j := by
        simpa [reciprocalIO,softmaxIO] using input
      rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_congr n B _ xs same i,
        VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_eq_formula n B hB xs i hi] at value
      simpa [reciprocalIO,softmaxIO] using value
    · intro r a miss _
      apply row_frame out inp ib ob n 1 n 1 n B hB s t he r a
      intro i hi eq
      rcases miss with hr | ha
      · exact hr eq.1.symm
      · exact ha i hi (by simpa [reciprocalIO,softmaxIO] using eq.2.symm)

/-- The logical-column formula is the independent reference, with tile data
extended only outside its finite capacity. Coverage makes that extension unused. -/
theorem logical_formula_eq_reference (n B : Nat) (cover : n ≤ B)
    (xs : Fin B → ℝ) (i : Fin B) :
    Real.exp (xs i) / (∑ j : Fin n, Real.exp (xs (Fin.castLE cover j))) =
      reference n (fun k => if hk : k < B then xs ⟨k, hk⟩ else 0) i.val := by
  unfold reference
  rw [← Fin.sum_univ_eq_sum_range]
  simp only [dif_pos i.isLt]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  simp only [dif_pos (lt_of_lt_of_le j.isLt cover)]
  rfl

/-- Preserve the original window theorem and expose its full logical-row
specialization under the launch coverage condition. -/
theorem original_logical_correct (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) (cover : n ≤ B) :
    Spec.Real (softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (xs (Fin.castLE cover j))) := by
  simpa only [RealModel.masked_sum_eq_logical n B cover] using
    original_correct out inp ib ob n B hB


/-- The original masked-window statement remains available without coverage. -/
theorem softmax_window_correctness (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) :
    Spec.Real (softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0) :=
  original_correct out inp ib ob n B hB


#specNonCircular reference avoiding [softmax_kernel,reciprocal_kernel]
#axiomsClean original_correct
#axiomsClean reciprocal_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.RealIO
