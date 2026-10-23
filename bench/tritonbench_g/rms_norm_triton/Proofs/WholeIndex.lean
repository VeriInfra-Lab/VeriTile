import bench.tritonbench_g.rms_norm_triton.Proofs.WholePublic
import bench.tritonbench_g.rms_norm_triton.Proofs.KernelIndex

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Every flat logical element is covered, uniquely, by the actual row grid.
Zero-column and empty-grid cases have no logical elements. -/
theorem flat_coverage (domain : InputDomain input weight normalized preexisting)
    (i : Nat) (hi : i < input.numel) :
    ∃ row : Fin (rows input normalized), ∃ col : Fin (columns normalized),
      i = row.val*columns normalized+col.val ∧
      ∀ (row' : Fin (rows input normalized)) (col' : Fin (columns normalized)),
        i = row'.val*columns normalized+col'.val → row = row' ∧ col = col' := by
  have shape := shape_product domain
  have pos : 0 < columns normalized := by
    by_contra h
    have zero : columns normalized = 0 := by omega
    rw [zero,Nat.mul_zero] at shape
    omega
  have rb : i / columns normalized < rows input normalized :=
    (Nat.div_lt_iff_lt_mul pos).2 (by omega)
  have cb := Nat.mod_lt i pos
  refine ⟨⟨_,rb⟩,⟨_,cb⟩,?_,?_⟩
  · simpa only [Nat.mul_comm] using (Nat.div_add_mod i (columns normalized)).symm
  · intro row col h
    have unique := row_column_unique (N := columns normalized)
      (row := i/columns normalized) (other := row.val)
      (i := i%columns normalized) (j := col.val) cb col.isLt
      (by simpa only [Nat.mul_comm] using (Nat.div_add_mod i (columns normalized)).trans h)
    exact ⟨Fin.ext unique.1,Fin.ext unique.2⟩

/-- Successful setup admits exactly the physical allocation prepared by the
same runtime path. Failed preparation/setup has no kernel admission premise. -/
def RuntimeAdmission (rt : Runtime α input weight normalized entry preexisting) :=
  ∀ xc wc output schedule, rt.copyInput = .ok xc → rt.copyWeight xc = .ok wc →
    rt.allocate xc wc = .ok output → rt.setup ⟨xc,wc,output⟩ = .ok schedule →
    FiniteAdmission (Prepared.mk xc wc output) normalized

def FiniteHostSpec [Inhabited α] (formula : Nat → Nat → α)
    (rt : Runtime α input weight normalized entry preexisting) (admitted : RuntimeAdmission rt)
    (outcome : Outcome α) : Prop :=
  HostSpec formula rt outcome ∧
  ∀ xc wc output schedule (hx : rt.copyInput = .ok xc) (hw : rt.copyWeight xc = .ok wc)
    (ha : rt.allocate xc wc = .ok output) (hs : rt.setup ⟨xc,wc,output⟩ = .ok schedule),
    IndexCorrespondence (Prepared.mk xc wc output) normalized (admitted xc wc output schedule hx hw ha hs)

theorem finite_host_outcomes [Inhabited α] (mdl : Algebra α) (eps : ℝ)
    (rt : Runtime α input weight normalized entry preexisting)
    (domain : InputDomain input weight normalized preexisting) (admitted : RuntimeAdmission rt) :
    FiniteHostSpec (sourceFormula mdl input normalized eps (input.read entry) (weight.read entry)) rt admitted
      (executeHost mdl eps rt) := by
  exact ⟨host_outcomes mdl eps rt domain,fun xc wc output schedule hx hw ha hs =>
    index_correspondence _ _ (admitted xc wc output schedule hx hw ha hs)⟩

theorem finite_real_host_outcomes (eps : ℝ)
    (rt : Runtime ℝ input weight normalized entry preexisting)
    (domain : InputDomain input weight normalized preexisting) (admitted : RuntimeAdmission rt) :
    FiniteHostSpec (fun row i => realFormula (columns normalized) eps
      (fun j => input.read entry (row*columns normalized+j)) (weight.read entry) i) rt admitted
      (executeHost realAlgebra eps rt) := by
  exact ⟨real_host_outcomes eps rt domain,fun xc wc output schedule hx hw ha hs =>
    index_correspondence _ _ (admitted xc wc output schedule hx hw ha hs)⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
