import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Refinement

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural Whole
noncomputable section

/-- Compose the unchanged full host/grid/execution theorem with the standard
resolve_neg frame. Private materialization allocations are exactly enumerated;
every other cell outside the fresh returned output remains unchanged. -/
theorem resolved_host {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (before prepared : State α) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    ∃ final, host M D r prepared=.ok (r.C,final) ∧ Result Q r prepared final ∧
      (∀ region offset, ¬Footprint r region offset →
        ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧ AddressSafety r prepared tensor := by
  obtain ⟨reference,hr,hv,hf⟩ := Whole.host_correct M D Q law r prepared
    tensor.freshA tensor.freshB tensor.dimensions tensor.contiguousA
  obtain ⟨final,he,hm⟩ := host_refinement M D r prepared reference hr tensor.dimensions tensor.contiguousA
  refine ⟨final,he,?_,?_,address_safety r prepared tensor⟩
  · constructor
    · simpa only [hm] using hv
    · simpa only [hm] using hf
  · intro region offset houtput hcopy
    rw [hm]
    exact (hf region offset houtput).trans (resolve.frame region offset houtput hcopy)

theorem original_input_frame {α : Type} [Inhabited α]
    (r : Request) (before prepared final : State α) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (resolve : ResolveNegContract r before prepared tensor A B)
    (frame : ∀ region offset, ¬Footprint r region offset →
      ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
      final.mem region offset=before.mem region offset) :
    (∀ offset, final.mem resolve.originalA offset=before.mem resolve.originalA offset) ∧
    (∀ offset, final.mem resolve.originalB offset=before.mem resolve.originalB offset) := by
  constructor
  · intro offset
    apply frame
    · rintro ⟨he,_⟩
      exact resolve.outputFreshA he.symm
    · rintro (⟨ha,he,_⟩|⟨hb,he,_⟩)
      · exact resolve.copyAFreshA ha he.symm
      · exact resolve.copyBFreshA hb he.symm
  · intro offset
    apply frame
    · rintro ⟨he,_⟩
      exact resolve.outputFreshB he.symm
    · rintro (⟨ha,he,_⟩|⟨hb,he,_⟩)
      · exact resolve.copyAFreshB ha he.symm
      · exact resolve.copyBFreshB hb he.symm

/-- Input preservation composed with whole execution: no external frame or
successful-execution premise remains in this end-to-end statement. -/
theorem resolved_host_with_input_frame {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (before prepared : State α) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    ∃ final, host M D r prepared=.ok (r.C,final) ∧ Result Q r prepared final ∧
      (∀ offset, final.mem resolve.originalA offset=before.mem resolve.originalA offset) ∧
      (∀ offset, final.mem resolve.originalB offset=before.mem resolve.originalB offset) ∧
      (∀ region offset, ¬Footprint r region offset →
        ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧ AddressSafety r prepared tensor := by
  obtain ⟨final,he,hresult,hframe,hsafety⟩ := resolved_host M D Q law r before prepared tensor A B resolve
  obtain ⟨hA,hB⟩ := original_input_frame r before prepared final tensor A B resolve hframe
  exact ⟨final,he,hresult,hA,hB,hframe,hsafety⟩

#axiomsClean resolved_host_with_input_frame

theorem empty_output_identity {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request) (prepared : State α)
    (dimensions : r.inner=r.innerB) (contiguous : r.aContiguous=Bool.true)
    (empty : r.rows=0 ∨ r.columns=0) : host M D r prepared=.ok (r.C,prepared) := by
  rcases empty with h | h
  all_goals cases hh : r.config <;>
    simp [host,dimensions,contiguous,programs,tileRows,tileColumns,h,runPrograms,hh,Config.bm,Config.bn]

theorem dimension_error {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request) (prepared : State α)
    (h : r.inner≠r.innerB) : host M D r prepared=.error .incompatibleDimensions := by
  simp [host,h]

theorem contiguity_error {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (r : Request) (prepared : State α)
    (hdims : r.inner=r.innerB) (h : r.aContiguous=Bool.false) :
    host M D r prepared=.error .noncontiguousA := by
  simp [host,hdims,h]

#axiomsClean resolved_host
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
