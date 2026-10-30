import bench.repairs.tritonbench_g.triton_matmul.Proofs.Whole
import bench.repairs.tritonbench_g.triton_matmul.Proofs.Addresses
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural
noncomputable section

/-- Execution is constructed independently of all four numerical ports. -/
theorem grid_executes {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (ids : List Nat) (s : State α) :
    ∃ final, runPrograms M D r ids s=some final := by
  induction ids generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons pid ids ih =>
    obtain ⟨next,he,_⟩ := Source.execution_memory M D r.parameters r.storage (instanceState r pid s)
    obtain ⟨final,hf⟩ := ih next
    exact ⟨final,by simpa only [runPrograms,kernel,he,Option.bind_some] using hf⟩

theorem host_executes {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (s : State α) (hd : r.inner=r.innerB) (ht : r.storage=r.storageB) :
    ∃ final, host M D r s=.ok (r.C,final) := by
  obtain ⟨final,he⟩ := grid_executes M D r (List.range (programs r)) s
  exact ⟨final,by simp [host,hd,ht,he]⟩

theorem resolved_host {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α)
    (r : Request) (law : PrimitiveLaws M D Q r.storage)
    (before prepared : State α) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    ∃ final, host M D r prepared=.ok (r.C,final) ∧ Result Q r prepared final ∧
      (∀ region offset, ¬Footprint r region offset →
        ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧
      (∀ offset, final.mem resolve.originalA offset=before.mem resolve.originalA offset) ∧
      (∀ offset, final.mem resolve.originalB offset=before.mem resolve.originalB offset) ∧
      AddressSafety r prepared tensor := by
  obtain ⟨final,he,hvalue,hframe⟩ := host_correct M D Q r law prepared
    tensor.freshA tensor.freshB tensor.dimensions tensor.storage
  have frame (region offset) (ho : ¬Footprint r region offset)
      (hc : ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset) :
      final.mem region offset=before.mem region offset :=
    (hframe region offset ho).trans (resolve.frame region offset ho hc)
  refine ⟨final,he,⟨hvalue,hframe⟩,frame,?_,?_,address_safety r prepared tensor⟩
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

theorem candidate_store_offset {α : Type} (r : Request) (s : State α) (i : FPValue.CIndex r.parameters) :
    Source.cSafeOffset r.parameters s i=
      safeCOffset r (FPValue.row r.parameters s i.1) (FPValue.col r.parameters s i.2.1) := by
  simp [Source.cSafeOffset,safeCOffset,safeCoordinate,Request.parameters,SourceScope.parameters]

theorem candidate_store_unique {α : Type} (r : Request) (s : State α)
    (p q : Nat) (hp : p<programs r) (hq : q<programs r) (i j : FPValue.CIndex r.parameters)
    (hi : FPValue.active r.parameters (instanceState r p s) i)
    (hj : FPValue.active r.parameters (instanceState r q s) j)
    (he : Source.cSafeOffset r.parameters (instanceState r p s) i=
      Source.cSafeOffset r.parameters (instanceState r q s) j) : p=q ∧ i=j := by
  have ha : Source.cSafeOffset r.parameters (instanceState r p s) i=
      FPValue.cOffset r.parameters (instanceState r p s) i := by
    simp only [Source.cSafeOffset,FPValue.cOffset,hi.1,hi.2,if_true]
  have hb : Source.cSafeOffset r.parameters (instanceState r q s) j=
      FPValue.cOffset r.parameters (instanceState r q s) j := by
    simp only [Source.cSafeOffset,FPValue.cOffset,hj.1,hj.2,if_true]
  rw [ha,hb] at he
  exact source_store_unique r s p q hp hq i j hi hj he

theorem empty_output_identity {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (s : State α) (hd : r.inner=r.innerB) (ht : r.storage=r.storageB)
    (empty : r.rows=0 ∨ r.columns=0) : host M D r s=.ok (r.C,s) := by
  rcases empty with h | h <;> simp [host,hd,ht,programs,tileRows,tileColumns,h,runPrograms]
theorem dimension_error {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (s : State α) (hd : r.inner≠r.innerB) :
    host M D r s=.error .incompatibleDimensions := by simp [host,hd]
theorem dtype_error {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (s : State α) (hd : r.inner=r.innerB) (ht : r.storage≠r.storageB) :
    host M D r s=.error .incompatibleDtypes := by simp [host,hd,ht]

#axiomsClean grid_executes
#axiomsClean host_executes
#axiomsClean resolved_host
#axiomsClean candidate_store_unique
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
