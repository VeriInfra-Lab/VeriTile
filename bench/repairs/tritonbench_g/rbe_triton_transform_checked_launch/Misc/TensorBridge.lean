import bench.repairs.tritonbench_g.rbe_triton_transform.Correct
import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Contract
import bench.repairs.tritonbench_g.rbe_triton_transform.FPEquiv

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.TensorBridge
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs
open RbeTritonTransform Binding
noncomputable section

abbrev idealResult := RbeTritonTransform.Correct.idealResult

/-- Admission includes the actual nonempty launch bounds. The frozen kernel proof supplies every output and frame result. -/
theorem kernel_correct (r : Request) (x y : Tensor) (input : RbeTritonTransformCheckedLaunch.Input x r) (out : Output x y)
    (memory : Nat → ℝ) (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r)) :
    VeriTile.Spec.Real (Contract.checkInputs x.actual.metadata ⟨Bool.true,r.position⟩=.ok () ∧
      ∃ final, Runtime.run ideal r "X" "Y" qs (initial x y memory)=some final ∧
      (∀ k,k<y.size → y.read (commit y memory final) k=idealResult r x memory k) ∧
      commit y memory final=reference y memory (idealResult r x memory) ∧
      (∀ a,¬footprint y a → commit y memory final a=memory a) ∧
      (∀ (region : RegionName) (o : Nat),region≠"Y" ∨ r.size≤o → final.mem region o=(initial x y memory).mem region o)) := by
  have original := RbeTritonTransform.Correct.kernel_correct r x y input.original out memory qs all
  exact ⟨(Contract.checkInputs_iff _ _).mpr input.admitted,original.2⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.TensorBridge

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.TensorBridge
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs
open RbeTritonTransform Binding Ordered
noncomputable section

/-- Admission includes the actual nonempty launch bounds. The frozen kernel proof supplies every output and frame result. -/
theorem kernel_native_equivalence (p : Primitives) (r : Request) (x y : Tensor)
    (input : RbeTritonTransformCheckedLaunch.Input x r) (out : Output x y)
    (memory : Bytes.Memory) (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r)) :
    Contract.checkInputs x.actual.metadata ⟨Bool.true,r.position⟩=.ok () ∧
      (∀ q∈qs,∀ i im,
        ((Index.rowWord r q i).slt (BitVec.ofNat 64 r.rows) &&
          (Index.columnWord r q i+BitVec.ofNat 64 (if im then 1 else 0)).slt (BitVec.ofNat 64 r.columns))=
          decide (active r (Runtime.location q i im))) ∧
      (∀ q∈qs,∀ i,(Index.frequencyWord r q i).toInt=
        (evenColumn r (Runtime.location q i Bool.false)%128 : Nat)) ∧
      (∀ q∈qs,∀ i,(Index.tokenWord r q i).toInt=
        r.position+(row r (Runtime.location q i Bool.false):Int)) ∧
      (∀ q∈qs,∀ i im,active r (Runtime.location q i im) →
        let o := r.offset (batch (Runtime.location q i im)) (row r (Runtime.location q i im))
          (column r (Runtime.location q i im))
        (Index.pointer x.actual.dataPtr r q i im).toNat=x.view.address o ∧
        (Index.pointer y.actual.dataPtr r q i im).toNat=y.view.address o ∧
        x.view.address o%4=0 ∧ y.view.address o%4=0 ∧
        (∀ byte,byte<4 → x.view.address o+byte<2^64 ∧ y.view.address o+byte<2^64)) ∧
      ∃ final,Runtime.run (operations p) r "X" "Y" qs (Bytes.initial x y memory)=some final ∧
        Bytes.project y memory final=Bytes.execute p r y qs (Bytes.initial x y memory) memory ∧
        Bytes.project y memory final=Bytes.reference y memory (Bytes.nativeResult p r x memory) ∧
        (∀ k,k<y.size → Torch.FP32.logicalValue (Bytes.load (Bytes.project y memory final))
          y.actual y.laws k=Bytes.nativeResult p r x memory k) ∧
        (∀ a,¬footprint y a → Bytes.project y memory final a=memory a) ∧
        (∀ a,x.actual.storage.base≤a → a<x.actual.storage.base+4*x.actual.storage.elements →
          Bytes.project y memory final a=memory a) ∧
        (∀ (region : RegionName) (o : Nat),region≠"Y" ∨ r.size≤o → final.mem region o=(Bytes.initial x y memory).mem region o) := by
  have original := RbeTritonTransform.FPEquiv.kernel_native_equivalence p r x y
    input.original out input.supported memory qs all
  exact ⟨(Contract.checkInputs_iff _ _).mpr input.admitted,original.2⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.TensorBridge
