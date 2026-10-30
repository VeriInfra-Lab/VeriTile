import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelPhysical
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelIndex

namespace VeriTile.Bench.Repairs.RbeTritonTransform.FPEquiv
open VeriTile Triton FP.Structural Binding Ordered
noncomputable section

/-- Successful full selected-source execution equals the independent ordered
native rotation at every output byte. Local scalar primitives and the observed
FMA profile are the model boundary; no unconditional FP=Real claim is made. -/
specification kernel_native_equivalence (p : Primitives) (r : Request) (x y : Tensor)
    (input : Input x r) (out : Output x y) (launch : Index.SupportedLaunch r)
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
  refine ⟨(Contract.checkInputs_iff _ _).mpr input.admitted,?_,?_,?_,?_,?_⟩
  · intro q hq i im
    exact Index.mask x r input launch q (all.mem_iff.mp hq) i im
  · intro q hq i
    exact Index.frequency x r input launch q (all.mem_iff.mp hq) i
  · intro q hq i
    exact Index.token x r input launch q (all.mem_iff.mp hq) i
  · intro q hq i im active
    exact Index.addresses x y r input out launch q (all.mem_iff.mp hq) i im active
  · obtain ⟨final,ran,values,frame⟩ := Runtime.whole (operations p) r "X" "Y" qs all (Bytes.initial x y memory) (by decide)
    have flat : ∀ k,k<y.size → ((final.mem "Y" k).read .fp32).word=Bytes.nativeResult p r x memory k := by
      intro k hk
      have coords := coordinates r k (by rw [←input.size,←out.size]; exact hk)
      have value := congrArg Tagged.word (values (batchIndex r k) (rowIndex r k) (columnIndex r k)
        coords.1 coords.2.1 coords.2.2.1)
      rw [coords.2.2.2,Bytes.logical_value p r x y input memory _ _ _ coords.1 coords.2.1 coords.2.2.1] at value
      exact value
    have whole := Bytes.project_reference y memory final _ flat
    refine ⟨final,ran,Bytes.run_bytes p r x y input out memory qs all final ran,whole,?_,?_,?_,frame⟩
    · intro k hk
      rw [←y.read_logical,whole,y.read_view _ k hk,Bytes.reference_load y memory _ k hk]
    · intro a outside
      rw [whole]
      exact Bytes.reference_frame y memory _ a outside
    · intro a lo hi
      rw [whole]
      exact Bytes.input_backing_frame x y out memory _ a ⟨lo,hi⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.FPEquiv
