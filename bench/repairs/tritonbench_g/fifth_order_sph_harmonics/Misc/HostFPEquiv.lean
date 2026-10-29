import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Initialized
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Equivalence
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Frames
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Lowering

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics
open VeriTile.Triton FP.Structural Packing Storage Host
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 64000

/-- The actual returned output object and all four bytes of each selected FP32
cell match the independent ordered binary32 reference. Only reached scalar
literal/add/sub/mul/sign operations are deferred; source, storage, masks, and
addresses are proved in the imported structural chain. -/
theorem fp_forward (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) (heap : Heap)
    (localLaws : ∀ r,r<q.coordinates.metadata.rows → ∀ c,
      FPProof.ForwardNumerics (Equivalence.exactLoads actual) rest softwareRest .ieeeValue
        (Equivalence.forwardInputs (forwardBuffers q allocated) heap r) c) :
    let result:=Host.forward (Equivalence.exactLoads actual) q admit allocated heap
    result.1.identity=(q.output allocated).identity ∧ result.2.1.savedCoordinates=q.coordinates ∧
    (∀ r,r<q.coordinates.metadata.rows → ∀ c : Fin 11,∀ k : Fin 4,
      result.2.2 ((q.output allocated).metadata.pointer+
        4*(r*(q.output allocated).metadata.columns+q.column.toNat+c.val)+k.val)=
      Packing.word (Software.forward .ieeeValue
        (Packing.value (Equivalence.forwardInputs (forwardBuffers q allocated) heap r 0))
        (Packing.value (Equivalence.forwardInputs (forwardBuffers q allocated) heap r 1))
        (Packing.value (Equivalence.forwardInputs (forwardBuffers q allocated) heap r 2)) c) k) ∧
    (∀ address,(∀ r,r<q.coordinates.metadata.rows → ∀ c : Fin 11,∀ k : Fin 4,
      address≠(q.output allocated).metadata.pointer+
        4*(r*(q.output allocated).metadata.columns+q.column.toNat+c.val)+k.val) → result.2.2 address=heap address) := by
  have h:=Equivalence.forward actual rest softwareRest .ieeeValue (parameters q admit allocated)
    (forwardBuffers q allocated) heap (parameters_fit q admit allocated allocation) localLaws
  exact ⟨rfl,rfl,h⟩

/-- The actual zeros_like gradient allocation is returned in the exact apply
argument arity. Every output byte matches the ordered corrected backward
reference on the actual coordinate/upstream addresses and initial zero words. -/
theorem fp_backward (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) (upstream gradient : TensorObject)
    (upValid : UpstreamContract q.coordinates.metadata upstream.metadata (q.output allocated).metadata.columns)
    (fresh : FreshGradient q.coordinates.metadata upstream.metadata gradient) (heap : Heap)
    (localLaws :
      let p:=backwardParameters (parameters q admit allocated) upstream.metadata
      let b:=backwardBuffers (save q allocated) upstream gradient
      let initial:=zeroGradient gradient.metadata.pointer p.rows heap
      ∀ r,r<p.rows → ∀ c,FPProof.BackwardNumerics (Equivalence.exactLoads actual) rest softwareRest .ieeeValue
        (Equivalence.backwardInputs p b initial r) c) :
    let p:=backwardParameters (parameters q admit allocated) upstream.metadata
    let b:=backwardBuffers (save q allocated) upstream gradient
    let result:=Host.backward (Equivalence.exactLoads actual) (parameters q admit allocated) (save q allocated) upstream gradient heap
    result.1.length=q.argumentCount ∧ result.1.head?=some (some gradient) ∧
    (∀ r,r<p.rows → ∀ c : Fin 3,∀ k : Fin 4,
      result.2 (gradient.metadata.pointer+4*(r*3+c.val)+k.val)=
      Initialized.reference p b heap r c k) ∧
    (∀ address,address<gradient.metadata.pointer ∨ gradient.metadata.pointer+12*p.rows≤address →
      result.2 address=heap address) := by
  let p:=backwardParameters (parameters q admit allocated) upstream.metadata
  let b:=backwardBuffers (save q allocated) upstream gradient
  let initial:=zeroGradient gradient.metadata.pointer p.rows heap
  have h:=Equivalence.backward actual rest softwareRest .ieeeValue p b initial
    (parameters_fit q admit allocated allocation) localLaws
  refine ⟨backward_arity q allocated gradient,backward_identity q allocated gradient,?_,?_⟩
  · intro r hr c k
    exact (h.1 r hr c k).trans (congrFun
      (Initialized.reference_before_allocation q admit allocated allocation upstream gradient upValid fresh heap r hr c) k)
  · intro address outside
    change address<gradient.metadata.pointer ∨ gradient.metadata.pointer+12*p.rows≤address at outside
    change (Physical.launch (Equivalence.exactLoads actual) .backward p b initial).bytes address=heap address
    rw [h.2 address]
    · simp only [initial,zeroGradient]
      rw [if_neg (by omega)]
    · intro r hr c k same
      have hc:=c.isLt
      have hk:=k.isLt
      change address=gradient.metadata.pointer+4*(r*3+c.val)+k.val at same
      omega
/-- Unselected output columns and both source tensors retain their input bits. -/
theorem forward_input_bytes (M : Algebra Word) (q : Request) (admit : Admitted q)
    (allocated : TensorObject) (allocation : Allocation q allocated) (heap : Heap)
    (r : Nat) (hr : r<q.coordinates.metadata.rows) (c : Fin 3) (k : Fin 4) :
    (Host.forward M q admit allocated heap).2.2 (q.coordinates.metadata.pointer+4*(r*3+c.val)+k.val)=
      heap (q.coordinates.metadata.pointer+4*(r*3+c.val)+k.val) :=
  Frames.forward_inputs M q admit allocated allocation heap r hr c k

#axiomsClean fp_forward
#axiomsClean fp_backward
#axiomsClean forward_input_bytes
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics
