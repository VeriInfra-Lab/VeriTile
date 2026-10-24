import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Physical
import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Numerics

/-! Optional historical whole-call API results. Kernel acceptance is exposed
in the canonical Correct/FPEquiv entries; these results impose no new required
host-lifecycle proof or numerical-conformance experiment. -/

namespace VeriTile.Bench.Repairs.IndexSelectBwd.FPEquiv
open VeriTile.Triton FP.Structural Logical Physical
open VeriTile.Torch.FP32 (logicalValue)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The whole actual FP32 caller-Tensor result, plus the complete exterior
byte frame, untouched rows, both read-role backing frames, same output object,
None return and restored device context. No numeric agreement is assumed. -/
theorem whole_tensor_fp32 (rest : Algebra Word) (r : Request)
    (checked : checkInputs r=Bool.true) (objects : Objects r) (storage : StorageBinding r)
    (m : Memory) (indices : IndexBinding r m) (device : VeriTile.Torch.Device) :
    let guard := (Native.checkInputs_iff r).mp checked
    let final := Physical.run rest r guard m objects.destination.object device
    let result := Host.run rest r guard (words m r.destination) (words m r.incoming)
      objects.destination.object device
    (∀ q : Coord r.R r.C,
      logicalValue (read final) objects.destination objects.destinationLaws (Host.flat q)=
        Host.orderedTarget (Source.scalar rest) r
          (logicalValue (read m) objects.destination objects.destinationLaws)
          (logicalValue (read m) objects.incoming objects.incomingLaws) q) ∧
    (∀ p,p<r.destination.metadata.dataPtr.toNat ∨ r.destination.metadata.dataPtr.toNat+4*(r.R*r.C) ≤ p → final p=m p) ∧
    (∀ q : Coord r.R r.C,(∀ i : Fin r.N,r.indices i≠(q.1.val : Int)) →
      ∀ b : Fin 4,final (Native.address r.destination 4 (Host.flat q)+b.val)=
        m (Native.address r.destination 4 (Host.flat q)+b.val)) ∧
    (∀ p,(r.incoming.storageBase ≤ p ∧ p<r.incoming.storageBase+r.incoming.storageBytes) ∨
      (r.index.storageBase ≤ p ∧ p<r.index.storageBase+r.index.storageBytes) → final p=m p) ∧
    (∀ i,i<r.N → indexAt r final i=r.indices i) ∧
    result.returned=none ∧ result.object=objects.destination.object ∧ result.device=device := by
  dsimp only
  refine ⟨?_,?_,?_,?_,?_,rfl,rfl,rfl⟩
  · intro q
    exact logical_readback rest r _ objects m device q
  · intro p hp
    exact Physical.exterior_frame rest r _ m objects.destination.object device p hp
  · intro q miss b
    exact unselected_bytes rest r _ m objects.destination.object device q miss b
  · intro p hp
    exact input_frame rest r _ storage m objects.destination.object device p hp
  · intro i hi
    exact (index_preserved rest r _ storage m objects.destination.object device i hi).trans (indices i hi).symm

/-- Conditional IEEE result. Only local FP32 add conformance on actual finite
operand pairs remains deferred; the source/order/address/frame work is proved. -/
theorem whole_tensor_ieee_if_local (rest : Algebra Word) (actual : Word → Word → Word)
    (localAdd : Numerics.LocalAddAgreement actual) (r : Request) (checked : checkInputs r=Bool.true)
    (objects : Objects r) (m : Memory) (device : VeriTile.Torch.Device)
    (domain : Numerics.Domain r ((Native.checkInputs_iff r).mp checked) actual
      (words m r.destination) (words m r.incoming)) (q : Coord r.R r.C) :
    logicalValue (read (Physical.run (Host.backend rest actual) r ((Native.checkInputs_iff r).mp checked)
      m objects.destination.object device)) objects.destination objects.destinationLaws (Host.flat q)=
      Host.orderedTarget Numerics.referenceAdd r
        (logicalValue (read m) objects.destination objects.destinationLaws)
        (logicalValue (read m) objects.incoming objects.incomingLaws) q := by
  have guard := (Native.checkInputs_iff r).mp checked
  rw [←destination_direct r guard objects _ q,word_source]
  rw [Numerics.source_reference rest actual localAdd r guard _ _ domain objects.destination.object device q]
  exact target_logical _ r guard objects m q

#axiomsClean whole_tensor_fp32
#axiomsClean whole_tensor_ieee_if_local
#specNonCircular Host.orderedTarget avoiding [Physical.run,Host.run,Program.run,Source.execution]
end
end VeriTile.Bench.Repairs.IndexSelectBwd.FPEquiv
