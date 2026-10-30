import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.quant_transpose_kernel.Proofs.FinitePhysical
import bench.repairs.tritonbench_g.quant_transpose_kernel.Contract

namespace VeriTile.Bench.Repairs.QuantTransposeKernel
open QuantTransposeFinite

theorem indexChecks_iff (a : Input) : indexChecks a = Bool.true ↔ IndexDomain a := by
  simp only [indexChecks, decide_eq_true_eq]
  constructor
  · rintro ⟨hm,hn,hmu,hnu,hsm,hsn,hg,hw,hi,ho⟩
    exact ⟨hm,hn,hmu,hnu,hsm,hsn,hg,hw,hi,ho⟩
  · intro h
    exact ⟨h.rows,h.columns,h.rowsUpper,h.columnsUpper,h.strideM,h.strideN,h.grid,h.width,h.inputOffset,h.outputOffset⟩

theorem checkInputs_iff (p : Metadata) : checkInputs p = .ok () ↔ InputContract p := by
  have hc : InputContract p ↔ metadataChecks p = Bool.true ∧ IndexDomain p.input := by
    constructor
    · intro h; exact ⟨h.metadata,h.indices⟩
    · rintro ⟨hm,hi⟩; exact ⟨hm,hi⟩
  rw [hc, ← indexChecks_iff]
  simp [checkInputs, ite_eq_iff]

theorem checkArguments_iff (args : Arguments) : checkArguments args = .ok () ↔ Admitted args := by
  cases args <;> simp [checkArguments, Admitted, checkInputs_iff]

theorem admitted_metadata (p : Metadata) (h : InputContract p) :
    p.input.isCuda = Bool.true ∧ p.input.shape.length = 2 ∧
      (p.input.sam = 1 ∨ p.input.san = 1) := by
  have hm := h.metadata
  simp only [metadataChecks, decide_eq_true_eq] at hm
  exact ⟨hm.2.2.1,hm.2.2.2.2.1,hm.2.2.2.2.2.1⟩

/-- Ordinary physical facts without the input-alignment assumption. -/
structure OrdinaryPhysicalBinding (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (p : PhysicalLayout) : Prop where
  inputBase : p.input.base=a.storageBase
  maximumBase : p.maximum.base=alloc.maximumBase
  inverseBase : p.inverse.base=alloc.scaleBase
  outputBase : p.output.base=alloc.outputBase
  inputValid : p.input.Valid
  maximumValid : p.maximum.Valid
  inverseValid : p.inverse.Valid
  outputValid : p.output.Valid
  inputCapacity : p.input.capacity=a.storageLength*bytes .fp32
  inputPointer : p.inputPointer.toNat=p.input.base+a.storageOffset*bytes .fp32
  maximumCapacity : bytes .fp32≤p.maximum.capacity
  inverseCapacity : bytes (compute .fp32)≤p.inverse.capacity
  outputCapacity : a.numel≤p.output.capacity
  inputDevice : p.input.ownerDevice=a.deviceIndex
  maximumDevice : p.maximum.ownerDevice=a.deviceIndex
  inverseDevice : p.inverse.ownerDevice=a.deviceIndex
  outputDevice : p.output.ownerDevice=alloc.currentDevice
  inputHostRead : p.input.readable a.deviceIndex
  inputKernelRead : p.input.readable alloc.currentDevice
  maximumHostWrite : p.maximum.writable a.deviceIndex
  inverseHostWrite : p.inverse.writable a.deviceIndex
  inverseKernelRead : p.inverse.readable alloc.currentDevice
  outputKernelWrite : p.output.writable alloc.currentDevice
  maximumAlignment : p.maximum.base%bytes .fp32=0
  inverseAlignment : p.inverse.base%bytes (compute .fp32)=0
  inputMaximum : p.input.Disjoint p.maximum
  inputInverse : p.input.Disjoint p.inverse
  inputOutput : p.input.Disjoint p.output
  maximumInverse : p.maximum.Disjoint p.inverse
  maximumOutput : p.maximum.Disjoint p.output
  inverseOutput : p.inverse.Disjoint p.output
  outputInitialWords : ∀ i, i<a.numel → p.initialBytes (p.output.base+i)=some (alloc.initialOutput i)
  inputWords : InputBytes .fp32 a alloc m x p

def OrdinaryPhysicalBinding.admitted {p : Metadata} (h : InputContract p)
    {alloc : Allocations} {m : Memory} {x : String} {physical : PhysicalLayout}
    (ordinary : OrdinaryPhysicalBinding p.input alloc m x physical)
    (pointer : physical.inputPointer.toNat=p.dataPointer) :
    PhysicalBinding .fp32 p.input alloc m x physical := by
  have hm := h.metadata
  simp only [metadataChecks, decide_eq_true_eq] at hm
  exact {
    inputBase := ordinary.inputBase
    maximumBase := ordinary.maximumBase
    inverseBase := ordinary.inverseBase
    outputBase := ordinary.outputBase
    inputValid := ordinary.inputValid
    maximumValid := ordinary.maximumValid
    inverseValid := ordinary.inverseValid
    outputValid := ordinary.outputValid
    inputCapacity := ordinary.inputCapacity
    inputPointer := ordinary.inputPointer
    maximumCapacity := ordinary.maximumCapacity
    inverseCapacity := ordinary.inverseCapacity
    outputCapacity := ordinary.outputCapacity
    inputDevice := ordinary.inputDevice
    maximumDevice := ordinary.maximumDevice
    inverseDevice := ordinary.inverseDevice
    outputDevice := ordinary.outputDevice
    inputHostRead := ordinary.inputHostRead
    inputKernelRead := ordinary.inputKernelRead
    maximumHostWrite := ordinary.maximumHostWrite
    inverseHostWrite := ordinary.inverseHostWrite
    inverseKernelRead := ordinary.inverseKernelRead
    outputKernelWrite := ordinary.outputKernelWrite
    maximumAlignment := ordinary.maximumAlignment
    inverseAlignment := ordinary.inverseAlignment
    inputMaximum := ordinary.inputMaximum
    inputInverse := ordinary.inputInverse
    inputOutput := ordinary.inputOutput
    maximumInverse := ordinary.maximumInverse
    maximumOutput := ordinary.maximumOutput
    inverseOutput := ordinary.inverseOutput
    outputInitialWords := ordinary.outputInitialWords
    inputWords := ordinary.inputWords
    inputAlignment := by simpa [bytes,pointer] using hm.2.2.2.2.2.2 }

#axiomsClean checkArguments_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.QuantTransposeKernel
