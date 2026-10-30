import bench.tritonbench_g.l2_norm_triton1.Contract
import bench.tritonbench_g.swiglu_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.L2NormTriton1
open VeriTile.Bench.Repairs.SwigluTriton.Memory

theorem contract_iff (x : TensorMetadata) : InputContract x ↔
    x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda" ∧
    x.shape≠[] ∧ (0<x.shape.getLastD 0 ∧ x.shape.getLastD 0≤16384) ∧
    x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧
    (0<x.numel → x.dataPtr.toNat%16=0) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.rank,h.width,h.contiguous,h.noNeg,h.alignment⟩
  · rintro ⟨hl,hd,hc,hr,hw,ht,hn,ha⟩; exact ⟨hl,hd,hc,hr,hw,ht,hn,ha⟩

theorem checkTensorInputs_iff (x : TensorMetadata) : checkTensorInputs x=.ok () ↔ InputContract x := by
  rw [contract_iff]
  simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff]

theorem checkInputs_iff (x : ArgumentMetadata) : checkInputs x=.ok () ↔ AdmittedArguments x := by
  cases x <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

/-- Ordinary Tensor observations and address law, with no kernel result or
normalization/copy postcondition. The metadata is that checked by Python. -/
structure TensorBinding (metadata : TensorMetadata) (input : View) : Prop where
  shape : metadata.shape=input.shape
  negative : metadata.isNeg=input.negative
  device : metadata.device.index=input.storage.owner
  backed : input.backed
  contiguous : metadata.isContiguous=Bool.true → ∀ i,i< input.numel →
    input.key i=⟨input.storage.owner,(input.key 0).address+4*i⟩
  pointer : 0< input.numel → metadata.dataPtr=BitVec.ofNat 64 (input.key 0).address

structure DirectInput (input : View) : Prop where
  positive : 0< input.shape.getLastD 0
  width : input.shape.getLastD 0≤16384
  negative : input.negative=Bool.false
  contiguous : ∀ i,i< input.numel → input.key i=⟨input.storage.owner,(input.key 0).address+4*i⟩
  aligned : 0< input.numel → (input.key 0).address%16=0

theorem direct_input (metadata : TensorMetadata) (input : View)
    (h : InputContract metadata) (binding : TensorBinding metadata input) : DirectInput input := by
  refine ⟨by rw [←binding.shape];exact h.width.1,by rw [←binding.shape];exact h.width.2,
    binding.negative ▸ h.noNeg,binding.contiguous h.contiguous,?_⟩
  intro nonempty
  have count : metadata.numel=input.numel := congrArg List.prod binding.shape
  have aligned:=h.alignment (by rw [count];exact nonempty)
  rw [binding.pointer nonempty,BitVec.toNat_ofNat] at aligned
  have bound:=binding.backed 0 nonempty
  have fits:=input.storage.representable
  have small : (input.key 0).address<2^64 := by
    unfold Allocation.contains at bound
    omega
  simpa only [Nat.mod_eq_of_lt small] using aligned

#axiomsClean checkInputs_iff
#axiomsClean direct_input
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.L2NormTriton1
