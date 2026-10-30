import bench.tritonbench_g.l2_norm_triton2.Contract
import bench.tritonbench_g.l2_norm_triton1.Proofs.Contract

namespace VeriTile.Bench.Repairs.L2NormTriton2
open VeriTile.Bench.Repairs.SwigluTriton.Memory
abbrev TensorBinding := L2NormTriton1.TensorBinding

theorem contract_iff (x dy : TensorMetadata) : InputContract x dy ↔
    (x.layout=.strided ∧ dy.layout=.strided) ∧
    (x.dtype=.floating .fp32 ∧ dy.dtype=.floating .fp32) ∧
    (x.device.kind="cuda" ∧ dy.device.kind="cuda") ∧ x.device=dy.device ∧
    dy.shape≠[] ∧ (0<dy.shape.getLastD 0 ∧ dy.shape.getLastD 0≤16384) ∧ x.numel=dy.numel ∧
    (x.isContiguous=Bool.true ∧ dy.isContiguous=Bool.true) ∧
    (x.isNeg=Bool.false ∧ dy.isNeg=Bool.false) ∧
    (0<x.numel → x.dataPtr.toNat%16=0 ∧ dy.dataPtr.toNat%16=0) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.sameDevice,h.rank,h.width,h.sameCount,h.contiguous,h.noNeg,h.alignment⟩
  · rintro ⟨hl,hd,hc,hv,hr,hw,he,ht,hn,ha⟩; exact ⟨hl,hd,hc,hv,hr,hw,he,ht,hn,ha⟩

theorem checkTensorInputs_iff (x dy : TensorMetadata) :
    checkTensorInputs x dy=.ok () ↔ InputContract x dy := by
  rw [contract_iff]
  simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff]

theorem checkInputs_iff (x dy : ArgumentMetadata) :
    checkInputs x dy=.ok () ↔ AdmittedArguments x dy := by
  cases x <;> cases dy <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

/-- Direct contiguous storage, independent of how backward groups the rows. -/
structure DirectView (input : View) : Prop where
  negative : input.negative=Bool.false
  contiguous : ∀ i,i < input.numel → input.key i=⟨input.storage.owner,(input.key 0).address+4*i⟩
  aligned : 0 < input.numel → (input.key 0).address%16=0

theorem direct_view (metadata : TensorMetadata) (input : View) (binding : TensorBinding metadata input)
    (contiguous : metadata.isContiguous=Bool.true) (negative : metadata.isNeg=Bool.false)
    (alignment : 0<metadata.numel → metadata.dataPtr.toNat%16=0) : DirectView input := by
  refine ⟨binding.negative ▸ negative,binding.contiguous contiguous,?_⟩
  intro nonempty
  have count : metadata.numel=input.numel := congrArg List.prod binding.shape
  have aligned:=alignment (by rw [count];exact nonempty)
  rw [binding.pointer nonempty,BitVec.toNat_ofNat] at aligned
  have bound:=binding.backed 0 nonempty
  have fits:=input.storage.representable
  have small : (input.key 0).address<2^64 := by unfold Allocation.contains at bound;omega
  simpa only [Nat.mod_eq_of_lt small] using aligned

#axiomsClean checkInputs_iff
#axiomsClean direct_view
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.L2NormTriton2
