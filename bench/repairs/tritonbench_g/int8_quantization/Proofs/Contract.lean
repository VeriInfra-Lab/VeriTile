import Mathlib
import bench.repairs.tritonbench_g.int8_quantization.Contract
import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceNative
namespace VeriTile.Bench.Repairs.Int8Quantization.Direct

theorem contract_iff (x : TensorMetadata) (block : Nat) : InputContract x block ↔
    (x.shape.length=3 ∨ x.shape.length=4) ∧ x.layout=.strided ∧ x.dtype=.floating .fp32 ∧
    x.device.kind="cuda" ∧ x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧
    0< block ∧ block*channels x≤2^20 ∧ x.numel<2^61 ∧ scaleCount x block<2^61 ∧
    (0< x.numel → x.dataPtr.toNat%4=0) := by
  constructor
  · intro h;exact ⟨h.rank,h.layout,h.dtype,h.cuda,h.contiguous,h.noNeg,h.blockPositive,h.tile,h.inputExtent,h.scaleExtent,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k⟩;exact ⟨a,b,c,d,e,f,g,h,i,j,k⟩

theorem checkTensorInputs_iff (x : TensorMetadata) (block : Nat) :
    checkTensorInputs x block=.ok () ↔ InputContract x block := by
  rw [contract_iff]
  simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff,and_assoc,Nat.pos_iff_ne_zero]
  tauto

theorem checkInputs_iff (x : ArgumentMetadata) (block : BlockArgument) :
    checkInputs x block=.ok () ↔ AdmittedArguments x block := by
  cases x <;> cases block <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff,ite_eq_iff]

theorem shape_eq (x : TensorMetadata) (block : Nat) (h : InputContract x block) :
    x.shape=batchShape x++[length x,channels x] := by
  rcases x with ⟨dtype,layout,device,shape,contig,neg,ptr⟩
  rcases shape with _|⟨a,shape⟩
  · have := h.rank;simp at this
  rcases shape with _|⟨b,shape⟩
  · have := h.rank;simp at this
  rcases shape with _|⟨c,shape⟩
  · have := h.rank;simp at this
  cases shape with
  | nil => rfl
  | cons d tail =>
    cases tail with
    | nil => rfl
    | cons e tail => have := h.rank;simp at this

theorem numel_eq (x : TensorMetadata) (block : Nat) (h : InputContract x block) :
    x.numel=batches x*length x*channels x := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [shape_eq x block h,List.prod_append]
  simp [batches,Nat.mul_assoc]

def width (x : TensorMetadata) (block : Nat) : Nat := 2^Nat.clog 2 (block*channels x)
theorem width_positive (x : TensorMetadata) (block : Nat) : 0< width x block := Nat.pow_pos (by decide)
theorem width_covers (x : TensorMetadata) (block : Nat) : block*channels x≤ width x block := Nat.le_pow_clog (by decide) _
theorem width_bound (x : TensorMetadata) (block : Nat) (h : InputContract x block) : width x block≤2^20 :=
  Nat.pow_le_pow_right (by decide) (Nat.clog_le_of_le_pow h.tile)

def parameters (x : TensorMetadata) (block : Nat) (query : Bool) (factor : ℝ) : Parameters :=
  ⟨batches x,length x,channels x,block,blocks x block,0,width x block,query,factor⟩

theorem native_admitted (x : TensorMetadata) (block : Nat) (h : InputContract x block)
    (query : Bool) (factor : ℝ) (positive : 0< channels x) :
    Revision002.Native.Admitted (parameters x block query factor) := by
  refine ⟨rfl,h.blockPositive,positive,width_covers x block,width_bound x block h,?_,?_⟩
  · have extent:=h.inputExtent
    rw [numel_eq x block h] at extent
    change 4*(batches x*length x*channels x)≤2^63-1
    omega
  · have extent:=h.scaleExtent
    change scaleCount x block<2^61 at extent
    change 4*(batches x*blocks x block)≤2^63-1
    unfold scaleCount at extent
    omega

#axiomsClean checkInputs_iff
#axiomsClean native_admitted
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Int8Quantization.Direct
