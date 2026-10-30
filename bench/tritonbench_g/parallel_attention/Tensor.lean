import bench.tritonbench_g.parallel_attention.NativeObserved
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.TensorBinding
open VeriTile.Triton FP.Structural
open NativeStage
noncomputable section
set_option maxHeartbeats 2000000

/-- Actual Tensor identity and storage-backed API metadata. This view binds a
source region, not a new allocation or a host-summed result. -/
structure View where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  contiguous : tensor.metadata.isContiguous=Bool.true
  noNeg : tensor.metadata.isNeg=Bool.false
  region : RegionName

def View.base (v : View) : Nat := v.tensor.metadata.dataPtr.toNat

theorem pointer_base (v : View) (positive : 0<v.tensor.metadata.numel) :
    v.base=(VeriTile.Torch.FP32.dataView v.tensor v.laws v.contiguous).base := by
  have pointer := VeriTile.Torch.FP32.actual_pointer v.tensor v.laws v.contiguous positive
  have span := (VeriTile.Torch.FP32.dataView v.tensor v.laws v.contiguous).spanFits
  have bound : (VeriTile.Torch.FP32.dataView v.tensor v.laws v.contiguous).base<2^64 := by
    change _+4*v.tensor.metadata.numel≤2^64 at span
    omega
  simpa only [View.base,VeriTile.Torch.FP32.DataView.address,Nat.mul_zero,Nat.add_zero,
    BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] using (congrArg BitVec.toNat pointer).symm

/-- Observe only aligned FP32 words of the actual Tensor storage window.
All other addresses retain the incoming memory reader. -/
def project (read : Nat → VeriTile.Torch.FP32.Word) (v : View) (s : State Word) (address : Nat) : VeriTile.Torch.FP32.Word :=
  if v.base≤address ∧ address<v.base+4*v.tensor.metadata.numel ∧ (address-v.base)%4=0 then
    ⟨(s.mem v.region ((address-v.base)/4)).read .real⟩ else read address

theorem projected (read : Nat → VeriTile.Torch.FP32.Word) (v : View) (s : State Word)
    (i : Nat) (inside : i<v.tensor.metadata.numel) :
    (project read v s (v.base+4*i)).bits=(s.mem v.region i).read .real := by
  have bound : v.base+4*i<v.base+4*v.tensor.metadata.numel := by omega
  simp [project,bound,Nat.mul_mod_right,Nat.mul_div_cancel_left]

/-- Exact Tensor logical-value observation, including a nonzero storage
 offset. Lazy negative views are excluded by the checked View metadata. -/
theorem logical_read (read : Nat → VeriTile.Torch.FP32.Word) (v : View) (s : State Word)
    (i : Nat) (inside : i<v.tensor.metadata.numel) :
    (VeriTile.Torch.FP32.logicalValue (project read v s) v.tensor v.laws i).bits=
      (s.mem v.region i).read .real := by
  rw [← VeriTile.Torch.FP32.direct_read_eq_logical _ v.tensor v.laws v.contiguous v.noNeg i inside]
  have base := pointer_base v (by omega)
  change (project read v s (_+4*i)).bits=_
  rw [← base]
  exact projected read v s i inside

/-- Source output readback transports directly to the user's supplied Tensor;
there is no desired Tensor-value premise. -/
theorem observed_value (read : Nat → VeriTile.Torch.FP32.Word) (v : View) (s : State Word)
    (i : Nat) (inside : i<v.tensor.metadata.numel) (word : Word)
    (cell : s.mem v.region i=.mk .real word) :
    (VeriTile.Torch.FP32.logicalValue (project read v s) v.tensor v.laws i).bits=word := by
  rw [logical_read read v s i inside,cell,Cell.read_same]

/-- Full address frame: exterior, unaligned, and inactive Tensor words all
retain the current input memory. The only binding is the input word reader. -/
theorem project_frame (read : Nat → VeriTile.Torch.FP32.Word) (v : View) (before after : State Word)
    (binding : ∀ i,i<v.tensor.metadata.numel → (before.mem v.region i).read .real=(read (v.base+4*i)).bits)
    (writes : Nat → Prop) (frame : ∀ i,¬writes i → after.mem v.region i=before.mem v.region i)
    (address : Nat) (outside : ¬∃ i,i<v.tensor.metadata.numel ∧ writes i ∧ address=v.base+4*i) :
    project read v after address=read address := by
  unfold project
  split
  · rename_i active
    have pos : address=v.base+4*((address-v.base)/4) := by omega
    have hi : (address-v.base)/4<v.tensor.metadata.numel := by omega
    have missing : ¬writes ((address-v.base)/4) := by
      intro h; exact outside ⟨_,hi,h,pos⟩
    rw [frame _ missing,binding _ hi,← pos]
  · rfl

#axiomsClean pointer_base
#axiomsClean projected
#axiomsClean logical_read
#axiomsClean observed_value
#axiomsClean project_frame
#specNonCircular project avoiding [NativeStage.queryRun,NativeStage.keyValueRun,NativeStage.forwardRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.TensorBinding
