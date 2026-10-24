import bench.repairs.tritonbench_g.matmul_kernel.Contract
import VeriTile.Torch.FP32
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelTensor
open VeriTile.Triton
open VeriTile.Torch.FP32 (Tensor TensorLaws)
noncomputable section
set_option maxHeartbeats 4000000

/-- Ordinary API binding of the prepared rank-two FP32 tensor. It records
actual strides, storage identity, nonzero offset and queried pointer. No
kernel execution or arithmetic result is a field of this structure. -/
structure Binding (v : View) (pointer storage device : Nat) where
  tensor : Tensor
  laws : TensorLaws tensor
  shape : tensor.metadata.shape=[v.rows,v.cols]
  deviceKind : tensor.metadata.device.kind="cuda"
  deviceIndex : tensor.metadata.device.index=device
  notNeg : tensor.metadata.isNeg=Bool.false
  pointerValue : tensor.dataPtr.toNat=pointer
  storageIdentity : tensor.storage.object=storage
  capacity : v.capacity=tensor.storage.elements-tensor.storageOffset
  strides : ∀ r,r<v.rows → ∀ c,c<v.cols →
    tensor.physicalIndex (r*v.cols+c)=tensor.storageOffset+r*v.sr+c*v.sc

theorem numel (b : Binding v pointer storage device) :
    b.tensor.metadata.numel=v.rows*v.cols := by
  simp [VeriTile.Torch.TensorMetadata.numel,b.shape]

theorem allocation (b : Binding v pointer storage device) : Checked.Allocation v := by
  by_cases hr : v.rows=0
  · exact Or.inl hr
  by_cases hc : v.cols=0
  · exact Or.inr (Or.inl hc)
  right;right
  have br : v.rows-1<v.rows := by omega
  have bc : v.cols-1<v.cols := by omega
  have bi : (v.rows-1)*v.cols+(v.cols-1)<b.tensor.metadata.numel := by
    rw [numel b]
    have hmul:=Nat.mul_le_mul_right v.cols (Nat.succ_le_of_lt br)
    rw [Nat.succ_mul] at hmul
    omega
  have h:=b.laws.inBounds _ bi
  rw [b.strides _ br _ bc] at h
  rw [b.capacity]
  omega

theorem pointer_nat (b : Binding v pointer storage device) (r c : Nat)
    (hr : r<v.rows) (hc : c<v.cols) :
    pointer=b.tensor.storage.address b.tensor.storageOffset := by
  have positive : 0<b.tensor.metadata.numel := by rw [numel b];exact Nat.mul_pos (by omega) (by omega)
  have idx:=b.laws.inBounds 0 positive
  have off:=b.strides 0 (by omega) 0 (by omega)
  simp only [Nat.zero_mul,Nat.zero_add,Nat.add_zero] at off
  rw [off] at idx
  have span:=b.tensor.storage.spanFits
  have bound : b.tensor.storage.address b.tensor.storageOffset<2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address]
    omega
  calc
    pointer=b.tensor.dataPtr.toNat:=b.pointerValue.symm
    _=b.tensor.storage.address b.tensor.storageOffset:=by
      rw [b.laws.pointer positive,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]

theorem address (b : Binding v pointer storage device) (r c : Nat)
    (hr : r<v.rows) (hc : c<v.cols) :
    pointer+4*v.addr r c=b.tensor.storage.address (b.tensor.physicalIndex (r*v.cols+c)) := by
  simp only [pointer_nat b r c hr hc,b.strides r hr c hc]
  simp only [View.addr,VeriTile.Torch.FP32.Storage.address]
  ring

def realPoint (memory : Nat → ℝ) (b : Binding v pointer storage device) (r c : Nat) : ℝ :=
  memory (b.tensor.storage.address (b.tensor.physicalIndex (r*v.cols+c)))
def nativePoint (memory : Nat → FP.Value .fp32) (b : Binding v pointer storage device) (r c : Nat) :=
  VeriTile.Torch.FP32.logicalValue memory b.tensor b.laws (r*v.cols+c)

theorem real_read (memory : Nat → ℝ) (b : Binding v pointer storage device)
    (r c : Nat) (hr : r<v.rows) (hc : c<v.cols) :
    memory (pointer+4*v.addr r c)=realPoint memory b r c := by
  rw [address b r c hr hc]
  rfl

theorem native_read (memory : Nat → FP.Value .fp32) (b : Binding v pointer storage device)
    (r c : Nat) (hr : r<v.rows) (hc : c<v.cols) :
    memory (pointer+4*v.addr r c)=nativePoint memory b r c := by
  rw [address b r c hr hc]
  simp [nativePoint,VeriTile.Torch.FP32.logicalValue,b.notNeg]

/-- Prepared inputs to the kernel, independent of host returns/allocation.
The metadata checker and actual shared Tensor bindings describe one launch. -/
structure Prepared where
  metadata : Checked.Metadata
  checked : Checked.checkInputs metadata=.ok ()
  a : Binding metadata.config.A metadata.pointerA metadata.storageA metadata.deviceA
  b : Binding metadata.config.B metadata.pointerB metadata.storageB metadata.deviceB
  c : Binding metadata.config.C metadata.pointerC metadata.storageC metadata.deviceC
  /-- Ordinary physical-allocation law for distinct backing storage identities. -/
  separateA : metadata.storageC≠metadata.storageA →
    c.tensor.storage.base+4*c.tensor.storage.elements≤a.tensor.storage.base ∨
    a.tensor.storage.base+4*a.tensor.storage.elements≤c.tensor.storage.base
  separateB : metadata.storageC≠metadata.storageB →
    c.tensor.storage.base+4*c.tensor.storage.elements≤b.tensor.storage.base ∨
    b.tensor.storage.base+4*b.tensor.storage.elements≤c.tensor.storage.base

def Prepared.config (p : Prepared) := p.metadata.config
theorem Prepared.valid (p : Prepared) : p.config.Valid :=
  (Checked.admitted_domain p.metadata p.checked (allocation p.a) (allocation p.b) (allocation p.c)).1

theorem active_bytes (b : Binding v pointer storage device) (r c : Nat)
    (hr : r<v.rows) (hc : c<v.cols) (byte : Fin 4) :
    let addr:=b.tensor.storage.address (b.tensor.physicalIndex (r*v.cols+c));
    b.tensor.storage.base≤addr+byte.val ∧
      addr+byte.val<b.tensor.storage.base+4*b.tensor.storage.elements ∧ addr+byte.val<2^64 := by
  have hi : r*v.cols+c<b.tensor.metadata.numel := by
    rw [numel b]
    have h:=Nat.mul_le_mul_right v.cols (Nat.succ_le_of_lt hr)
    rw [Nat.succ_mul] at h
    omega
  have bounds:=b.laws.inBounds _ hi
  have span:=b.tensor.storage.spanFits
  have hb:=byte.isLt
  simp only [VeriTile.Torch.FP32.Storage.address]
  omega

#axiomsClean allocation
#axiomsClean address
#axiomsClean native_read
#axiomsClean Prepared.valid
#axiomsClean active_bytes
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelTensor
