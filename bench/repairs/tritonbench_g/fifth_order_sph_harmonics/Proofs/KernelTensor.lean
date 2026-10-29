import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Access
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Storage
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelTensor
open VeriTile.Triton
open VeriTile.Torch.FP32 (Tensor TensorLaws)
noncomputable section
set_option maxHeartbeats 4000000

/-- Ordinary binding of the queried two-dimensional view to an actual shared
FP32 Tensor. The stride law covers sliced, transposed and zero-stride upstream
views; it does not assert contiguity or any kernel result. -/
structure Binding (view : TensorView) where
  tensor : Tensor
  laws : TensorLaws tensor
  metadata : tensor.metadata=view.tensor
  storageBase : tensor.storage.base=view.storageBase
  /-- Complete FP32 words in the observed byte storage; an unused trailing
  partial word does not narrow the source's admitted byte capacity. -/
  capacity : tensor.storage.elements=view.capacity/4
  strides : ∀ r,r<view.rows → ∀ c,c<view.columns →
    tensor.physicalIndex (r*view.columns+c)=
      tensor.storageOffset+r*view.rowStride.toNat+c*view.colStride.toNat

theorem numel (b : Binding view) (kind : TensorKind view) :
    b.tensor.metadata.numel=view.rows*view.columns := by
  rw [b.metadata]
  simp only [VeriTile.Torch.TensorMetadata.numel,kind.2.2.2.2.2,List.prod_cons,List.prod_nil,Nat.mul_one]

theorem pointer_nat (b : Binding view) (kind : TensorKind view)
    (rows : 0<view.rows) (columns : 0<view.columns) :
    view.pointer=b.tensor.storage.address b.tensor.storageOffset := by
  have positive : 0<b.tensor.metadata.numel := by rw [numel b kind];exact Nat.mul_pos rows columns
  have inside:=b.laws.inBounds 0 positive
  have offset:=b.strides 0 rows 0 columns
  simp only [Nat.zero_mul,Nat.zero_add,Nat.add_zero] at offset
  rw [offset] at inside
  have span:=b.tensor.storage.spanFits
  have bound : b.tensor.storage.address b.tensor.storageOffset<2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address]
    omega
  have pointer:=b.laws.pointer positive
  simp only [Tensor.dataPtr,b.metadata] at pointer
  simp only [TensorView.pointer,pointer,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]

/-- The source's observed data pointer plus strides is the actual physical
Tensor element address, including a nonzero storage offset. -/
theorem address (b : Binding view) (kind : TensorKind view) (r c : Nat)
    (hr : r<view.rows) (hc : c<view.columns) :
    Access.address view r c=b.tensor.storage.address (b.tensor.physicalIndex (r*view.columns+c)) := by
  rw [Access.address,pointer_nat b kind (by omega) (by omega),b.strides r hr c hc]
  simp only [VeriTile.Torch.FP32.Storage.address]
  ring

theorem actual_pointer (b : Binding view) :
    BitVec.ofNat 64 view.pointer=b.tensor.dataPtr := by
  simp [TensorView.pointer,Tensor.dataPtr,b.metadata]

def realPoint (memory : Nat → ℝ) (b : Binding view) (r c : Nat) : ℝ :=
  memory (b.tensor.storage.address (b.tensor.physicalIndex (r*view.columns+c)))

def nativePoint (heap : Storage.Heap) (b : Binding view) (r c : Nat) : FP.Value .fp32 :=
  VeriTile.Torch.FP32.logicalValue (fun a=>Packing.value (Storage.read heap a))
    b.tensor b.laws (r*view.columns+c)

theorem real_read (memory : Nat → ℝ) (b : Binding view) (kind : TensorKind view)
    (r c : Nat) (hr : r<view.rows) (hc : c<view.columns) :
    memory (Access.address view r c)=realPoint memory b r c := by
  rw [address b kind r c hr hc]
  rfl

theorem native_read (heap : Storage.Heap) (b : Binding view) (kind : TensorKind view)
    (r c : Nat) (hr : r<view.rows) (hc : c<view.columns) :
    Packing.value (Storage.read heap (Access.address view r c))=nativePoint heap b r c := by
  rw [address b kind r c hr hc]
  simp only [nativePoint,VeriTile.Torch.FP32.logicalValue,b.metadata,kind.2.2.2.1,
    Bool.false_eq_true,ite_false]

theorem active_bytes (b : Binding view) (kind : TensorKind view) (valid : TensorBounds view)
    (r c : Nat) (hr : r<view.rows) (hc : c<view.columns) (byte : Nat) (hb : byte<4) :
    let a:=b.tensor.storage.address (b.tensor.physicalIndex (r*view.columns+c));
    a%4=0 ∧ b.tensor.storage.base≤a+byte ∧
      a+byte<b.tensor.storage.base+4*b.tensor.storage.elements ∧ a+byte<2^64 := by
  have bounds:=Access.storage_access view valid r c hr hc
  rw [address b kind r c hr hc] at bounds
  have indexBound : r*view.columns+c<view.rows*view.columns := by
    have step:=Nat.mul_le_mul_right view.columns (Nat.succ_le_of_lt hr)
    rw [Nat.succ_mul] at step
    omega
  have inside:=b.laws.inBounds (r*view.columns+c) (by rw [numel b kind];exact indexBound)
  have span:=b.tensor.storage.spanFits
  dsimp only
  have aligned:=bounds.2.2.2
  refine ⟨aligned,?_,?_,?_⟩ <;>
    simp only [VeriTile.Torch.FP32.Storage.address] <;>
    omega

theorem coordinate_real (memory : Nat → ℝ) (b : Binding view) (valid : Coordinates view)
    (r : Nat) (hr : r<view.rows) (c : Fin 3) :
    memory (view.pointer+4*(r*3+c.val))=realPoint memory b r c.val := by
  rw [←Access.coordinate_address view valid]
  exact real_read memory b valid.1 r c.val hr (by rw [valid.2.1];exact c.isLt)

theorem coordinate_native (heap : Storage.Heap) (b : Binding view) (valid : Coordinates view)
    (r : Nat) (hr : r<view.rows) (c : Fin 3) :
    Packing.value (Storage.read heap (view.pointer+4*(r*3+c.val)))=nativePoint heap b r c.val := by
  rw [←Access.coordinate_address view valid]
  exact native_read heap b valid.1 r c.val hr (by rw [valid.2.1];exact c.isLt)

#axiomsClean pointer_nat
#axiomsClean address
#axiomsClean actual_pointer
#axiomsClean native_read
#axiomsClean active_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelTensor
