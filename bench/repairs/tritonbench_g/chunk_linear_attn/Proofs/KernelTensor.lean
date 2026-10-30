import bench.repairs.tritonbench_g.chunk_linear_attn.Contract
import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.streamk_matmul.Proofs.ByteMemory

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelTensor
open VeriTile Triton FP.Structural Torch
namespace Bytes
abbrev read32 := VeriTile.Bench.Repairs.StreamK.Checked.ByteMemory.read32
end Bytes
noncomputable section
set_option maxHeartbeats 8000000

/-- Adapter for the already-proved finite word memory. The actual Tensor is the
shared core type, retaining object identity, backing storage and view offset.
These are ordinary kernel-buffer API facts, never output equations. -/
structure Binding (old : Host.Tensor) (actual : Torch.FP32.Tensor) : Prop where
  laws : Torch.FP32.TensorLaws actual
  identity : actual.object=old.identity
  shape : actual.metadata.shape=old.shape
  device : actual.metadata.device=⟨"cuda",old.device⟩
  pointer : actual.metadata.dataPtr.toNat=4*old.base
  contiguous : actual.metadata.isContiguous=Bool.true
  noNeg : actual.metadata.isNeg=Bool.false

/-- Only descriptors actually supplied to a kernel are dense here. The raw
backward cotangent may be strided; its existing materialized buffer is bound by
the incoming/terminalGradient descriptors. No copy implementation is proved. -/
structure Buffers (c : Host.Invocation) (a : Host.Allocations) where
  actual : Host.Tensor → Torch.FP32.Tensor
  binding : ∀ r old,Host.descriptor c a r=some old → Binding old (actual old)

theorem numel_eq (old : Host.Tensor) (actual : Torch.FP32.Tensor) (h : Binding old actual) :
    actual.metadata.numel=old.shape.prod := by simp only [Torch.TensorMetadata.numel,h.shape]

theorem address (old : Host.Tensor) (actual : Torch.FP32.Tensor) (h : Binding old actual)
    (i : Nat) (hi : i<old.shape.prod) :
    actual.storage.address (actual.physicalIndex i)=4*(old.base+i) := by
  have bound : i<actual.metadata.numel := by rw [numel_eq old actual h]; exact hi
  have nonempty : 0<actual.metadata.numel := by omega
  have first := h.laws.inBounds 0 nonempty
  rw [h.laws.contiguous h.contiguous 0 nonempty] at first
  have span := actual.storage.spanFits
  have small : actual.storage.address actual.storageOffset<2^64 := by
    unfold Torch.FP32.Storage.address
    omega
  have ptr := congrArg BitVec.toNat (h.laws.pointer nonempty)
  simp only [Torch.FP32.Tensor.dataPtr,BitVec.toNat_ofNat,Nat.mod_eq_of_lt small,h.pointer] at ptr
  rw [h.laws.contiguous h.contiguous i bound]
  simp only [Torch.FP32.Storage.address] at ptr ⊢
  omega

def logicalReal (memory : Nat → Cell ℝ) (actual : Torch.FP32.Tensor) (i : Nat) : ℝ :=
  (memory (actual.storage.address (actual.physicalIndex i)/4)).read .real

theorem real_read (old : Host.Tensor) (actual : Torch.FP32.Tensor) (h : Binding old actual)
    (memory : Nat → Cell ℝ) (i : Nat) (hi : i<old.shape.prod) :
    logicalReal memory actual i=(memory (old.base+i)).read .real := by
  simp only [logicalReal,address old actual h i hi,Nat.mul_div_cancel_left _ (by decide : 0<4)]

def logicalNative (bytes : Nat → BitVec 8) (actual : Torch.FP32.Tensor)
    (laws : Torch.FP32.TensorLaws actual) (i : Nat) : Torch.FP32.Word :=
  Torch.FP32.logicalValue (fun addr => ⟨Bytes.read32 bytes addr⟩) actual laws i

theorem native_read (old : Host.Tensor) (actual : Torch.FP32.Tensor) (h : Binding old actual)
    (bytes : Nat → BitVec 8) (i : Nat) (hi : i<old.shape.prod) :
    logicalNative bytes actual h.laws i=⟨Bytes.read32 bytes (4*(old.base+i))⟩ := by
  simp only [logicalNative,Torch.FP32.logicalValue,h.noNeg,Bool.false_eq_true,if_false,
    address old actual h i hi]

def wordMemory (bytes : Nat → BitVec 8) : Nat → BitVec 32 := fun i => Bytes.read32 bytes (4*i)
def realMemory (c : Host.Invocation) (a : Host.Allocations) (physical : Nat → Cell ℝ)
    (fallback : Memory ℝ) : Memory ℝ := Host.logical c a physical fallback
def nativeMemory (c : Host.Invocation) (a : Host.Allocations) (bytes : Nat → BitVec 8)
    (fallback : RegionName → Nat → BitVec 32) : RegionName → Nat → BitVec 32 :=
  regionBits c a (wordMemory bytes) fallback

theorem real_input (c : Host.Invocation) (a : Host.Allocations) (buffers : Buffers c a)
    (physical : Nat → Cell ℝ) (fallback : Memory ℝ) (r : RegionName) (t : Host.Tensor)
    (ht : Host.descriptor c a r=some t) (i : Nat) (hi : i<t.shape.prod) :
    (realMemory c a physical fallback r i).read .real=logicalReal physical (buffers.actual t) i := by
  simp only [realMemory,Host.logical,ht]
  exact (real_read t (buffers.actual t) (buffers.binding r t ht) physical i hi).symm

/-- Every input word in symbolic source memory is decoded from the actual four
FP32 storage bytes and equals the shared Tensor logical value. This covers all
FP32 bit patterns; no finiteness or numerical-agreement premise is present. -/
theorem native_input (c : Host.Invocation) (a : Host.Allocations) (buffers : Buffers c a)
    (bytes : Nat → BitVec 8) (fallback : RegionName → Nat → BitVec 32) (r : RegionName) (t : Host.Tensor)
    (ht : Host.descriptor c a r=some t) (i : Nat) (hi : i<t.shape.prod) :
    nativeMemory c a bytes fallback r i=
      (logicalNative bytes (buffers.actual t) (buffers.binding r t ht).laws i).bits := by
  rw [native_read t (buffers.actual t) (buffers.binding r t ht) bytes i hi]
  simp only [nativeMemory,regionBits,ht,wordMemory]

#axiomsClean address
#axiomsClean real_input
#axiomsClean native_input
#specNonCircular Binding avoiding [Host.execute,Host.publish,Real.Call.result]
#specNonCircular Buffers avoiding [Host.execute,Host.publish,Real.Call.result]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelTensor
