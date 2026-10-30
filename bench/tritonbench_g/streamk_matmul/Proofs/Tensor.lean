import bench.tritonbench_g.streamk_matmul.Proofs.SourceBridge
import bench.tritonbench_g.streamk_matmul.Proofs.Contract
import bench.tritonbench_g.streamk_matmul.Proofs.Index
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.TensorModel
open Whole001 Checked
open VeriTile.Torch
noncomputable section
set_option maxHeartbeats 8000000

/-- Adapter reuses the parent finite-memory proof without introducing another
Tensor model. All fields come from the shared actual FP32 Tensor binding. -/
def legacy (t : FP32.Tensor) : TensorBinding.Tensor :=
  ⟨t.object,t.metadata,t.storage.object,t.storage.base,t.storage.elements,t.storageOffset,t.physicalIndex⟩

theorem legacy_laws (t : FP32.Tensor) (laws : FP32.TensorLaws t)
    (hc : t.metadata.isContiguous=Bool.true) : TensorBinding.Laws (legacy t) := by
  refine ⟨laws.dtype,laws.layout,laws.inBounds,?_,laws.contiguous⟩
  intro nonempty
  have bound := laws.inBounds 0 nonempty
  rw [laws.contiguous hc 0 nonempty] at bound
  simp only [Nat.add_zero] at bound
  have span := t.storage.spanFits
  have pointerBound : t.storage.address t.storageOffset<2^64 := by
    unfold FP32.Storage.address
    omega
  change t.storage.base+4*t.storageOffset<2^64 at pointerBound
  have ptr := congrArg BitVec.toNat (laws.pointer nonempty)
  simpa only [FP32.Tensor.dataPtr,BitVec.toNat_ofNat,Nat.mod_eq_of_lt pointerBound,legacy,
    FP32.Storage.address] using ptr

def logicalReal (read : Nat → ℝ) (t : FP32.Tensor) (i : Nat) : ℝ :=
  read (t.storage.address (t.physicalIndex i))

def matrixProduct (read : Nat → ℝ) (a b : FP32.Tensor) (i j : Nat) : ℝ :=
  ∑ k : Fin (Checked.Contract.columns a.metadata),
    logicalReal read a (i*Checked.Contract.columns a.metadata+k.val)*
    logicalReal read b (k.val*Checked.Contract.columns b.metadata+j)

theorem word_logical (read : Nat → FP32.Word) (t : FP32.Tensor) (laws : FP32.TensorLaws t)
    (hn : t.metadata.isNeg=Bool.false) (i : Nat) :
    TensorBinding.logicalValue read (legacy t) i=FP32.logicalValue read t laws i := by
  simp only [FP32.logicalValue,hn,Bool.false_eq_true,if_false,
    TensorBinding.logicalValue,TensorBinding.physicalAddress,legacy,FP32.Storage.address]

def params (a b : FP32.Tensor) (c : Contract.Config) (h : Contract.InputContract a.metadata b.metadata c) : Source.Parameters :=
  TensorBinding.parameters (legacy a) (legacy b) c h.prior.profile

/-- Ordinary binding of the caller's output buffer. Shape/dtype/layout/address
and fresh writable storage are resource/model contracts, not extra restrictions
on the Python input tensors or assumptions about the kernel's computed value. -/
structure OutputBinding (p : Source.Parameters) (v : ByteExecution.Layout p) (out : FP32.Tensor) : Prop where
  laws : FP32.TensorLaws out
  shape : out.metadata.shape=[p.h.m,p.h.n]
  contiguous : out.metadata.isContiguous=Bool.true
  noNeg : out.metadata.isNeg=Bool.false
  pointer : out.metadata.dataPtr.toNat=v.outputBase
  aligned : v.outputBase%4=0

structure Buffers (a b : FP32.Tensor) (p : Source.Parameters) (v : ByteExecution.Layout p) (out : FP32.Tensor) : Prop where
  output : OutputBinding p v out
  memory : ByteExecution.Laws v
  scratchAligned : v.scratchActive=Bool.true → v.scratchBase%4=0
  outputA : ByteMemory.Separate (ByteExecution.cView v) ⟨a.storage.base,a.storage.elements⟩
  outputB : ByteMemory.Separate (ByteExecution.cView v) ⟨b.storage.base,b.storage.elements⟩
  scratchA : ByteMemory.Separate (ByteExecution.sView v) ⟨a.storage.base,a.storage.elements⟩
  scratchB : ByteMemory.Separate (ByteExecution.sView v) ⟨b.storage.base,b.storage.elements⟩

theorem output_numel (v : ByteExecution.Layout p) (out : FP32.Tensor) (h : OutputBinding p v out) :
    out.metadata.numel=p.h.m*p.h.n := by
  simp only [TensorMetadata.numel,h.shape,List.prod_cons,List.prod_nil,Nat.mul_one]

theorem real_output_read (v : ByteExecution.Layout p) (out : FP32.Tensor) (h : OutputBinding p v out)
    (read : Nat → ℝ) (i : Source.C p) :
    read (v.outputBase+4*i.val)=logicalReal read out i.val := by
  have result := TensorBinding.direct_read read (legacy out) (legacy_laws out h.laws h.contiguous)
    h.contiguous i.val (by change i.val<out.metadata.numel; rw [output_numel v out h]; exact i.isLt)
  simpa only [legacy,h.pointer,TensorBinding.logicalValue,TensorBinding.physicalAddress,
    logicalReal,FP32.Storage.address] using result

theorem native_output_read (v : ByteExecution.Layout p) (out : FP32.Tensor) (h : OutputBinding p v out)
    (read : Nat → FP32.Word) (i : Source.C p) :
    read (v.outputBase+4*i.val)=FP32.logicalValue read out h.laws i.val := by
  have result := TensorBinding.direct_read read (legacy out) (legacy_laws out h.laws h.contiguous)
    h.contiguous i.val (by change i.val<out.metadata.numel; rw [output_numel v out h]; exact i.isLt)
  simpa only [legacy,h.pointer] using result.trans (word_logical read out h.laws h.noNeg i.val)

theorem outside_storage (c : ByteMemory.View) (t : FP32.Tensor)
    (separate : ByteMemory.Separate c ⟨t.storage.base,t.storage.elements⟩)
    (address : Nat) (inside : t.storage.base≤address ∧ address<t.storage.base+4*t.storage.elements) :
    ByteMemory.Outside c address := by
  unfold ByteMemory.Outside ByteMemory.Separate at *
  dsimp only at separate
  omega

#axiomsClean legacy_laws
#axiomsClean word_logical
#axiomsClean real_output_read
#axiomsClean native_output_read
#specNonCircular matrixProduct avoiding [Source.call,Reader.execute,Source.chunkValue,Returned.value]
#specNonCircular Buffers avoiding [Source.call,Reader.execute,Returned.value]
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.TensorModel
