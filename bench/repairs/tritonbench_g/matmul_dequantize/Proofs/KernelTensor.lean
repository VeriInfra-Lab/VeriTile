import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.TensorBinding

namespace VeriTile.Bench.Repairs.MatmulDequantize.KernelTensor
open VeriTile.Triton
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open TensorBinding
noncomputable section

/-- Ordinary logical storage reads of a resolved half/int32/FP32 Tensor.
The reader uses the same named allocation and actual physical index as the
source byte binding. No arithmetic result or successful execution is assumed. -/
def logicalByte (t : TensorBinding.Tensor) (width : Nat)
    (raw : ByteBridge.Memory) (i b : Nat) : BitVec 8 :=
  raw t.storageName (width * t.physicalIndex i + b)

theorem logical_byte (d : Host.Tensor) (t : TensorBinding.Tensor)
    (binding : Binding d t) (width : Nat) (raw : ByteBridge.Memory)
    (r n : Nat) (hr : r < d.rows) (hn : n < d.columns) (b : Nat) :
    logicalByte t width raw (r*d.columns+n) b =
      raw d.region (width*(d.storageOffset+r*d.rowStride+n*d.columnStride)+b) := by
  simp only [logicalByte, binding.name, binding.coordinates r n hr hn]

/-- Absolute-address FP32 reader for this actual named allocation. -/
def fp32Read (d : Host.Tensor) (raw : ByteBridge.Memory) (address : Nat) :
    VeriTile.Torch.FP32.Word :=
  ⟨ByteBridge.decode32 (fun b => raw d.region (address-d.base+b.val))⟩

/-- The shared FP32 Tensor's logical value is exactly the input word used by
the source. The absence of a deferred negation flag comes from its checked
binding. All FP32 words remain admitted, including NaNs and infinities. -/
theorem shared_read (d : Host.Tensor) (t : TensorBinding.Tensor)
    (binding : Binding d t) (dtype : d.dtype = .fp32)
    (x : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws x)
    (selected : t.fp32 = some x) (raw : ByteBridge.Memory)
    (r n : Nat) (hr : r < d.rows) (hn : n < d.columns) :
    (VeriTile.Torch.FP32.logicalValue (fp32Read d raw) x laws (r*d.columns+n)).bits =
      ByteBridge.raw32 raw d (r*d.rowStride+n*d.columnStride) := by
  obtain ⟨y,hy,_,_,metadata,_,base,_,offset,coordinates⟩ := binding.shared dtype
  have same : y=x := Option.some.inj (hy.symm.trans selected)
  subst y
  have noNeg : x.metadata.isNeg = Bool.false := by rw [metadata]; exact binding.negative
  have coord : x.physicalIndex (r*d.columns+n) =
      d.storageOffset+r*d.rowStride+n*d.columnStride := by
    rw [coordinates]
    exact binding.coordinates r n hr hn
  simp only [VeriTile.Torch.FP32.logicalValue,noNeg,Bool.false_eq_true,↓reduceIte,
    fp32Read,VeriTile.Torch.FP32.Storage.address,base,binding.base,coord,
    Nat.add_sub_cancel_left,ByteBridge.raw32,Nat.add_assoc]

/-- A public connection of every live FP32 input/output descriptor to the
ordinary shared Tensor API, derived from metadata/storage binding alone. -/
def SharedReads (a : Host.Arguments) (route : Host.Route)
    (t : Tensors a route) (raw : ByteBridge.Memory) : Prop :=
  ∀ d ∈ live a route, d.dtype = .fp32 →
    ∀ x (laws : VeriTile.Torch.FP32.TensorLaws x), (t.get d.region).fp32 = some x →
      ∀ r n, r < d.rows → n < d.columns →
        (VeriTile.Torch.FP32.logicalValue (fp32Read d raw) x laws (r*d.columns+n)).bits =
          ByteBridge.raw32 raw d (r*d.rowStride+n*d.columnStride)

theorem shared_reads (a : Host.Arguments) (route : Host.Route)
    (t : Tensors a route) (raw : ByteBridge.Memory) : SharedReads a route t raw := by
  intro d member dtype x laws selected r n hr hn
  exact shared_read d (t.get d.region) (t.bound d member) dtype x laws selected raw r n hr hn

#axiomsClean logical_byte
#axiomsClean shared_read
#axiomsClean shared_reads
end
end VeriTile.Bench.Repairs.MatmulDequantize.KernelTensor
