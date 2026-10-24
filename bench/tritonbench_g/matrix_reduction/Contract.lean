import bench.tritonbench_g.matrix_reduction.Kernels
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton TypedSource

def storageDType : Storage → VeriTile.Torch.DType
  | .fp16 => .other "float16"
  | .fp32 => .floating .fp32
def storageBytes : Storage → Nat | .fp16 => 2 | .fp32 => 4

/-- Successful generated input buffer. The original wrapper constructs this
Tensor with randn; it has no caller-provided Tensor to normalize or reject.
These ordinary allocation facts contain no random-sample or maximum result. -/
structure InputStorage (storage : Storage) (M N : Nat) where
  metadata : VeriTile.Torch.TensorMetadata
  base : Nat
  shape : metadata.shape = [M,N]
  dtype : metadata.dtype = storageDType storage
  layout : metadata.layout = .strided
  cuda : metadata.device.kind = "cuda"
  contiguous : metadata.isContiguous = Bool.true
  noNeg : metadata.isNeg = Bool.false
  pointer : metadata.dataPtr = BitVec.ofNat 64 base
  aligned : base % storageBytes storage = 0
  span : base + storageBytes storage * (M*N) ≤ 2^64

def InputStorage.address (x : InputStorage storage M N) (i : Fin M) (j : Fin N) : Nat :=
  x.base + storageBytes storage * (i.val*N+j.val)

/-- Real specification uses the same generated contiguous storage addresses. -/
def InputValues (x : InputStorage storage M N) (read : Nat → ℝ)
    (s : BlockState) : Prop :=
  ∀ i : Fin M, ∀ j : Fin N,
    s.readMem "x" (i.val*N+j.val) = read (x.address i j)

end VeriTile.Bench.TritonBenchG.MatrixReduction
