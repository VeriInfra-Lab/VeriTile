import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RequestCore

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostCumulative
open VeriTile.Triton HostFP HostRequest
noncomputable section

structure Request (α : Type) where
  B : Nat
  H : Nat
  T : Nat
  S : Nat
  BT : Nat
  BS : Nat
  input : Tensor α
  before : Nat → α

structure Request.Valid {α : Type} (x : Request α) : Prop where
  shape : x.input.shape=[x.B,x.H,x.T,x.S]
  bt : 16≤x.BT
  block : ∃ n, x.BT=2^n
  bs : x.BS=16 ∨ x.BS=32 ∨ x.BS=64

def Request.config {α : Type} (x : Request α) (input output : RegionName) : HostFP.Cumulative :=
  ⟨x.input.dtype.dtype,input,output,x.B*x.H,x.T,x.S,x.BT,x.BS⟩


structure Allocated {α : Type} (x : Request α) (input output : RegionName) where
  inputBuffer : Allocation.Buffer
  outputBuffer : Allocation.Buffer
  distinct : input≠output
  inputDtype : inputBuffer.dtype=x.input.dtype
  outputDtype : outputBuffer.dtype=.fp32
  inputExtent : x.B*x.H*(padded x.T x.BT)*(padded x.S 64) ≤ inputBuffer.extent
  outputExtent : x.B*x.H*(padded x.T x.BT)*(padded x.S 64) ≤ outputBuffer.extent
  privateInput : inputBuffer.allocationId≠x.input.allocationId
  privateOutput : outputBuffer.allocationId≠x.input.allocationId
  disjoint : inputBuffer.allocationId≠outputBuffer.allocationId

structure ReturnPlacement {α : Type} (x : Request α) where
  objectId : Nat
  allocationId : Nat
  privateOutput : allocationId≠x.input.allocationId

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostCumulative
