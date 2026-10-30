import VeriTile.Triton.DSL.CallSignature

/-! Full original and candidate JIT function bodies, including the distinct
in-place Python functions. Runtime n_elements retains its declared int32 ABI;
the Nat call binding is the existing mathematical index projection. Only
BLOCK_SIZE is constexpr. All physical native bounds remain in the proofs. -/
namespace VeriTile.Bench.Repairs.TritonMul2
open VeriTile.Triton

def sourceSignature : CallSignature where
  pointers := [("in_ptr0",.fp32),("out_ptr",.fp32)]
  scalars := [("n_elements",.i32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["in_ptr0"]
  writePorts := ["out_ptr"]

def inplaceSignature : CallSignature where
  pointers := [("ptr",.fp32)]
  scalars := [("n_elements",.i32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["ptr"]
  writePorts := ["ptr"]

def sourceKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let in_ptr0 := X
  let out_ptr := O
  let n_elements := n
  let BLOCK_SIZE := B
  with_triton_call sourceSignature call_args [n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = 2 * x
    tl.store(out_ptr + offsets, output, mask=mask)
  }

def inplaceSourceKernel (X : RegionName) (n B : Nat) : ComputeKernel :=
  let ptr := X
  let n_elements := n
  let BLOCK_SIZE := B
  with_triton_call inplaceSignature call_args [n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(ptr + offsets, mask=mask)
    output = 2 * x
    tl.store(ptr + offsets, output, mask=mask)
  }

def commutedSourceKernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let in_ptr0 := X
  let out_ptr := O
  let n_elements := n
  let BLOCK_SIZE := B
  with_triton_call sourceSignature call_args [n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = x * 2
    tl.store(out_ptr + offsets, output, mask=mask)
  }

def inplaceCommutedSourceKernel (X : RegionName) (n B : Nat) : ComputeKernel :=
  let ptr := X
  let n_elements := n
  let BLOCK_SIZE := B
  with_triton_call inplaceSignature call_args [n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(ptr + offsets, mask=mask)
    output = x * 2
    tl.store(ptr + offsets, output, mask=mask)
  }

def grid (n block : Nat) : Nat := (n+block-1)/block
end VeriTile.Bench.Repairs.TritonMul2
