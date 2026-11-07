import bench.tritonbench_g.swiglu_triton.Proofs.Support.Parameters
import VeriTile.Triton.DSL.PureJitFunctions

/-! Canonical repair bodies and their existing row-index commutation candidate.
The helper has its own original body. Pointer storage and runtime strides are
provided by the call; the bodies retain every cast, mask and source operation. -/
namespace VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluTriton (Storage)
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def siluSource : PureJitFunction := with_triton_params (x) constexpr_args [] in triton {
    return x * tl.sigmoid(x)
}

def forwardSignature (storage partner : Storage) : CallSignature where
  pointers := [("a_ptr",match storage with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("b_ptr",match partner with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("c_ptr",match storage with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5)]
  scalars := [("stride",.integer)]
  constexpr := ["n_cols","BLOCK_SIZE"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def backwardSignature (storage partner gradient : Storage) : CallSignature where
  pointers := [("dc_ptr",match gradient with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("a_ptr",match storage with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("b_ptr",match partner with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5)]
  scalars := [("stride",.integer)]
  constexpr := ["n_cols","BLOCK_SIZE"]
  readPorts := ["dc_ptr","a_ptr","b_ptr"]
  writePorts := ["a_ptr","b_ptr"]

def forwardTypedKernel (storage partner : Storage)
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (forwardSignature storage partner) call_args [stride := stride] pure_functions [silu := siluSource] in triton {
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)
  }

def forwardCommutedTypedKernel (storage partner : Storage)
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (forwardSignature storage partner) call_args [stride := stride] pure_functions [silu := siluSource] in triton {
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += stride * program_id
    b_ptr += stride * program_id
    c_ptr += stride * program_id

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)
  }

def backwardTypedKernel (storage partner gradient : Storage)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (backwardSignature storage partner gradient) call_args [stride := stride] in triton {
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)
  }

def backwardCommutedTypedKernel (storage partner gradient : Storage)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (backwardSignature storage partner gradient) call_args [stride := stride] in triton {
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += stride * program_id
    a_ptr += stride * program_id
    b_ptr += stride * program_id

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)
  }

def forwardKernel (a b c : RegionName) (stride cols block : Nat) : ComputeKernel :=
  forwardTypedKernel .fp32 .fp32 a b c stride cols block

def forwardIndexCommuted (a b c : RegionName) (stride cols block : Nat) : ComputeKernel :=
  forwardCommutedTypedKernel .fp32 .fp32 a b c stride cols block

def backwardKernel (dc a b : RegionName) (stride cols block : Nat) : ComputeKernel :=
  backwardTypedKernel .fp32 .fp32 .fp32 dc a b stride cols block

def backwardIndexCommuted (dc a b : RegionName) (stride cols block : Nat) : ComputeKernel :=
  backwardCommutedTypedKernel .fp32 .fp32 .fp32 dc a b stride cols block

/-- Exactly the source configuration branches; HIP has 16 warps at the largest
blocks while CUDA has 32. Columns beyond 65536 are rejected before launch. -/
def warps (hip : Bool) (block : Nat) : Nat :=
  if block ≥ 32768 then (if hip then 16 else 32)
  else if block ≥ 8192 then 16 else if block ≥ 2048 then 8 else 4

/-- Actual contiguous rows have stride cols, including non-power-of-two widths. -/
def directSource (backward : Bool) (block cols : Nat) : ComputeKernel :=
  if backward then backwardKernel "DC" "A" "B" cols cols block
  else forwardKernel "A" "B" "C" cols cols block

/-- Existing typed-source candidate: exchange only the signed row-base
multiplication. Both actual stages retain their floating instruction order. -/
def directIndexCommutedSource (backward : Bool) (block cols : Nat) : ComputeKernel :=
  if backward then backwardIndexCommuted "DC" "A" "B" cols cols block
  else forwardIndexCommuted "A" "B" "C" cols cols block

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SwigluTriton
