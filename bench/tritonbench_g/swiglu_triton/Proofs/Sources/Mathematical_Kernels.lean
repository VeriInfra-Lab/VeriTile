import bench.tritonbench_g.swiglu_triton.Kernels

/-! Compatibility interfaces for all 36 forward and 216 backward storage
profiles. Every specialization uses the same canonical pasted body. -/
namespace VeriTile.Bench.TritonBenchG.SwigluTriton
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton

def typedFwd (storage partner : Storage) (commuted : Bool)
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  if commuted then forwardCommutedTypedKernel storage partner a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  else forwardTypedKernel storage partner a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE

def typedBwd (storage partner gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  if commuted then backwardCommutedTypedKernel storage partner gradient dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  else backwardTypedKernel storage partner gradient dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp32_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp32 .fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp32_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp32 .fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp32_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp32 .bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp32_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp32 .fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp32_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp32 .f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp32_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp32 .f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp16_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp16 .fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp16_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp16 .fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp16_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp16 .bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp16_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp16 .fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp16_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp16 .f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp16_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp16 .f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_bf16_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .bf16 .fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_bf16_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .bf16 .fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_bf16_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .bf16 .bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_bf16_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .bf16 .fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_bf16_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .bf16 .f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_bf16_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .bf16 .f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp64_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp64 .fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp64_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp64 .fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp64_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp64 .bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp64_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp64 .fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp64_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp64 .f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_fp64_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .fp64 .f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e4_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e4 .fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e4_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e4 .fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e4_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e4 .bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e4_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e4 .fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e4_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e4 .f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e4_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e4 .f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e5_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e5 .fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e5_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e5 .fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e5_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e5 .bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e5_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e5 .fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e5_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e5 .f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

def typedBwd_f8e5_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  typedBwd .f8e5 .f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluTriton
