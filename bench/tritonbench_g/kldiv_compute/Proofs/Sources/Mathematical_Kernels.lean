import bench.tritonbench_g.kldiv_compute.Proofs.Sources.Selected
import VeriTile.Triton
import VeriTile.Examples.Common

/-! Compatibility names for every retained dtype profile. All programs are
instances of the two canonical raw quotations in `Kernels.lean`; this module
contains no alternative source generator. The existing native support
classifier is unchanged. -/
namespace VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource
open VeriTile.Triton

/-- Both-fp8 input pairs remain outside the pinned native compiler's admitted
source profiles. This classifier does not certify hardware lowering. -/
def sourceSupported (x y : Storage) : Bool :=
  match x, y with
  | .f8e4, .f8e4 | .f8e4, .f8e5 | .f8e5, .f8e4 | .f8e5, .f8e5 => Bool.false
  | _, _ => Bool.true

abbrev Source := RegionName → RegionName → RegionName → Nat → Nat → ComputeKernel

def kernel_fp32_fp32 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp32 .fp32 output v

def kernel_fp32_fp16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp32 .fp16 output v

def kernel_fp32_bf16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp32 .bf16 output v

def kernel_fp32_fp64 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp32 .fp64 output v

def kernel_fp32_f8e4 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp32 .f8e4 output v

def kernel_fp32_f8e5 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp32 .f8e5 output v

def kernel_fp16_fp32 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp16 .fp32 output v

def kernel_fp16_fp16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp16 .fp16 output v

def kernel_fp16_bf16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp16 .bf16 output v

def kernel_fp16_fp64 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp16 .fp64 output v

def kernel_fp16_f8e4 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp16 .f8e4 output v

def kernel_fp16_f8e5 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp16 .f8e5 output v

def kernel_bf16_fp32 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .bf16 .fp32 output v

def kernel_bf16_fp16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .bf16 .fp16 output v

def kernel_bf16_bf16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .bf16 .bf16 output v

def kernel_bf16_fp64 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .bf16 .fp64 output v

def kernel_bf16_f8e4 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .bf16 .f8e4 output v

def kernel_bf16_f8e5 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .bf16 .f8e5 output v

def kernel_fp64_fp32 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp64 .fp32 output v

def kernel_fp64_fp16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp64 .fp16 output v

def kernel_fp64_bf16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp64 .bf16 output v

def kernel_fp64_fp64 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp64 .fp64 output v

def kernel_fp64_f8e4 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp64 .f8e4 output v

def kernel_fp64_f8e5 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .fp64 .f8e5 output v

def kernel_f8e4_fp32 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e4 .fp32 output v

def kernel_f8e4_fp16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e4 .fp16 output v

def kernel_f8e4_bf16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e4 .bf16 output v

def kernel_f8e4_fp64 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e4 .fp64 output v

def kernel_f8e4_f8e4 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e4 .f8e4 output v

def kernel_f8e4_f8e5 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e4 .f8e5 output v

def kernel_f8e5_fp32 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e5 .fp32 output v

def kernel_f8e5_fp16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e5 .fp16 output v

def kernel_f8e5_bf16 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e5 .bf16 output v

def kernel_f8e5_fp64 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e5 .fp64 output v

def kernel_f8e5_f8e4 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e5 .f8e4 output v

def kernel_f8e5_f8e5 (output : Storage) (v : Bool) : Source :=
  VeriTile.Bench.Repairs.KldivCompute.KernelDSL.family .f8e5 .f8e5 output v


def kernel (x y output : Storage) (v : Bool) : Source :=
  match x, y with
  | .fp32, .fp32 => kernel_fp32_fp32 output v
  | .fp32, .fp16 => kernel_fp32_fp16 output v
  | .fp32, .bf16 => kernel_fp32_bf16 output v
  | .fp32, .fp64 => kernel_fp32_fp64 output v
  | .fp32, .f8e4 => kernel_fp32_f8e4 output v
  | .fp32, .f8e5 => kernel_fp32_f8e5 output v
  | .fp16, .fp32 => kernel_fp16_fp32 output v
  | .fp16, .fp16 => kernel_fp16_fp16 output v
  | .fp16, .bf16 => kernel_fp16_bf16 output v
  | .fp16, .fp64 => kernel_fp16_fp64 output v
  | .fp16, .f8e4 => kernel_fp16_f8e4 output v
  | .fp16, .f8e5 => kernel_fp16_f8e5 output v
  | .bf16, .fp32 => kernel_bf16_fp32 output v
  | .bf16, .fp16 => kernel_bf16_fp16 output v
  | .bf16, .bf16 => kernel_bf16_bf16 output v
  | .bf16, .fp64 => kernel_bf16_fp64 output v
  | .bf16, .f8e4 => kernel_bf16_f8e4 output v
  | .bf16, .f8e5 => kernel_bf16_f8e5 output v
  | .fp64, .fp32 => kernel_fp64_fp32 output v
  | .fp64, .fp16 => kernel_fp64_fp16 output v
  | .fp64, .bf16 => kernel_fp64_bf16 output v
  | .fp64, .fp64 => kernel_fp64_fp64 output v
  | .fp64, .f8e4 => kernel_fp64_f8e4 output v
  | .fp64, .f8e5 => kernel_fp64_f8e5 output v
  | .f8e4, .fp32 => kernel_f8e4_fp32 output v
  | .f8e4, .fp16 => kernel_f8e4_fp16 output v
  | .f8e4, .bf16 => kernel_f8e4_bf16 output v
  | .f8e4, .fp64 => kernel_f8e4_fp64 output v
  | .f8e4, .f8e4 => kernel_f8e4_f8e4 output v
  | .f8e4, .f8e5 => kernel_f8e4_f8e5 output v
  | .f8e5, .fp32 => kernel_f8e5_fp32 output v
  | .f8e5, .fp16 => kernel_f8e5_fp16 output v
  | .f8e5, .bf16 => kernel_f8e5_bf16 output v
  | .f8e5, .fp64 => kernel_f8e5_fp64 output v
  | .f8e5, .f8e4 => kernel_f8e5_f8e4 output v
  | .f8e5, .f8e5 => kernel_f8e5_f8e5 output v

end VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource
