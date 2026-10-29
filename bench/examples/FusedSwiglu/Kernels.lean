import VeriTile.Triton.DSL.CallSignature

/-! New spec-derived Python, not a recovered historical benchmark. Complete
masked fused/two-stage programs preserve FP32 arithmetic, the BF16 intermediate
and final casts. Stage B explicitly converts its BF16 load to FP32. Separate
arithmetic uses enable_fp_fusion=False. -/
namespace VeriTile.Bench.Examples.FusedSwiglu.Kernels
open VeriTile Triton

def swiglu_fusedSignature : CallSignature where
  pointers := [("X",.fp32), ("Y",.fp32), ("OUT",.bf16)]
  scalars := []
  constexpr := ["ncols", "BLOCK_N"]

def swiglu_fused (X Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  with_triton_signature swiglu_fusedSignature in
  triton {
    start_col = tl.program_id(0) * BLOCK_N
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.0)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.0)
    sil = (x * tl.sigmoid(x)).to(tl.bfloat16)
    out = sil * y
    tl.store(OUT + cols, out.to(tl.bfloat16), mask=cols < ncols)
  }

def silu_stepSignature : CallSignature where
  pointers := [("X",.fp32), ("S",.bf16)]
  scalars := []
  constexpr := ["ncols", "BLOCK_N"]

def silu_step (X S : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  with_triton_signature silu_stepSignature in
  triton {
    start_col = tl.program_id(0) * BLOCK_N
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.0)
    s = x * tl.sigmoid(x)
    tl.store(S + cols, s.to(tl.bfloat16), mask=cols < ncols)
  }

def mul_stepSignature : CallSignature where
  pointers := [("S",.bf16), ("Y",.fp32), ("OUT",.bf16)]
  scalars := []
  constexpr := ["ncols", "BLOCK_N"]

def mul_step (S Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  with_triton_signature mul_stepSignature in
  triton {
    start_col = tl.program_id(0) * BLOCK_N
    cols = start_col + tl.arange(0, BLOCK_N)
    z = tl.load(S + cols, mask=cols < ncols, other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.0)
    out = z * y
    tl.store(OUT + cols, out.to(tl.bfloat16), mask=cols < ncols)
  }

def swiglu_unfused (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  ComputeKernel.seq [X, Y] [OUT]
    [silu_step X S ncols BLOCK_N, mul_step S Y OUT ncols BLOCK_N]
end VeriTile.Bench.Examples.FusedSwiglu.Kernels
