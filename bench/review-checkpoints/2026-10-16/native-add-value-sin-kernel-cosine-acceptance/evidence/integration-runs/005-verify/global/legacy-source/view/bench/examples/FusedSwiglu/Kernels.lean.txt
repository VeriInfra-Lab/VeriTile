import VeriTile.Triton.DSL

/-!
FusedSwiglu: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.FusedSwiglu.Kernels
open VeriTile Triton

/-- Original fused SwiGLU, including the intermediate and output bf16 casts. -/
def swiglu_fused (X Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  sil = (x * tl.sigmoid(x)).to(tl.bfloat16)
  out = sil * y
  tl.store(OUT + cols, (out).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- Parsed step A before making its default fp32 arithmetic explicit. -/
def silu_step_source (X S : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  s = x * tl.sigmoid(x)
  tl.store(S + cols, (s).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- The `s` assignment uses the same fp32 arithmetic as the fused expression.
This annotation changes no operation, cast, load, store or statement order. -/
def siluPrecision : ComputeStmt → ComputeStmt
  | .assign .real shape "s" (.alg e) =>
      .assign .real shape "s" (.alg (.withPrecision .real .fp32 e))
  | stmt => stmt

/-- Step A of the unfused pipeline, with its source default precision explicit.
The intermediate bf16 store remains after the sigmoid and multiplication. -/
def silu_step (X S : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  match silu_step_source X S ncols BLOCK_N with
  | .mk inputs outputs body => .mk inputs outputs (body.map siluPrecision)

/-- Step B of the unfused pipeline: load the bf16 intermediate back and
multiply by `y`. The `S` load is bf16-typed (it reads bf16 cells); the
mixed-dtype multiply upcasts it to ℝ exactly. -/
def mul_step (S Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  z = tl.load(S + cols, mask=cols < $(ncols)).to(tl.bfloat16)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  out = z * y
  tl.store(OUT + cols, (out).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- Original two-stage pipeline, modeled as one concatenated kernel. -/
def swiglu_unfused (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  ComputeKernel.seq [X, Y] [OUT]
    [silu_step X S ncols BLOCK_N, mul_step S Y OUT ncols BLOCK_N]

end VeriTile.Bench.Examples.FusedSwiglu.Kernels
