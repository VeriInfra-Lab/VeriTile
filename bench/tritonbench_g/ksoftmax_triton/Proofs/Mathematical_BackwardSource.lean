/-
Source-audit target: _softmax_backward
Source-audit Lean: backwardSource%
Source-audit args: gradientType=tl.float32, savedType=tl.float32, destinationType=tl.float32, version=original

This is the declared selected fp32 view. The metadata selects a source for
comparison; it does not assert that the audit or every native profile passed.
-/
import VeriTile.Triton.DSL

/-!
Complete typed quotation of `_softmax_backward`. The three storage types are
independent at the JIT boundary: the Python wrapper only checks equal shapes.
The explicit destination conversion exposes the original store conversion.
All source branches and both read/write masks remain in the quotation.

This module is a source definition, not a claim that every storage profile
has passed native compilation, source comparison, or execution verification.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Typed
open Lean VeriTile.Triton

inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat →
  Bool → Bool → Bool → ComputeKernel

syntax "backwardSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(backwardSource%[$gradientType:tritonDType, $savedType:tritonDType,
        $destinationType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let gradientAddress : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| grad_out_ptrs = GradOut + m * $(stride_gm) + n * $(stride_gn) + k)
      else
        `(tritonStmt| grad_out_ptrs = GradOut + $(stride_gm) * m + n * $(stride_gn) + k)
    `(fun (GradIn GradOut Out : RegionName)
        (stride_bm stride_bn stride_gm stride_gn stride_om stride_on K DEPTH : Nat)
        (LOG CAUSAL IS_FP16 : Bool) =>
      ComputeKernel.mk [GradOut, Out] [GradIn] (triton {
  tl.region GradOut = $gradientType, Out = $savedType, GradIn = $destinationType
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  $gradientAddress
  out_ptrs = Out + m * $(stride_om) + n * $(stride_on) + k
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0))
  o = tl.load(out_ptrs, mask=io_mask, other=float(0))
  if CAUSAL {
    zero = float(0)
    zero = (zero).to(g.dtype)
    g = tl.where(k > n, zero, g)
    o = tl.where(k > n, zero, o)
  }
  if LOG {
    s = tl.sum(g, 0)
    if IS_FP16 {
      o = (o).to(tl.float32)
    }
    grad_in = g - tl.exp(o) * s
  } else {
    s = tl.sum(g * o, 0)
    grad_in = o * (g - s)
  }
  grad_in_ptrs = GradIn + m * $(stride_bm) + n * $(stride_bn) + k
  tl.store(grad_in_ptrs, (grad_in).to($destinationType:tritonDType), mask=k < $(K))
}).surfaceBody)

/-- Initial homogeneous profile family. The quotation above separately
exposes all three dtype arguments; this family does not assert that the
Python wrapper requires equal dtypes. -/
def homogeneousKernel (storage : Storage) (commuted : Bool) : Source :=
  match storage, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float32, tl.float32, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float64, tl.float64, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float64, tl.float64, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float16, tl.float16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e5, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e5, tl.float8e5, tl.float8e5, indexCommuted]

end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Typed
