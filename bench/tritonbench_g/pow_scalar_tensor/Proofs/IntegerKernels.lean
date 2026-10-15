import bench.tritonbench_g.pow_scalar_tensor.Kernels

/-! Complete dynamic-int32 exponent source, using the pinned native powif
path. Integer exponent provenance is retained through the identity input cast;
integer exponent values are never converted to floating. Natural strides and
a real scalar binder are the present source scope; signed addressing, integer
scalar binders and nonfloating output casts remain separate paths. -/
namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedIntegerSource
open VeriTile.Triton Lean

syntax "powScalarInteger%" "[" tritonDType "]" num num : term
set_option hygiene false in
macro_rules
  | `(powScalarInteger%[$output:tritonDType] $single:num $candidate:num) => do
    let offset : TSyntax `tritonStmt ← if candidate.getNat = 0 then
      `(tritonStmt| offset0 = tile_id0 * $(B))
      else `(tritonStmt| offset0 = $(B) * tile_id0)
    let tail : Array (TSyntax `tritonStmt) := #[
      ← `(tritonStmt| tile_id0 = tile_id), offset,
      ← `(tritonStmt| in0_bptr = tl.make_block_ptr(base=X, shape=($(N)),
          strides=($(sx)), offsets=(offset0), block_shape=($(B)), order=(0))),
      ← `(tritonStmt| in0 = (tl.load(in0_bptr,
          boundary_check=([0] : List Nat))).to(tl.int32)),
      ← `(tritonStmt| out0 = tl.extra.cuda.libdevice.pow(
          $((base : ℝ)).to(tl.float32), in0)),
      ← `(tritonStmt| out0_bptr = tl.make_block_ptr(base=Y, shape=($(N)),
          strides=($(sy)), offsets=(offset0), block_shape=($(B)), order=(0))),
      ← `(tritonStmt| tl.store(out0_bptr, (out0).to($output:tritonDType),
          boundary_check=([0] : List Nat))) ]
    let branch : Array (TSyntax `tritonStmt) ← if single.getNat = 1 then
      pure (#[← `(tritonStmt| tile_id = pid)] ++ tail)
      else pure #[
        ← `(tritonStmt| num_ctas = tl.num_programs(0)),
        ← `(tritonStmt| for j in range($(0), $(tiles)) {
          tile_id = pid + j * num_ctas
          $tail:tritonStmt*
        }) ]
    `(fun (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) =>
      ComputeKernel.mk [X] [Y] (triton {
        tl.region X = tl.int32, Y = $output
        pid = tl.program_id(0)
        num_tiles0 = tl.cdiv($(N), $(B))
        $branch:tritonStmt*
      }).surfaceBody)

/-- Destination labels are shared with the fp32-exponent source. -/
abbrev FloatingOutput := TypedSource.FloatingOutput

abbrev Source := ℝ → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

def fp32Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarInteger%[tl.float32] 1 0
  | Bool.true, Bool.true => powScalarInteger%[tl.float32] 1 1
  | Bool.false, Bool.false => powScalarInteger%[tl.float32] 0 0
  | Bool.false, Bool.true => powScalarInteger%[tl.float32] 0 1

def fp64Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarInteger%[tl.float64] 1 0
  | Bool.true, Bool.true => powScalarInteger%[tl.float64] 1 1
  | Bool.false, Bool.false => powScalarInteger%[tl.float64] 0 0
  | Bool.false, Bool.true => powScalarInteger%[tl.float64] 0 1

def fp16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarInteger%[tl.float16] 1 0
  | Bool.true, Bool.true => powScalarInteger%[tl.float16] 1 1
  | Bool.false, Bool.false => powScalarInteger%[tl.float16] 0 0
  | Bool.false, Bool.true => powScalarInteger%[tl.float16] 0 1

def bf16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarInteger%[tl.bfloat16] 1 0
  | Bool.true, Bool.true => powScalarInteger%[tl.bfloat16] 1 1
  | Bool.false, Bool.false => powScalarInteger%[tl.bfloat16] 0 0
  | Bool.false, Bool.true => powScalarInteger%[tl.bfloat16] 0 1

def f8e4Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarInteger%[tl.float8e4nv] 1 0
  | Bool.true, Bool.true => powScalarInteger%[tl.float8e4nv] 1 1
  | Bool.false, Bool.false => powScalarInteger%[tl.float8e4nv] 0 0
  | Bool.false, Bool.true => powScalarInteger%[tl.float8e4nv] 0 1

def f8e5Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarInteger%[tl.float8e5] 1 0
  | Bool.true, Bool.true => powScalarInteger%[tl.float8e5] 1 1
  | Bool.false, Bool.false => powScalarInteger%[tl.float8e5] 0 0
  | Bool.false, Bool.true => powScalarInteger%[tl.float8e5] 0 1

/-- Both complete branch bodies with unspecified boundary padding, explicit
base fp32 conversion, same-width int32 exponent copy and destination conversion. Candidate commutes only the
natural tile-offset product; task count and the unused cdiv remain present.
The template resolves both pointers' `type.element_ty` to their declared
labels before expanding `.to`; this is source metadata resolution,
not an additional cast or a change to Python's conversion semantics. -/
def kernel (output : FloatingOutput) (oneTile candidate : Bool) : Source :=
  match output with
  | .fp32 => fp32Kernel oneTile candidate
  | .fp64 => fp64Kernel oneTile candidate
  | .fp16 => fp16Kernel oneTile candidate
  | .bf16 => bf16Kernel oneTile candidate
  | .f8e4 => f8e4Kernel oneTile candidate
  | .f8e5 => f8e5Kernel oneTile candidate

end VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedIntegerSource
