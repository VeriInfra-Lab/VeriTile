import VeriTile.Triton

/-!
Source-audit target: pow_func_scalar_tensor_kernel_rank_1
Source-audit Lean: powScalarFloating%
Source-audit args: output=tl.float32, single=0, candidate=0
Source-audit params: val0=base, in0_ptr=X, out0_ptr=Y, in0_stride0=sx, out0_stride0=sy, s0=N, num_tasks=tasks, tiles_per_cta=tiles, tile_size0=B
Source-audit constexpr: {"python":{"one_tile_per_cta":false},"lean":{"source":"powScalarFloating%","args":{"output":"tl.float32","single":0,"candidate":0}}}
Source-audit orders: {"python":{"in0_stride_order0":0,"out0_stride_order0":0},"lean":{"source":"powScalarFloating%","args":{"output":"tl.float32","single":0,"candidate":0}},"wrapper":{"name":"pow_func_scalar_tensor_wrapper_rank_1","launch":"pow_func_scalar_tensor_kernel_rank_1","source_sha256":"142398199794dc7df724c341d3f86b1af60f20a9280b695efeac5e68d757471d","rank":1}}
Source-audit extern: {"python":{"helper":"pow_func_scalar_tensor","alias":"_pow"},"lean":{"source":"powScalarFloating%","args":{"output":"tl.float32","single":0,"candidate":0}},"native":{"profile":"pow_fp32_fp32_grid_sm90","sha256":"798104fc0d8bf03353de00e05dd85413578421549e8b79f22f36d52a9a8798c7"}}
Selected fp32 floating multi-tile original profile only; single-tile, other floating storage and integer-exponent profiles remain separate obligations.
-/

/-! The actual fp32-exponent source for both constexpr branches. The current
block-pointer infrastructure admits natural strides only. The separate dynamic
int32-exponent source lives in Proofs/IntegerKernels. Signed strides, integer
runtime base conversion and nonfloating destinations remain explicit source
obligations. The historical exact-real surfaces live in Proofs/LegacyReal. -/
namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedSource
open VeriTile.Triton Lean

syntax "powScalarFloating%" "[" tritonDType "]" num num : term
set_option hygiene false in
macro_rules
  | `(powScalarFloating%[$output:tritonDType] $single:num $candidate:num) => do
    let offset : TSyntax `tritonStmt ← if candidate.getNat = 0 then
      `(tritonStmt| offset0 = tile_id0 * $(B))
      else `(tritonStmt| offset0 = $(B) * tile_id0)
    let tail : Array (TSyntax `tritonStmt) := #[
      ← `(tritonStmt| tile_id0 = tile_id), offset,
      ← `(tritonStmt| in0_bptr = tl.make_block_ptr(base=X, shape=($(N)),
          strides=($(sx)), offsets=(offset0), block_shape=($(B)), order=(0))),
      ← `(tritonStmt| in0 = (tl.load(in0_bptr,
          boundary_check=([0] : List Nat))).to(X.type.element_ty)),
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
        tl.region X = tl.float32, Y = $output
        pid = tl.program_id(0)
        num_tiles0 = tl.cdiv($(N), $(B))
        $branch:tritonStmt*
      }).surfaceBody)

/-- Formal floating destination labels; source exponent storage is fp32.
The pinned native extern dispatcher separately admits int32 exponents. -/
inductive FloatingOutput where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

abbrev Source := ℝ → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

def fp32Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float32] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float32] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float32] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float32] 0 1

def fp64Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float64] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float64] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float64] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float64] 0 1

def fp16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float16] 0 1

def bf16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.bfloat16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.bfloat16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.bfloat16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.bfloat16] 0 1

def f8e4Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e4nv] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e4nv] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e4nv] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e4nv] 0 1

def f8e5Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e5] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e5] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e5] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e5] 0 1

/-- Both complete branch bodies with unspecified boundary padding, explicit
base fp32 conversion and destination conversion. Candidate commutes only the
natural tile-offset product; task count and the unused cdiv remain present.
The template resolves Python's `out0_bptr.type.element_ty` to the declared
destination label before expanding `.to`; this is source metadata resolution,
not an additional cast or a change to Python's conversion semantics. -/
def kernel (output : FloatingOutput) (oneTile candidate : Bool) : Source :=
  match output with
  | .fp32 => fp32Kernel oneTile candidate
  | .fp64 => fp64Kernel oneTile candidate
  | .fp16 => fp16Kernel oneTile candidate
  | .bf16 => bf16Kernel oneTile candidate
  | .f8e4 => f8e4Kernel oneTile candidate
  | .f8e5 => f8e5Kernel oneTile candidate

end VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedSource
