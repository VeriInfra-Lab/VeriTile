import VeriTile.Triton.DSL

/-!
Source-audit args: input=tl.float16, weight=half, version=original
Selected fp16 input and half-weight original matmul profile only; fp8, other input storage and rbe_triton remain separate obligations.
-/

/-
Source-audit target: rms_matmul_rbe
Source-audit Lean: rmsRbeSource%
Source-audit params: x_ptr=X, w_ptr=W, rms_w_ptr=RMS, out_ptr=OUT, stride_x_batch=sxb, stride_x_m=sxm, stride_x_k=sxk, stride_w_k=swk, stride_w_n=swn, stride_rms_w=srms, stride_out_batch=sob, stride_out_m=som, stride_out_n=son, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK
Source-audit constexpr: {"python":{"USE_FP8":false},"lean":{"source":"rmsRbeSource%","args":{"input":"tl.float16","weight":"half","version":"original"}}}
This selects the second Python JIT. The separate unresolved rbe_triton source
and every unresolved dtype/weight specialization remain recorded obligations.
-/

/-! Complete typed `rms_matmul_rbe` source, including both weight profiles,
all unmasked ceiling-counted K iterations, native integer power, typed dot,
the separate fp32 accumulation and each E5M2 decode/conversion. The wrapper's
int8 arm stores through an E5M2 pointer; the fp16 arm stores through fp16.
Unused start-position, rotary flag and theta parameters remain in the signature.
The candidate commutes only the final output-row index product.

The additional `rbe_triton` JIT has an unresolved `get_freq_multi_tokens` call
in the Python module. It is a separate source-resolution obligation, recorded
in source_scope.md; these definitions do not supply a replacement callee. -/
namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.TypedSource
open Lean VeriTile.Triton
set_option linter.unusedVariables false
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000

syntax "rmsRbeSource%" "[" tritonDType "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(rmsRbeSource%[$input:tritonDType, $weight:ident, $version:ident]) => do
    unless weight.getId == `half || weight.getId == `fp8 do
      Macro.throwError "expected half or fp8 weight storage"
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let fp8 := weight.getId == `fp8
    let weightDType : TSyntax `tritonDType ← if fp8 then `(tritonDType| tl.int8)
      else `(tritonDType| tl.float16)
    let outputDType : TSyntax `tritonDType ← if fp8 then `(tritonDType| tl.float8e5)
      else `(tritonDType| tl.float16)
    let rmsDecode : Array (TSyntax `tritonStmt) ← if fp8 then
      pure #[← `(tritonStmt| rms_w = rms_w.to(tl.float8e5, bitcast=True)),
        ← `(tritonStmt| rms_w = rms_w.to(tl.float16))]
      else pure #[]
    let weightDecode : Array (TSyntax `tritonStmt) ← if fp8 then
      pure #[← `(tritonStmt| w = w.to(tl.float8e5, bitcast=True)),
        ← `(tritonStmt| w = w.to(tl.float32)),
        ← `(tritonStmt| w = w.to(tl.float16))]
      else pure #[]
    let outputRow : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| offs_m = pid_m * $(BM) + tl.arange(0, $(BM)))
      else `(tritonStmt| offs_m = $(BM) * pid_m + tl.arange(0, $(BM)))
    `(fun (X W RMS OUT : RegionName)
      (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
      (start_token_position : Int) (RBE_EPILOGUE : Bool) (THETA EPS : ℝ) =>
      ComputeKernel.mk [X, W, RMS] [OUT] (triton {
        tl.region X = $input, W = $weightDType, RMS = $weightDType, OUT = $outputDType
        pid_batch = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
        pid_m = pid // tl.cdiv($(N), $(BN))
        pid_n = pid % tl.cdiv($(N), $(BN))
        offs_m = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
        offs_n = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
        offs_k = tl.arange(0, $(BK))
        x_ptrs = X + (pid_batch * $(sxb) + offs_m[:, None] * $(sxm) + offs_k[None, :] * $(sxk))
        w_ptrs = W + (offs_k[:, None] * $(swk) + offs_n[None, :] * $(swn))
        accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
        rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
        x_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
        for _i in range($(0), tl.cdiv($(K), $(BK))) {
          x = tl.load(x_ptrs)
          x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
          rms_w = tl.load(rms_w_ptrs)
          $rmsDecode*
          x = x * rms_w
          w = tl.load(w_ptrs)
          $weightDecode*
          accumulator += tl.dot(x, w)
          x_ptrs += $(BK) * $(sxk)
          w_ptrs += $(BK) * $(swk)
          rms_w_ptrs += $(BK) * $(srms)
        }
        x_mean = tl.sum(x_sum, axis=1) / $(K) + $(EPS)
        x_norm = tl.math.rsqrt(x_mean)
        accumulator = accumulator * x_norm[:, None]
        $outputRow
        offs_n = pid_n * $(BN) + tl.arange(0, $(BN))
        out_ptrs = OUT + (pid_batch * $(sob) + offs_m[:, None] * $(som) + offs_n[None, :] * $(son))
        out_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(N))
        tl.store(out_ptrs, accumulator.to($outputDType), mask=out_mask)
      }).surfaceBody)

inductive Storage where
  | fp16 | bf16 | fp32 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat →
  Nat → Nat → Nat → Int → Bool → ℝ → ℝ → ComputeKernel

/-- The full floating-input family. The wrapper does not constrain X dtype;
frontend dot acceptance and native lowering must be checked separately. -/
def kernel (input : Storage) (fp8 variant : Bool) : Source :=
  match input, fp8, variant with
  | .fp16, Bool.false, Bool.false => rmsRbeSource%[tl.float16, half, original]
  | .fp16, Bool.false, Bool.true => rmsRbeSource%[tl.float16, half, commuted]
  | .fp16, Bool.true, Bool.false => rmsRbeSource%[tl.float16, fp8, original]
  | .fp16, Bool.true, Bool.true => rmsRbeSource%[tl.float16, fp8, commuted]
  | .bf16, Bool.false, Bool.false => rmsRbeSource%[tl.bfloat16, half, original]
  | .bf16, Bool.false, Bool.true => rmsRbeSource%[tl.bfloat16, half, commuted]
  | .bf16, Bool.true, Bool.false => rmsRbeSource%[tl.bfloat16, fp8, original]
  | .bf16, Bool.true, Bool.true => rmsRbeSource%[tl.bfloat16, fp8, commuted]
  | .fp32, Bool.false, Bool.false => rmsRbeSource%[tl.float32, half, original]
  | .fp32, Bool.false, Bool.true => rmsRbeSource%[tl.float32, half, commuted]
  | .fp32, Bool.true, Bool.false => rmsRbeSource%[tl.float32, fp8, original]
  | .fp32, Bool.true, Bool.true => rmsRbeSource%[tl.float32, fp8, commuted]
  | .fp64, Bool.false, Bool.false => rmsRbeSource%[tl.float64, half, original]
  | .fp64, Bool.false, Bool.true => rmsRbeSource%[tl.float64, half, commuted]
  | .fp64, Bool.true, Bool.false => rmsRbeSource%[tl.float64, fp8, original]
  | .fp64, Bool.true, Bool.true => rmsRbeSource%[tl.float64, fp8, commuted]
  | .f8e4, Bool.false, Bool.false => rmsRbeSource%[tl.float8e4nv, half, original]
  | .f8e4, Bool.false, Bool.true => rmsRbeSource%[tl.float8e4nv, half, commuted]
  | .f8e4, Bool.true, Bool.false => rmsRbeSource%[tl.float8e4nv, fp8, original]
  | .f8e4, Bool.true, Bool.true => rmsRbeSource%[tl.float8e4nv, fp8, commuted]
  | .f8e5, Bool.false, Bool.false => rmsRbeSource%[tl.float8e5, half, original]
  | .f8e5, Bool.false, Bool.true => rmsRbeSource%[tl.float8e5, half, commuted]
  | .f8e5, Bool.true, Bool.false => rmsRbeSource%[tl.float8e5, fp8, original]
  | .f8e5, Bool.true, Bool.true => rmsRbeSource%[tl.float8e5, fp8, commuted]

end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.TypedSource
