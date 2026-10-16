import VeriTile.Triton.DSL

/-!
Source-audit target: rms_matmul_rbe
Source-audit Lean: rmsQkvSource%
Source-audit params: x_ptr=X, w_ptr=W, rms_w_ptr=RMS, out_ptr=OUT, stride_x_batch=sxb, stride_x_m=sxm, stride_x_k=sxk, stride_w_k=swk, stride_w_n=swn, stride_rms_w=srms, stride_out_batch=sob, stride_out_m=som, stride_out_n=son, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK
Source-audit constexpr: {"python":{"USE_FP8":false},"lean":{"source":"rmsQkvSource%","args":{"input":"tl.float16","weight":"half","outputDType":"tl.float16","version":"original"}}}
Source-audit args: input=tl.float16, weight=half, outputDType=tl.float16, version=original
Selected fp16 input/weights/output original profile only; fp8, other storage and the separate qkv JIT remain separate obligations.
-/

/-! Complete typed callee source for both wrapper weight profiles. Input X
has six formal floating storage profiles. Output storage is independent of
weight storage: the FP8 Q pass writes fp16, while K and V write E5M2.
Each unmasked K iteration, integer power, dot and decode/conversion is retained.
The candidate commutes only the final output-row natural index product.
Native frontend admission is a separate obligation from this formal family. -/
namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.TypedSource
open Lean VeriTile.Triton
set_option linter.unusedVariables false
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000

syntax "rmsQkvSource%" "[" tritonDType "," ident "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(rmsQkvSource%[$input:tritonDType, $weight:ident, $outputDType:tritonDType, $version:ident]) => do
    unless weight.getId == `half || weight.getId == `fp8 do
      Macro.throwError "expected half or fp8 weight storage"
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let fp8 := weight.getId == `fp8
    let weightDType : TSyntax `tritonDType ← if fp8 then `(tritonDType| tl.int8)
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

/-- Output storage can be selected independently of USE_FP8. Both fp16
and E5M2 output profiles are needed by the source QKV caller. -/
def kernel (input : Storage) (fp8 outputFP8 variant : Bool) : Source :=
  match input, fp8, outputFP8, variant with
  | .fp16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float16, half, tl.float16, original]
  | .fp16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float16, half, tl.float16, commuted]
  | .fp16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float16, half, tl.float8e5, original]
  | .fp16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float16, half, tl.float8e5, commuted]
  | .fp16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float16, original]
  | .fp16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float16, commuted]
  | .fp16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float8e5, original]
  | .fp16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float8e5, commuted]
  | .bf16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float16, original]
  | .bf16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float16, commuted]
  | .bf16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, original]
  | .bf16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, commuted]
  | .bf16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, original]
  | .bf16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, commuted]
  | .bf16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, original]
  | .bf16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, commuted]
  | .fp32, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float32, half, tl.float16, original]
  | .fp32, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float32, half, tl.float16, commuted]
  | .fp32, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float32, half, tl.float8e5, original]
  | .fp32, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float32, half, tl.float8e5, commuted]
  | .fp32, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float16, original]
  | .fp32, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float16, commuted]
  | .fp32, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float8e5, original]
  | .fp32, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float8e5, commuted]
  | .fp64, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float64, half, tl.float16, original]
  | .fp64, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float64, half, tl.float16, commuted]
  | .fp64, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float64, half, tl.float8e5, original]
  | .fp64, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float64, half, tl.float8e5, commuted]
  | .fp64, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float16, original]
  | .fp64, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float16, commuted]
  | .fp64, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float8e5, original]
  | .fp64, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float8e5, commuted]
  | .f8e4, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float16, original]
  | .f8e4, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float16, commuted]
  | .f8e4, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, original]
  | .f8e4, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, commuted]
  | .f8e4, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, original]
  | .f8e4, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, commuted]
  | .f8e4, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, original]
  | .f8e4, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, commuted]
  | .f8e5, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float16, original]
  | .f8e5, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float16, commuted]
  | .f8e5, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float8e5, original]
  | .f8e5, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float8e5, commuted]
  | .f8e5, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float16, original]
  | .f8e5, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float16, commuted]
  | .f8e5, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, original]
  | .f8e5, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, commuted]

/-- The exact three-call expansion of rms_matmul_rbe_qkv. All passes restart
from program ids and preserve their independent weight/output strides.
The source's rotary flags True, True, False are retained at the call sites;
the callee does not use them or apply a rotary epilogue. -/
def qkv (input : Storage) (fp8 variant : Bool)
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ) : ComputeKernel :=
  .mk [X, QW, KW, VW, RMS] [Q, KOut, V]
    ((kernel input fp8 Bool.false variant X QW RMS Q M N K
      sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK start Bool.true theta EPS).surfaceBody ++
     (kernel input fp8 fp8 variant X KW RMS KOut M N K
      sxb sxm sxk skwk skwn srms skb skm skn BM BN BK start Bool.true theta EPS).surfaceBody ++
     (kernel input fp8 fp8 variant X VW RMS V M N K
      sxb sxm sxk svwk svwn srms svb svm svn BM BN BK start Bool.false theta EPS).surfaceBody)

end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.TypedSource
