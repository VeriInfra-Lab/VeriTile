import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Kernels

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.LegacySource
open Lean VeriTile.Triton

structure Arguments where
  (X W RMS OUT : RegionName)
  (M N K sxb sxm sxk swk swn srms sob som son : Nat)
  (BM BN BK headDim : Nat)
  start : Int
  rotary : Bool

/-- The first two syntax declarations only introduce the incoming scalar
registers; `drop 2` removes them. Actual execution binds EPS and THETA words
directly, separately from the literal-zero operation.

Mathematical cdiv represents the native quotient/remainder form, which
avoids overflowing the numerator. Int64 bounds live in the proof helpers. -/
syntax "rmsScalarDomainSource%" "[" ident "," tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(rmsScalarDomainSource%[$weight:ident, $outputDType:tritonDType]) => do
    unless weight.getId == `half || weight.getId == `fp8 do
      Macro.throwError "expected half or fp8 weight storage"
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
    `(fun (p : Arguments) =>
      let X := p.X
      let W := p.W
      let RMS := p.RMS
      let OUT := p.OUT
      ComputeKernel.mk [X,W,RMS] [OUT] ((triton {
        tl.region X = tl.float32, W = $weightDType, RMS = $weightDType, OUT = $outputDType
        EPS = $((0 : ℝ))
        THETA = $((0 : ℝ))
        pid_batch = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
        pid_m = pid // tl.cdiv($(p.N), $(p.BN))
        pid_n = pid % tl.cdiv($(p.N), $(p.BN))
        offs_m = (pid_m * $(p.BM) + tl.arange(0, $(p.BM))) % $(p.M)
        offs_n = (pid_n * $(p.BN) + tl.arange(0, $(p.BN))) % $(p.N)
        offs_k = tl.arange(0, $(p.BK))
        x_ptrs = X + pid_batch * $(p.sxb) + offs_m[:, None] * $(p.sxm) + offs_k[None, :] * $(p.sxk)
        w_ptrs = W + offs_k[:, None] * $(p.swk) + offs_n[None, :] * $(p.swn)
        accumulator = tl.zeros([$(p.BM), $(p.BN)], dtype=tl.float32)
        rms_w_ptrs = RMS + tl.arange(0, $(p.BK))[None, :] * $(p.srms)
        x_sum = tl.zeros([$(p.BM), $(p.BK)], dtype=tl.float32)
        for step in range($(0), tl.cdiv($(p.K), $(p.BK))) {
          x = tl.load(x_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
          x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
          rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
          $rmsDecode*
          x = x * rms_w
          w = tl.load(w_ptrs, mask=offs_k[:, None] < $(p.K) - step * $(p.BK), other=0.0)
          $weightDecode*
          w = w.to(tl.float32)
          accumulator += tl.dot(x, w)
          x_ptrs += $(p.BK) * $(p.sxk)
          w_ptrs += $(p.BK) * $(p.swk)
          rms_w_ptrs += $(p.BK) * $(p.srms)
        }
        x_mean = tl.sum(x_sum, axis=1) / tl.toReal($(p.K)).to(tl.float32) + EPS
        x_norm = tl.math.rsqrt(x_mean)
        accumulator = accumulator * x_norm[:, None]
        if $((p.rotary : Bool)) {
          pairs = tl.reshape(accumulator, [$(p.BM), $(p.BN / 2), $(2)])
          real = tl.split(pairs, 0)
          imag = tl.split(pairs, 1)
          pair_n = pid_n * $(p.BN) + $(2) * tl.arange(0, $(p.BN / 2))
          row_m = pid_m * $(p.BM) + tl.arange(0, $(p.BM))
          exponent = tl.toReal(pair_n % $(p.headDim)).to(tl.float32) / $(p.headDim)
          frequency = tl.extra.cuda.libdevice.pow(THETA, exponent)
          position = $((p.start : Int)) + row_m
          angle = tl.toReal(position).to(tl.float32)[:, None] / frequency[None, :]
          cosine = tl.cos(angle)
          sine = tl.sin(angle)
          rotated_real = real * cosine - imag * sine
          rotated_imag = real * sine + imag * cosine
          accumulator = tl.reshape(tl.join(rotated_real, rotated_imag), [$(p.BM), $(p.BN)])
        }
        offs_m = pid_m * $(p.BM) + tl.arange(0, $(p.BM))
        offs_n = pid_n * $(p.BN) + tl.arange(0, $(p.BN))
        out_ptrs = OUT + pid_batch * $(p.sob) + offs_m[:, None] * $(p.som) + offs_n[None, :] * $(p.son)
        out_mask = (offs_m[:, None] < $(p.M)) & (offs_n[None, :] < $(p.N))
        tl.store(out_ptrs, accumulator.to($outputDType), mask=out_mask)
      }).surfaceBody.drop 2))

def kernel (fp8 outputFP8 : Bool) : Arguments → ComputeKernel :=
  match fp8,outputFP8 with
  | Bool.false,Bool.false => rmsScalarDomainSource%[half, tl.float16]
  | Bool.false,Bool.true => rmsScalarDomainSource%[half, tl.float8e5]
  | Bool.true,Bool.false => rmsScalarDomainSource%[fp8, tl.float16]
  | Bool.true,Bool.true => rmsScalarDomainSource%[fp8, tl.float8e5]

/-- Literal Q/K/V callee order and separate destination precision. -/
def qkv (fp8 : Bool) (q k v : Arguments) : ComputeKernel :=
  .mk [q.X,q.W,k.W,v.W,q.RMS] [q.OUT,k.OUT,v.OUT]
    ((kernel fp8 Bool.false {q with rotary:=Bool.true}).surfaceBody ++
     (kernel fp8 fp8 {k with rotary:=Bool.true}).surfaceBody ++
     (kernel fp8 fp8 {v with rotary:=Bool.false}).surfaceBody)

end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.LegacySource

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton

theorem readable_source_preserved (fp8 outputFP8 : Bool) (p : TypedSource.Arguments) :
    TypedSource.kernel fp8 outputFP8 p = LegacySource.kernel fp8 outputFP8
      ⟨p.X,p.W,p.RMS,p.OUT,p.M,p.N,p.K,p.sxb,p.sxm,p.sxk,p.swk,p.swn,p.srms,
       p.sob,p.som,p.son,p.BM,p.BN,p.BK,p.headDim,p.start,p.rotary⟩ := by
  cases fp8 <;> cases outputFP8 <;> rfl
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
