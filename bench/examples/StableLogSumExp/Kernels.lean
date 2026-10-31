import VeriTile.Triton.DSL

/-! Direct log-sum-exp and its repaired conditional candidate. Both retain
libdevice.exp, tl.log, the scalar output address and bf16 output conversion.
The candidate preserves both numerical fallback branches. -/

namespace VeriTile.Bench.Examples.StableLogSumExp.Kernels
open VeriTile Triton

def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := libdevice.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

/-- Candidate composed from `log_mul_split .tl` and `log_exp_cancel .tl`.
All logs retain tl.log, exactly as in the original direct reference. The rounded product keeps
its direct log on [0.5, 2]; the split branch eliminates log(exp(m)) only when
0.5 < |m| ≤ 80. Unused log arguments are one and unused exp arguments are zero.
The source preserves both conditional fallbacks and the final bf16 cast. -/
def optimizedLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x := tl.load($(xReg) + offs)
  m := tl.max(x, axis=0)
  e := libdevice.exp(x - m)
  s := tl.sum(e, axis=0)
  p := s * libdevice.exp(m)
  keep := (p >= 0.5) & (p <= 2.0)
  direct := tl.log(tl.where(keep, p, 1.0))
  split_m := tl.where(keep, 0.0, m)
  simplify := (0.5 < tl.abs(split_m)) & (tl.abs(split_m) <= 80.0)
  fallback := tl.log(libdevice.exp(tl.where(simplify, 0.0, split_m)))
  log_em := tl.where(simplify, split_m, fallback)
  split := tl.log(tl.where(keep, 1.0, s)) + log_em
  y := tl.where(keep, direct, split)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

end VeriTile.Bench.Examples.StableLogSumExp.Kernels
