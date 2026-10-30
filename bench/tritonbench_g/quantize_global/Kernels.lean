import VeriTile.Triton

/-! Source-only definition: FP32 input/maximum, explicit div.rn numerical graph,
llrint and int8 store. Native integer casts are checked in Proofs/Indices. -/
namespace QuantizeGlobalRepair
open VeriTile.Triton
noncomputable def source (X Maximum Out : RegionName) (N base B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Maximum = tl.float32, Out = tl.int8
  pid = tl.program_id(0)
  offsets = pid * $(B) + tl.arange(0, $(B)) + $(base)
  mask = offsets < $(N)
  safe_offsets = tl.where(mask, offsets, $((0 : Nat)))
  x = tl.load(X + safe_offsets, mask=mask, other=0.0).to(tl.float32)
  absmax = tl.load(Maximum).to(tl.float32)
  denominator = tl.where(absmax == (0.0).to(tl.float32), 1.0, absmax)
  normalized = x / denominator
  scaled = 127.0 * normalized
  lower_bounded = tl.where(scaled < (-127.0).to(tl.float32), -127.0, scaled)
  bounded = tl.where(lower_bounded > (127.0).to(tl.float32), 127.0, lower_bounded)
  output = tl.extra.cuda.libdevice.llrint(bounded)
  tl.store(Out + safe_offsets, output, mask=mask)
}


end QuantizeGlobalRepair
