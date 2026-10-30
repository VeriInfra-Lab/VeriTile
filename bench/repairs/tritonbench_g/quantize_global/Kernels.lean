import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature

/-! Canonical original FP32 precise-division quantizer. The quotation is the
unchanged selected Python function body. Its runtime integer annotations and
pointer storage types are outside the body; native bounds remain in Indices. -/
namespace QuantizeGlobalRepair
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("absmax_ptr",.fp32),("output_ptr",.i8)]
  scalars := [("n_elements",.i64),("base",.i64)]
  constexpr := ["BLOCK_SIZE"]

noncomputable def source (X Maximum Out : RegionName) (N base B : Nat) : ComputeKernel :=
  let x_ptr := X
  let absmax_ptr := Maximum
  let output_ptr := Out
  let BLOCK_SIZE := B
  with_triton_call callSignature call_args [n_elements := N, base := base] in triton {
      pid = tl.program_id(axis=0).to(tl.int64)
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64) + base
      mask = offsets < n_elements
      safe_offsets = tl.where(mask, offsets, 0)
      x = tl.load(x_ptr + safe_offsets, mask=mask, other=0.0).to(tl.float32)
      absmax = tl.load(absmax_ptr).to(tl.float32)
      denominator = tl.where(absmax == 0.0, 1.0, absmax)
      normalized = tl.div_rn(x, denominator)
      scaled = 127.0 * normalized
      lower_bounded = tl.where(scaled < -127.0, -127.0, scaled)
      bounded = tl.where(lower_bounded > 127.0, 127.0, lower_bounded)
      output = tl.extra.cuda.libdevice.llrint(bounded)
      tl.store(output_ptr + safe_offsets, output, mask=mask)
  }

end QuantizeGlobalRepair
