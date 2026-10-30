import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_pad_routing_map
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vneg_ptr : RegionName := "vneg"
  let vremainder_ptr : RegionName := "vremainder"
  let veq_ptr : RegionName := "veq"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vcumsum_ptr : RegionName := "vcumsum"
  let vle_ptr : RegionName := "vle"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vwhere_ptr : RegionName := "vwhere"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  triton {
    r0_z = tl.arange($(2048))
    r0_value = tl.where(tl.load($(input0_ptr) + (((r0_z % $(128)) * $(16)) + ((r0_z / $(128)) % $(16))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r0_value = tl.broadcast_to(r0_value, [$(2048)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(16))
    r1_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r1_lane in range($(128)) {
      r1_value = r1_value + tl.load($(vto_copy_ptr) + ((r1_z * $(128)) + r1_lane), dtype=tl.int64)
    }
    r1_value = tl.broadcast_to(r1_value, [$(16)])
    tl.store($(vsum_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(16))
    r2_value = -tl.load($(vsum_1_ptr) + r2_z, dtype=tl.int64)
    r2_value = tl.broadcast_to(r2_value, [$(16)])
    tl.store($(vneg_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(16))
    r3_value = (tl.load($(vneg_ptr) + r3_z, dtype=tl.int64) % tl.full([], $(16), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(16)])
    tl.store($(vremainder_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(2048))
    r4_value = (tl.load($(vto_copy_ptr) + ((((r4_z / $(128)) % $(16)) * $(128)) + (r4_z % $(128))), dtype=tl.int64) == tl.full([], $(0), dtype=tl.int64))
    r4_value = tl.broadcast_to(r4_value, [$(2048)])
    tl.store($(veq_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2048))
    r5_value = tl.where(tl.load($(veq_ptr) + ((((r5_z / $(128)) % $(16)) * $(128)) + (r5_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(2048)])
    tl.store($(vto_copy_1_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2048))
    r6_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(2048)])
    for r6_lane in range($(128)) {
      r6_value = tl.where(r6_lane <= (r6_z % $(128)), (r6_value + tl.load($(vto_copy_1_ptr) + ((((r6_z / $(128)) % $(16)) * $(128)) + (r6_lane % $(128))), dtype=tl.int64)), r6_value)
    }
    r6_value = tl.broadcast_to(r6_value, [$(2048)])
    tl.store($(vcumsum_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(2048))
    r7_value = (tl.load($(vcumsum_ptr) + ((((r7_z / $(128)) % $(16)) * $(128)) + (r7_z % $(128))), dtype=tl.int64) <= tl.load($(vremainder_ptr) + ((r7_z / $(128)) % $(16)), dtype=tl.int64))
    r7_value = tl.broadcast_to(r7_value, [$(2048)])
    tl.store($(vle_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(2048))
    r8_value = (tl.load($(veq_ptr) + ((((r8_z / $(128)) % $(16)) * $(128)) + (r8_z % $(128))), dtype=tl.int1) & tl.load($(vle_ptr) + ((((r8_z / $(128)) % $(16)) * $(128)) + (r8_z % $(128))), dtype=tl.int1))
    r8_value = tl.broadcast_to(r8_value, [$(2048)])
    tl.store($(vbitwise_and_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(2048))
    r9_value = tl.where(tl.load($(vbitwise_and_ptr) + ((((r9_z / $(128)) % $(16)) * $(128)) + (r9_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.load($(vto_copy_ptr) + ((((r9_z / $(128)) % $(16)) * $(128)) + (r9_z % $(128))), dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(2048)])
    tl.store($(vwhere_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(2048))
    r10_value = tl.toReal(tl.load($(vwhere_ptr) + (((r10_z % $(16)) * $(128)) + ((r10_z / $(16)) % $(128))), dtype=tl.int64))
    r10_value = r10_value.to(tl.bfloat16)
    r10_value = tl.broadcast_to(r10_value, [$(2048)])
    tl.store($(vto_copy_2_ptr) + r10_z, r10_value)
    z = tl.arange($(2048))
    value = tl.load($(vto_copy_2_ptr) + ((((z / $(16)) % $(128)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2048)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vremainder_ptr : RegionName := "vremainder"
  let veq_ptr : RegionName := "veq"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vcumsum_ptr : RegionName := "vcumsum"
  let vle_ptr : RegionName := "vle"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vwhere_ptr : RegionName := "vwhere"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vremainder_ptr : RegionName := "candidate_vremainder"
  let candidate_vne_ptr : RegionName := "candidate_vne"
  let candidate_vrsub_ptr : RegionName := "candidate_vrsub"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_veq_ptr : RegionName := "candidate_veq"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vcumsum_ptr : RegionName := "candidate_vcumsum"
  let candidate_vle_ptr : RegionName := "candidate_vle"
  let candidate_vbitwise_and_ptr : RegionName := "candidate_vbitwise_and"
  let candidate_vwhere_1_ptr : RegionName := "candidate_vwhere_1"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  triton {
    r0_z = tl.arange($(2048))
    r0_value = tl.where(tl.load($(input0_ptr) + (((r0_z % $(128)) * $(16)) + ((r0_z / $(128)) % $(16))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r0_value = tl.broadcast_to(r0_value, [$(2048)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(16))
    r1_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r1_lane in range($(128)) {
      r1_value = r1_value + tl.load($(candidate_vto_copy_ptr) + ((r1_z * $(128)) + r1_lane), dtype=tl.int64)
    }
    r1_value = tl.broadcast_to(r1_value, [$(16)])
    tl.store($(candidate_vsum_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(16))
    r2_value = (tl.load($(candidate_vsum_1_ptr) + r2_z, dtype=tl.int64) % tl.full([], $(16), dtype=tl.int64))
    r2_value = tl.broadcast_to(r2_value, [$(16)])
    tl.store($(candidate_vremainder_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(16))
    r3_value = (tl.load($(candidate_vremainder_ptr) + r3_z, dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(16)])
    tl.store($(candidate_vne_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(16))
    r4_value = (tl.full([], $(16), dtype=tl.int64) - tl.load($(candidate_vremainder_ptr) + r4_z, dtype=tl.int64))
    r4_value = tl.broadcast_to(r4_value, [$(16)])
    tl.store($(candidate_vrsub_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(16))
    r5_value = tl.where(tl.load($(candidate_vne_ptr) + r5_z, dtype=tl.int1), tl.load($(candidate_vrsub_ptr) + r5_z, dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(16)])
    tl.store($(candidate_vwhere_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2048))
    r6_value = (tl.load($(candidate_vto_copy_ptr) + ((((r6_z / $(128)) % $(16)) * $(128)) + (r6_z % $(128))), dtype=tl.int64) == tl.full([], $(0), dtype=tl.int64))
    r6_value = tl.broadcast_to(r6_value, [$(2048)])
    tl.store($(candidate_veq_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(2048))
    r7_value = tl.where(tl.load($(candidate_veq_ptr) + ((((r7_z / $(128)) % $(16)) * $(128)) + (r7_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r7_value = tl.broadcast_to(r7_value, [$(2048)])
    tl.store($(candidate_vto_copy_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(2048))
    r8_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(2048)])
    for r8_lane in range($(128)) {
      r8_value = tl.where(r8_lane <= (r8_z % $(128)), (r8_value + tl.load($(candidate_vto_copy_1_ptr) + ((((r8_z / $(128)) % $(16)) * $(128)) + (r8_lane % $(128))), dtype=tl.int64)), r8_value)
    }
    r8_value = tl.broadcast_to(r8_value, [$(2048)])
    tl.store($(candidate_vcumsum_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(2048))
    r9_value = (tl.load($(candidate_vcumsum_ptr) + ((((r9_z / $(128)) % $(16)) * $(128)) + (r9_z % $(128))), dtype=tl.int64) <= tl.load($(candidate_vwhere_ptr) + ((r9_z / $(128)) % $(16)), dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(2048)])
    tl.store($(candidate_vle_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(2048))
    r10_value = (tl.load($(candidate_veq_ptr) + ((((r10_z / $(128)) % $(16)) * $(128)) + (r10_z % $(128))), dtype=tl.int1) & tl.load($(candidate_vle_ptr) + ((((r10_z / $(128)) % $(16)) * $(128)) + (r10_z % $(128))), dtype=tl.int1))
    r10_value = tl.broadcast_to(r10_value, [$(2048)])
    tl.store($(candidate_vbitwise_and_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(2048))
    r11_value = tl.where(tl.load($(candidate_vbitwise_and_ptr) + ((((r11_z / $(128)) % $(16)) * $(128)) + (r11_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.load($(candidate_vto_copy_ptr) + ((((r11_z / $(128)) % $(16)) * $(128)) + (r11_z % $(128))), dtype=tl.int64))
    r11_value = tl.broadcast_to(r11_value, [$(2048)])
    tl.store($(candidate_vwhere_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(2048))
    r12_value = tl.toReal(tl.load($(candidate_vwhere_1_ptr) + (((r12_z % $(16)) * $(128)) + ((r12_z / $(16)) % $(128))), dtype=tl.int64))
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(2048)])
    tl.store($(candidate_vto_copy_3_ptr) + r12_z, r12_value)
    z = tl.arange($(2048))
    value = tl.load($(candidate_vto_copy_3_ptr) + ((((z / $(16)) % $(128)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2048)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_pad_routing_map
