import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_adamw_multi_foreach_vs_loop
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vmul_ptr : RegionName := "vmul"
  let vlerp_ptr : RegionName := "vlerp"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vaddcmul_ptr : RegionName := "vaddcmul"
  let vsqrt_ptr : RegionName := "vsqrt"
  let vdiv_ptr : RegionName := "vdiv"
  let vadd_ptr : RegionName := "vadd"
  let vaddcdiv_ptr : RegionName := "vaddcdiv"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vlerp_1_ptr : RegionName := "vlerp_1"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vaddcmul_1_ptr : RegionName := "vaddcmul_1"
  let vsqrt_1_ptr : RegionName := "vsqrt_1"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vaddcdiv_1_ptr : RegionName := "vaddcdiv_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vlerp_2_ptr : RegionName := "vlerp_2"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vaddcmul_2_ptr : RegionName := "vaddcmul_2"
  let vsqrt_2_ptr : RegionName := "vsqrt_2"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vaddcdiv_2_ptr : RegionName := "vaddcdiv_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vlerp_3_ptr : RegionName := "vlerp_3"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vaddcmul_3_ptr : RegionName := "vaddcmul_3"
  let vsqrt_3_ptr : RegionName := "vsqrt_3"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vaddcdiv_3_ptr : RegionName := "vaddcdiv_3"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vlerp_4_ptr : RegionName := "vlerp_4"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vaddcmul_4_ptr : RegionName := "vaddcmul_4"
  let vsqrt_4_ptr : RegionName := "vsqrt_4"
  let vdiv_4_ptr : RegionName := "vdiv_4"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vaddcdiv_4_ptr : RegionName := "vaddcdiv_4"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vlerp_5_ptr : RegionName := "vlerp_5"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vaddcmul_5_ptr : RegionName := "vaddcmul_5"
  let vsqrt_5_ptr : RegionName := "vsqrt_5"
  let vdiv_5_ptr : RegionName := "vdiv_5"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vaddcdiv_5_ptr : RegionName := "vaddcdiv_5"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vlerp_6_ptr : RegionName := "vlerp_6"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vaddcmul_6_ptr : RegionName := "vaddcmul_6"
  let vsqrt_6_ptr : RegionName := "vsqrt_6"
  let vdiv_6_ptr : RegionName := "vdiv_6"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vaddcdiv_6_ptr : RegionName := "vaddcdiv_6"
  let vto_copy_7_ptr : RegionName := "vto_copy_7"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vlerp_7_ptr : RegionName := "vlerp_7"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vaddcmul_7_ptr : RegionName := "vaddcmul_7"
  let vsqrt_7_ptr : RegionName := "vsqrt_7"
  let vdiv_7_ptr : RegionName := "vdiv_7"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vaddcdiv_7_ptr : RegionName := "vaddcdiv_7"
  let vto_copy_8_ptr : RegionName := "vto_copy_8"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vlerp_8_ptr : RegionName := "vlerp_8"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vaddcmul_8_ptr : RegionName := "vaddcmul_8"
  let vsqrt_8_ptr : RegionName := "vsqrt_8"
  let vdiv_8_ptr : RegionName := "vdiv_8"
  let vadd_8_ptr : RegionName := "vadd_8"
  let vaddcdiv_8_ptr : RegionName := "vaddcdiv_8"
  let vto_copy_9_ptr : RegionName := "vto_copy_9"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vlerp_9_ptr : RegionName := "vlerp_9"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vaddcmul_9_ptr : RegionName := "vaddcmul_9"
  let vsqrt_9_ptr : RegionName := "vsqrt_9"
  let vdiv_9_ptr : RegionName := "vdiv_9"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vaddcdiv_9_ptr : RegionName := "vaddcdiv_9"
  let vto_copy_10_ptr : RegionName := "vto_copy_10"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vlerp_10_ptr : RegionName := "vlerp_10"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vaddcmul_10_ptr : RegionName := "vaddcmul_10"
  let vsqrt_10_ptr : RegionName := "vsqrt_10"
  let vdiv_10_ptr : RegionName := "vdiv_10"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vaddcdiv_10_ptr : RegionName := "vaddcdiv_10"
  let vto_copy_11_ptr : RegionName := "vto_copy_11"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vlerp_11_ptr : RegionName := "vlerp_11"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vaddcmul_11_ptr : RegionName := "vaddcmul_11"
  let vsqrt_11_ptr : RegionName := "vsqrt_11"
  let vdiv_11_ptr : RegionName := "vdiv_11"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vaddcdiv_11_ptr : RegionName := "vaddcdiv_11"
  let vto_copy_12_ptr : RegionName := "vto_copy_12"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vlerp_12_ptr : RegionName := "vlerp_12"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vaddcmul_12_ptr : RegionName := "vaddcmul_12"
  let vsqrt_12_ptr : RegionName := "vsqrt_12"
  let vdiv_12_ptr : RegionName := "vdiv_12"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vaddcdiv_12_ptr : RegionName := "vaddcdiv_12"
  let vto_copy_13_ptr : RegionName := "vto_copy_13"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vlerp_13_ptr : RegionName := "vlerp_13"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vaddcmul_13_ptr : RegionName := "vaddcmul_13"
  let vsqrt_13_ptr : RegionName := "vsqrt_13"
  let vdiv_13_ptr : RegionName := "vdiv_13"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vaddcdiv_13_ptr : RegionName := "vaddcdiv_13"
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vlerp_14_ptr : RegionName := "vlerp_14"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vaddcmul_14_ptr : RegionName := "vaddcmul_14"
  let vsqrt_14_ptr : RegionName := "vsqrt_14"
  let vdiv_14_ptr : RegionName := "vdiv_14"
  let vadd_14_ptr : RegionName := "vadd_14"
  let vaddcdiv_14_ptr : RegionName := "vaddcdiv_14"
  let vto_copy_15_ptr : RegionName := "vto_copy_15"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vlerp_15_ptr : RegionName := "vlerp_15"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vaddcmul_15_ptr : RegionName := "vaddcmul_15"
  let vsqrt_15_ptr : RegionName := "vsqrt_15"
  let vdiv_15_ptr : RegionName := "vdiv_15"
  let vadd_15_ptr : RegionName := "vadd_15"
  let vaddcdiv_15_ptr : RegionName := "vaddcdiv_15"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vlerp_16_ptr : RegionName := "vlerp_16"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vaddcmul_16_ptr : RegionName := "vaddcmul_16"
  let vsqrt_16_ptr : RegionName := "vsqrt_16"
  let vdiv_16_ptr : RegionName := "vdiv_16"
  let vadd_16_ptr : RegionName := "vadd_16"
  let vaddcdiv_16_ptr : RegionName := "vaddcdiv_16"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vlerp_17_ptr : RegionName := "vlerp_17"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vaddcmul_17_ptr : RegionName := "vaddcmul_17"
  let vsqrt_17_ptr : RegionName := "vsqrt_17"
  let vdiv_17_ptr : RegionName := "vdiv_17"
  let vadd_17_ptr : RegionName := "vadd_17"
  let vaddcdiv_17_ptr : RegionName := "vaddcdiv_17"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vlerp_18_ptr : RegionName := "vlerp_18"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vaddcmul_18_ptr : RegionName := "vaddcmul_18"
  let vsqrt_18_ptr : RegionName := "vsqrt_18"
  let vdiv_18_ptr : RegionName := "vdiv_18"
  let vadd_18_ptr : RegionName := "vadd_18"
  let vaddcdiv_18_ptr : RegionName := "vaddcdiv_18"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vlerp_19_ptr : RegionName := "vlerp_19"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vaddcmul_19_ptr : RegionName := "vaddcmul_19"
  let vsqrt_19_ptr : RegionName := "vsqrt_19"
  let vdiv_19_ptr : RegionName := "vdiv_19"
  let vadd_19_ptr : RegionName := "vadd_19"
  let vaddcdiv_19_ptr : RegionName := "vaddcdiv_19"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vlerp_20_ptr : RegionName := "vlerp_20"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vaddcmul_20_ptr : RegionName := "vaddcmul_20"
  let vsqrt_20_ptr : RegionName := "vsqrt_20"
  let vdiv_20_ptr : RegionName := "vdiv_20"
  let vadd_20_ptr : RegionName := "vadd_20"
  let vaddcdiv_20_ptr : RegionName := "vaddcdiv_20"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vlerp_21_ptr : RegionName := "vlerp_21"
  let vmul_43_ptr : RegionName := "vmul_43"
  let vaddcmul_21_ptr : RegionName := "vaddcmul_21"
  let vsqrt_21_ptr : RegionName := "vsqrt_21"
  let vdiv_21_ptr : RegionName := "vdiv_21"
  let vadd_21_ptr : RegionName := "vadd_21"
  let vaddcdiv_21_ptr : RegionName := "vaddcdiv_21"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vmul_44_ptr : RegionName := "vmul_44"
  let vlerp_22_ptr : RegionName := "vlerp_22"
  let vmul_45_ptr : RegionName := "vmul_45"
  let vaddcmul_22_ptr : RegionName := "vaddcmul_22"
  let vsqrt_22_ptr : RegionName := "vsqrt_22"
  let vdiv_22_ptr : RegionName := "vdiv_22"
  let vadd_22_ptr : RegionName := "vadd_22"
  let vaddcdiv_22_ptr : RegionName := "vaddcdiv_22"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmul_46_ptr : RegionName := "vmul_46"
  let vlerp_23_ptr : RegionName := "vlerp_23"
  let vmul_47_ptr : RegionName := "vmul_47"
  let vaddcmul_23_ptr : RegionName := "vaddcmul_23"
  let vsqrt_23_ptr : RegionName := "vsqrt_23"
  let vdiv_23_ptr : RegionName := "vdiv_23"
  let vadd_23_ptr : RegionName := "vadd_23"
  let vaddcdiv_23_ptr : RegionName := "vaddcdiv_23"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vmul_48_ptr : RegionName := "vmul_48"
  let vlerp_24_ptr : RegionName := "vlerp_24"
  let vmul_49_ptr : RegionName := "vmul_49"
  let vaddcmul_24_ptr : RegionName := "vaddcmul_24"
  let vsqrt_24_ptr : RegionName := "vsqrt_24"
  let vdiv_24_ptr : RegionName := "vdiv_24"
  let vadd_24_ptr : RegionName := "vadd_24"
  let vaddcdiv_24_ptr : RegionName := "vaddcdiv_24"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vmul_50_ptr : RegionName := "vmul_50"
  let vlerp_25_ptr : RegionName := "vlerp_25"
  let vmul_51_ptr : RegionName := "vmul_51"
  let vaddcmul_25_ptr : RegionName := "vaddcmul_25"
  let vsqrt_25_ptr : RegionName := "vsqrt_25"
  let vdiv_25_ptr : RegionName := "vdiv_25"
  let vadd_25_ptr : RegionName := "vadd_25"
  let vaddcdiv_25_ptr : RegionName := "vaddcdiv_25"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmul_52_ptr : RegionName := "vmul_52"
  let vlerp_26_ptr : RegionName := "vlerp_26"
  let vmul_53_ptr : RegionName := "vmul_53"
  let vaddcmul_26_ptr : RegionName := "vaddcmul_26"
  let vsqrt_26_ptr : RegionName := "vsqrt_26"
  let vdiv_26_ptr : RegionName := "vdiv_26"
  let vadd_26_ptr : RegionName := "vadd_26"
  let vaddcdiv_26_ptr : RegionName := "vaddcdiv_26"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vmul_54_ptr : RegionName := "vmul_54"
  let vlerp_27_ptr : RegionName := "vlerp_27"
  let vmul_55_ptr : RegionName := "vmul_55"
  let vaddcmul_27_ptr : RegionName := "vaddcmul_27"
  let vsqrt_27_ptr : RegionName := "vsqrt_27"
  let vdiv_27_ptr : RegionName := "vdiv_27"
  let vadd_27_ptr : RegionName := "vadd_27"
  let vaddcdiv_27_ptr : RegionName := "vaddcdiv_27"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_56_ptr : RegionName := "vmul_56"
  let vlerp_28_ptr : RegionName := "vlerp_28"
  let vmul_57_ptr : RegionName := "vmul_57"
  let vaddcmul_28_ptr : RegionName := "vaddcmul_28"
  let vsqrt_28_ptr : RegionName := "vsqrt_28"
  let vdiv_28_ptr : RegionName := "vdiv_28"
  let vadd_28_ptr : RegionName := "vadd_28"
  let vaddcdiv_28_ptr : RegionName := "vaddcdiv_28"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vmul_58_ptr : RegionName := "vmul_58"
  let vlerp_29_ptr : RegionName := "vlerp_29"
  let vmul_59_ptr : RegionName := "vmul_59"
  let vaddcmul_29_ptr : RegionName := "vaddcmul_29"
  let vsqrt_29_ptr : RegionName := "vsqrt_29"
  let vdiv_29_ptr : RegionName := "vdiv_29"
  let vadd_29_ptr : RegionName := "vadd_29"
  let vaddcdiv_29_ptr : RegionName := "vaddcdiv_29"
  let vto_copy_30_ptr : RegionName := "vto_copy_30"
  let vmul_60_ptr : RegionName := "vmul_60"
  let vlerp_30_ptr : RegionName := "vlerp_30"
  let vmul_61_ptr : RegionName := "vmul_61"
  let vaddcmul_30_ptr : RegionName := "vaddcmul_30"
  let vsqrt_30_ptr : RegionName := "vsqrt_30"
  let vdiv_30_ptr : RegionName := "vdiv_30"
  let vadd_30_ptr : RegionName := "vadd_30"
  let vaddcdiv_30_ptr : RegionName := "vaddcdiv_30"
  let vto_copy_31_ptr : RegionName := "vto_copy_31"
  let vmul_62_ptr : RegionName := "vmul_62"
  let vlerp_31_ptr : RegionName := "vlerp_31"
  let vmul_63_ptr : RegionName := "vmul_63"
  let vaddcmul_31_ptr : RegionName := "vaddcmul_31"
  let vsqrt_31_ptr : RegionName := "vsqrt_31"
  let vdiv_31_ptr : RegionName := "vdiv_31"
  let vadd_31_ptr : RegionName := "vadd_31"
  let vaddcdiv_31_ptr : RegionName := "vaddcdiv_31"
  let vstack_ptr : RegionName := "vstack"
  triton {
    r0_z = tl.arange($(512))
    r0_value = tl.load($(input0_ptr) + r0_z, dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(512)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(512))
    r1_value = (tl.load($(parameter0_ptr) + r1_z, dtype=tl.float32) * 0.99999)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(512)])
    tl.store($(vmul_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = (0.0 + (tl.load($(vto_copy_ptr) + r2_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(vlerp_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(512))
    r3_value = (0.0 * 0.999)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(512)])
    tl.store($(vmul_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(512))
    r4_value = (tl.load($(vmul_1_ptr) + r4_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_ptr) + r4_z, dtype=tl.float32) * tl.load($(vto_copy_ptr) + r4_z, dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(512)])
    tl.store($(vaddcmul_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = tl.sqrt(tl.load($(vaddcmul_ptr) + r5_z, dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(vsqrt_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(512))
    r6_value = (tl.load($(vsqrt_ptr) + r6_z, dtype=tl.float32) / 0.031622776601683805)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(512)])
    tl.store($(vdiv_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(512))
    r7_value = (tl.load($(vdiv_ptr) + r7_z, dtype=tl.float32) + 0.00000001)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(512)])
    tl.store($(vadd_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = (tl.load($(vmul_ptr) + r8_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_ptr) + r8_z, dtype=tl.float32) / tl.load($(vadd_ptr) + r8_z, dtype=tl.float32)))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(vaddcdiv_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(512))
    r9_value = tl.load($(input0_ptr) + (r9_z + $(512)), dtype=tl.bfloat16)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(512)])
    tl.store($(vto_copy_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_value = (tl.load($(parameter0_ptr) + (r10_z + $(512)), dtype=tl.float32) * 0.99999)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(vmul_2_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(512))
    r11_value = (0.0 + (tl.load($(vto_copy_1_ptr) + r11_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(512)])
    tl.store($(vlerp_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_value = (0.0 * 0.999)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(vmul_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = (tl.load($(vmul_3_ptr) + r13_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_1_ptr) + r13_z, dtype=tl.float32) * tl.load($(vto_copy_1_ptr) + r13_z, dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(vaddcmul_1_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(512))
    r14_value = tl.sqrt(tl.load($(vaddcmul_1_ptr) + r14_z, dtype=tl.float32))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(512)])
    tl.store($(vsqrt_1_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(512))
    r15_value = (tl.load($(vsqrt_1_ptr) + r15_z, dtype=tl.float32) / 0.031622776601683805)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(vdiv_1_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(512))
    r16_value = (tl.load($(vdiv_1_ptr) + r16_z, dtype=tl.float32) + 0.00000001)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(512)])
    tl.store($(vadd_1_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(512))
    r17_value = (tl.load($(vmul_2_ptr) + r17_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_1_ptr) + r17_z, dtype=tl.float32) / tl.load($(vadd_1_ptr) + r17_z, dtype=tl.float32)))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(512)])
    tl.store($(vaddcdiv_1_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(512))
    r18_value = tl.load($(input0_ptr) + (r18_z + $(1024)), dtype=tl.bfloat16)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(512)])
    tl.store($(vto_copy_2_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(512))
    r19_value = (tl.load($(parameter0_ptr) + (r19_z + $(1024)), dtype=tl.float32) * 0.99999)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(512)])
    tl.store($(vmul_4_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(512))
    r20_value = (0.0 + (tl.load($(vto_copy_2_ptr) + r20_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(512)])
    tl.store($(vlerp_2_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(512))
    r21_value = (0.0 * 0.999)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(512)])
    tl.store($(vmul_5_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(512))
    r22_value = (tl.load($(vmul_5_ptr) + r22_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_2_ptr) + r22_z, dtype=tl.float32) * tl.load($(vto_copy_2_ptr) + r22_z, dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(512)])
    tl.store($(vaddcmul_2_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(512))
    r23_value = tl.sqrt(tl.load($(vaddcmul_2_ptr) + r23_z, dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(512)])
    tl.store($(vsqrt_2_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(512))
    r24_value = (tl.load($(vsqrt_2_ptr) + r24_z, dtype=tl.float32) / 0.031622776601683805)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(512)])
    tl.store($(vdiv_2_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(512))
    r25_value = (tl.load($(vdiv_2_ptr) + r25_z, dtype=tl.float32) + 0.00000001)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(512)])
    tl.store($(vadd_2_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(512))
    r26_value = (tl.load($(vmul_4_ptr) + r26_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_2_ptr) + r26_z, dtype=tl.float32) / tl.load($(vadd_2_ptr) + r26_z, dtype=tl.float32)))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(512)])
    tl.store($(vaddcdiv_2_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(512))
    r27_value = tl.load($(input0_ptr) + (r27_z + $(1536)), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(512)])
    tl.store($(vto_copy_3_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(512))
    r28_value = (tl.load($(parameter0_ptr) + (r28_z + $(1536)), dtype=tl.float32) * 0.99999)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(512)])
    tl.store($(vmul_6_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(512))
    r29_value = (0.0 + (tl.load($(vto_copy_3_ptr) + r29_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(512)])
    tl.store($(vlerp_3_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(512))
    r30_value = (0.0 * 0.999)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(512)])
    tl.store($(vmul_7_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(512))
    r31_value = (tl.load($(vmul_7_ptr) + r31_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_3_ptr) + r31_z, dtype=tl.float32) * tl.load($(vto_copy_3_ptr) + r31_z, dtype=tl.float32))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(512)])
    tl.store($(vaddcmul_3_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(512))
    r32_value = tl.sqrt(tl.load($(vaddcmul_3_ptr) + r32_z, dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(512)])
    tl.store($(vsqrt_3_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(512))
    r33_value = (tl.load($(vsqrt_3_ptr) + r33_z, dtype=tl.float32) / 0.031622776601683805)
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(512)])
    tl.store($(vdiv_3_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(512))
    r34_value = (tl.load($(vdiv_3_ptr) + r34_z, dtype=tl.float32) + 0.00000001)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(512)])
    tl.store($(vadd_3_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(512))
    r35_value = (tl.load($(vmul_6_ptr) + r35_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_3_ptr) + r35_z, dtype=tl.float32) / tl.load($(vadd_3_ptr) + r35_z, dtype=tl.float32)))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(512)])
    tl.store($(vaddcdiv_3_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(512))
    r36_value = tl.load($(input0_ptr) + (r36_z + $(2048)), dtype=tl.bfloat16)
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(512)])
    tl.store($(vto_copy_4_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(512))
    r37_value = (tl.load($(parameter0_ptr) + (r37_z + $(2048)), dtype=tl.float32) * 0.99999)
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(512)])
    tl.store($(vmul_8_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(512))
    r38_value = (0.0 + (tl.load($(vto_copy_4_ptr) + r38_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(512)])
    tl.store($(vlerp_4_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(512))
    r39_value = (0.0 * 0.999)
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(512)])
    tl.store($(vmul_9_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(512))
    r40_value = (tl.load($(vmul_9_ptr) + r40_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_4_ptr) + r40_z, dtype=tl.float32) * tl.load($(vto_copy_4_ptr) + r40_z, dtype=tl.float32))
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(512)])
    tl.store($(vaddcmul_4_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(512))
    r41_value = tl.sqrt(tl.load($(vaddcmul_4_ptr) + r41_z, dtype=tl.float32))
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(512)])
    tl.store($(vsqrt_4_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(512))
    r42_value = (tl.load($(vsqrt_4_ptr) + r42_z, dtype=tl.float32) / 0.031622776601683805)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(512)])
    tl.store($(vdiv_4_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(512))
    r43_value = (tl.load($(vdiv_4_ptr) + r43_z, dtype=tl.float32) + 0.00000001)
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(512)])
    tl.store($(vadd_4_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(512))
    r44_value = (tl.load($(vmul_8_ptr) + r44_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_4_ptr) + r44_z, dtype=tl.float32) / tl.load($(vadd_4_ptr) + r44_z, dtype=tl.float32)))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(512)])
    tl.store($(vaddcdiv_4_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(512))
    r45_value = tl.load($(input0_ptr) + (r45_z + $(2560)), dtype=tl.bfloat16)
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(512)])
    tl.store($(vto_copy_5_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(512))
    r46_value = (tl.load($(parameter0_ptr) + (r46_z + $(2560)), dtype=tl.float32) * 0.99999)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(512)])
    tl.store($(vmul_10_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(512))
    r47_value = (0.0 + (tl.load($(vto_copy_5_ptr) + r47_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(512)])
    tl.store($(vlerp_5_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(512))
    r48_value = (0.0 * 0.999)
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(512)])
    tl.store($(vmul_11_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(512))
    r49_value = (tl.load($(vmul_11_ptr) + r49_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_5_ptr) + r49_z, dtype=tl.float32) * tl.load($(vto_copy_5_ptr) + r49_z, dtype=tl.float32))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(512)])
    tl.store($(vaddcmul_5_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(512))
    r50_value = tl.sqrt(tl.load($(vaddcmul_5_ptr) + r50_z, dtype=tl.float32))
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(512)])
    tl.store($(vsqrt_5_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(512))
    r51_value = (tl.load($(vsqrt_5_ptr) + r51_z, dtype=tl.float32) / 0.031622776601683805)
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(512)])
    tl.store($(vdiv_5_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(512))
    r52_value = (tl.load($(vdiv_5_ptr) + r52_z, dtype=tl.float32) + 0.00000001)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(512)])
    tl.store($(vadd_5_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(512))
    r53_value = (tl.load($(vmul_10_ptr) + r53_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_5_ptr) + r53_z, dtype=tl.float32) / tl.load($(vadd_5_ptr) + r53_z, dtype=tl.float32)))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(512)])
    tl.store($(vaddcdiv_5_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(512))
    r54_value = tl.load($(input0_ptr) + (r54_z + $(3072)), dtype=tl.bfloat16)
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(512)])
    tl.store($(vto_copy_6_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(512))
    r55_value = (tl.load($(parameter0_ptr) + (r55_z + $(3072)), dtype=tl.float32) * 0.99999)
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(512)])
    tl.store($(vmul_12_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(512))
    r56_value = (0.0 + (tl.load($(vto_copy_6_ptr) + r56_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(512)])
    tl.store($(vlerp_6_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(512))
    r57_value = (0.0 * 0.999)
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(512)])
    tl.store($(vmul_13_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(512))
    r58_value = (tl.load($(vmul_13_ptr) + r58_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_6_ptr) + r58_z, dtype=tl.float32) * tl.load($(vto_copy_6_ptr) + r58_z, dtype=tl.float32))
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(512)])
    tl.store($(vaddcmul_6_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(512))
    r59_value = tl.sqrt(tl.load($(vaddcmul_6_ptr) + r59_z, dtype=tl.float32))
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(512)])
    tl.store($(vsqrt_6_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(512))
    r60_value = (tl.load($(vsqrt_6_ptr) + r60_z, dtype=tl.float32) / 0.031622776601683805)
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(512)])
    tl.store($(vdiv_6_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(512))
    r61_value = (tl.load($(vdiv_6_ptr) + r61_z, dtype=tl.float32) + 0.00000001)
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(512)])
    tl.store($(vadd_6_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(512))
    r62_value = (tl.load($(vmul_12_ptr) + r62_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_6_ptr) + r62_z, dtype=tl.float32) / tl.load($(vadd_6_ptr) + r62_z, dtype=tl.float32)))
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(512)])
    tl.store($(vaddcdiv_6_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(512))
    r63_value = tl.load($(input0_ptr) + (r63_z + $(3584)), dtype=tl.bfloat16)
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(512)])
    tl.store($(vto_copy_7_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(512))
    r64_value = (tl.load($(parameter0_ptr) + (r64_z + $(3584)), dtype=tl.float32) * 0.99999)
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(512)])
    tl.store($(vmul_14_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(512))
    r65_value = (0.0 + (tl.load($(vto_copy_7_ptr) + r65_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(512)])
    tl.store($(vlerp_7_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(512))
    r66_value = (0.0 * 0.999)
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(512)])
    tl.store($(vmul_15_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(512))
    r67_value = (tl.load($(vmul_15_ptr) + r67_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_7_ptr) + r67_z, dtype=tl.float32) * tl.load($(vto_copy_7_ptr) + r67_z, dtype=tl.float32))
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(512)])
    tl.store($(vaddcmul_7_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(512))
    r68_value = tl.sqrt(tl.load($(vaddcmul_7_ptr) + r68_z, dtype=tl.float32))
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(512)])
    tl.store($(vsqrt_7_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(512))
    r69_value = (tl.load($(vsqrt_7_ptr) + r69_z, dtype=tl.float32) / 0.031622776601683805)
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(512)])
    tl.store($(vdiv_7_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(512))
    r70_value = (tl.load($(vdiv_7_ptr) + r70_z, dtype=tl.float32) + 0.00000001)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(512)])
    tl.store($(vadd_7_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(512))
    r71_value = (tl.load($(vmul_14_ptr) + r71_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_7_ptr) + r71_z, dtype=tl.float32) / tl.load($(vadd_7_ptr) + r71_z, dtype=tl.float32)))
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(512)])
    tl.store($(vaddcdiv_7_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(512))
    r72_value = tl.load($(input0_ptr) + (r72_z + $(4096)), dtype=tl.bfloat16)
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(512)])
    tl.store($(vto_copy_8_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(512))
    r73_value = (tl.load($(parameter0_ptr) + (r73_z + $(4096)), dtype=tl.float32) * 0.99999)
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(512)])
    tl.store($(vmul_16_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(512))
    r74_value = (0.0 + (tl.load($(vto_copy_8_ptr) + r74_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(512)])
    tl.store($(vlerp_8_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(512))
    r75_value = (0.0 * 0.999)
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(512)])
    tl.store($(vmul_17_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(512))
    r76_value = (tl.load($(vmul_17_ptr) + r76_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_8_ptr) + r76_z, dtype=tl.float32) * tl.load($(vto_copy_8_ptr) + r76_z, dtype=tl.float32))
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(512)])
    tl.store($(vaddcmul_8_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(512))
    r77_value = tl.sqrt(tl.load($(vaddcmul_8_ptr) + r77_z, dtype=tl.float32))
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(512)])
    tl.store($(vsqrt_8_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(512))
    r78_value = (tl.load($(vsqrt_8_ptr) + r78_z, dtype=tl.float32) / 0.031622776601683805)
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(512)])
    tl.store($(vdiv_8_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(512))
    r79_value = (tl.load($(vdiv_8_ptr) + r79_z, dtype=tl.float32) + 0.00000001)
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(512)])
    tl.store($(vadd_8_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(512))
    r80_value = (tl.load($(vmul_16_ptr) + r80_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_8_ptr) + r80_z, dtype=tl.float32) / tl.load($(vadd_8_ptr) + r80_z, dtype=tl.float32)))
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(512)])
    tl.store($(vaddcdiv_8_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(512))
    r81_value = tl.load($(input0_ptr) + (r81_z + $(4608)), dtype=tl.bfloat16)
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(512)])
    tl.store($(vto_copy_9_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(512))
    r82_value = (tl.load($(parameter0_ptr) + (r82_z + $(4608)), dtype=tl.float32) * 0.99999)
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(512)])
    tl.store($(vmul_18_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(512))
    r83_value = (0.0 + (tl.load($(vto_copy_9_ptr) + r83_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(512)])
    tl.store($(vlerp_9_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(512))
    r84_value = (0.0 * 0.999)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(512)])
    tl.store($(vmul_19_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(512))
    r85_value = (tl.load($(vmul_19_ptr) + r85_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_9_ptr) + r85_z, dtype=tl.float32) * tl.load($(vto_copy_9_ptr) + r85_z, dtype=tl.float32))
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(512)])
    tl.store($(vaddcmul_9_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(512))
    r86_value = tl.sqrt(tl.load($(vaddcmul_9_ptr) + r86_z, dtype=tl.float32))
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(512)])
    tl.store($(vsqrt_9_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(512))
    r87_value = (tl.load($(vsqrt_9_ptr) + r87_z, dtype=tl.float32) / 0.031622776601683805)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(512)])
    tl.store($(vdiv_9_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(512))
    r88_value = (tl.load($(vdiv_9_ptr) + r88_z, dtype=tl.float32) + 0.00000001)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(512)])
    tl.store($(vadd_9_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(512))
    r89_value = (tl.load($(vmul_18_ptr) + r89_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_9_ptr) + r89_z, dtype=tl.float32) / tl.load($(vadd_9_ptr) + r89_z, dtype=tl.float32)))
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(512)])
    tl.store($(vaddcdiv_9_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(512))
    r90_value = tl.load($(input0_ptr) + (r90_z + $(5120)), dtype=tl.bfloat16)
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(512)])
    tl.store($(vto_copy_10_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(512))
    r91_value = (tl.load($(parameter0_ptr) + (r91_z + $(5120)), dtype=tl.float32) * 0.99999)
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(512)])
    tl.store($(vmul_20_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(512))
    r92_value = (0.0 + (tl.load($(vto_copy_10_ptr) + r92_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(512)])
    tl.store($(vlerp_10_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(512))
    r93_value = (0.0 * 0.999)
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(512)])
    tl.store($(vmul_21_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(512))
    r94_value = (tl.load($(vmul_21_ptr) + r94_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_10_ptr) + r94_z, dtype=tl.float32) * tl.load($(vto_copy_10_ptr) + r94_z, dtype=tl.float32))
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(512)])
    tl.store($(vaddcmul_10_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(512))
    r95_value = tl.sqrt(tl.load($(vaddcmul_10_ptr) + r95_z, dtype=tl.float32))
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(512)])
    tl.store($(vsqrt_10_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(512))
    r96_value = (tl.load($(vsqrt_10_ptr) + r96_z, dtype=tl.float32) / 0.031622776601683805)
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(512)])
    tl.store($(vdiv_10_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(512))
    r97_value = (tl.load($(vdiv_10_ptr) + r97_z, dtype=tl.float32) + 0.00000001)
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(512)])
    tl.store($(vadd_10_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(512))
    r98_value = (tl.load($(vmul_20_ptr) + r98_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_10_ptr) + r98_z, dtype=tl.float32) / tl.load($(vadd_10_ptr) + r98_z, dtype=tl.float32)))
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(512)])
    tl.store($(vaddcdiv_10_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(512))
    r99_value = tl.load($(input0_ptr) + (r99_z + $(5632)), dtype=tl.bfloat16)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(512)])
    tl.store($(vto_copy_11_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(512))
    r100_value = (tl.load($(parameter0_ptr) + (r100_z + $(5632)), dtype=tl.float32) * 0.99999)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(512)])
    tl.store($(vmul_22_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(512))
    r101_value = (0.0 + (tl.load($(vto_copy_11_ptr) + r101_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(512)])
    tl.store($(vlerp_11_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(512))
    r102_value = (0.0 * 0.999)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(512)])
    tl.store($(vmul_23_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(512))
    r103_value = (tl.load($(vmul_23_ptr) + r103_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_11_ptr) + r103_z, dtype=tl.float32) * tl.load($(vto_copy_11_ptr) + r103_z, dtype=tl.float32))
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(512)])
    tl.store($(vaddcmul_11_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(512))
    r104_value = tl.sqrt(tl.load($(vaddcmul_11_ptr) + r104_z, dtype=tl.float32))
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(512)])
    tl.store($(vsqrt_11_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(512))
    r105_value = (tl.load($(vsqrt_11_ptr) + r105_z, dtype=tl.float32) / 0.031622776601683805)
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(512)])
    tl.store($(vdiv_11_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(512))
    r106_value = (tl.load($(vdiv_11_ptr) + r106_z, dtype=tl.float32) + 0.00000001)
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(512)])
    tl.store($(vadd_11_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(512))
    r107_value = (tl.load($(vmul_22_ptr) + r107_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_11_ptr) + r107_z, dtype=tl.float32) / tl.load($(vadd_11_ptr) + r107_z, dtype=tl.float32)))
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(512)])
    tl.store($(vaddcdiv_11_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(512))
    r108_value = tl.load($(input0_ptr) + (r108_z + $(6144)), dtype=tl.bfloat16)
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(512)])
    tl.store($(vto_copy_12_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(512))
    r109_value = (tl.load($(parameter0_ptr) + (r109_z + $(6144)), dtype=tl.float32) * 0.99999)
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(512)])
    tl.store($(vmul_24_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(512))
    r110_value = (0.0 + (tl.load($(vto_copy_12_ptr) + r110_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(512)])
    tl.store($(vlerp_12_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(512))
    r111_value = (0.0 * 0.999)
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(512)])
    tl.store($(vmul_25_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(512))
    r112_value = (tl.load($(vmul_25_ptr) + r112_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_12_ptr) + r112_z, dtype=tl.float32) * tl.load($(vto_copy_12_ptr) + r112_z, dtype=tl.float32))
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(512)])
    tl.store($(vaddcmul_12_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(512))
    r113_value = tl.sqrt(tl.load($(vaddcmul_12_ptr) + r113_z, dtype=tl.float32))
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(512)])
    tl.store($(vsqrt_12_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(512))
    r114_value = (tl.load($(vsqrt_12_ptr) + r114_z, dtype=tl.float32) / 0.031622776601683805)
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(512)])
    tl.store($(vdiv_12_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(512))
    r115_value = (tl.load($(vdiv_12_ptr) + r115_z, dtype=tl.float32) + 0.00000001)
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(512)])
    tl.store($(vadd_12_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(512))
    r116_value = (tl.load($(vmul_24_ptr) + r116_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_12_ptr) + r116_z, dtype=tl.float32) / tl.load($(vadd_12_ptr) + r116_z, dtype=tl.float32)))
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(512)])
    tl.store($(vaddcdiv_12_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(512))
    r117_value = tl.load($(input0_ptr) + (r117_z + $(6656)), dtype=tl.bfloat16)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(512)])
    tl.store($(vto_copy_13_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(512))
    r118_value = (tl.load($(parameter0_ptr) + (r118_z + $(6656)), dtype=tl.float32) * 0.99999)
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(512)])
    tl.store($(vmul_26_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(512))
    r119_value = (0.0 + (tl.load($(vto_copy_13_ptr) + r119_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(512)])
    tl.store($(vlerp_13_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(512))
    r120_value = (0.0 * 0.999)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(512)])
    tl.store($(vmul_27_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(512))
    r121_value = (tl.load($(vmul_27_ptr) + r121_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_13_ptr) + r121_z, dtype=tl.float32) * tl.load($(vto_copy_13_ptr) + r121_z, dtype=tl.float32))
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(512)])
    tl.store($(vaddcmul_13_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(512))
    r122_value = tl.sqrt(tl.load($(vaddcmul_13_ptr) + r122_z, dtype=tl.float32))
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(512)])
    tl.store($(vsqrt_13_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(512))
    r123_value = (tl.load($(vsqrt_13_ptr) + r123_z, dtype=tl.float32) / 0.031622776601683805)
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(512)])
    tl.store($(vdiv_13_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(512))
    r124_value = (tl.load($(vdiv_13_ptr) + r124_z, dtype=tl.float32) + 0.00000001)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(512)])
    tl.store($(vadd_13_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(512))
    r125_value = (tl.load($(vmul_26_ptr) + r125_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_13_ptr) + r125_z, dtype=tl.float32) / tl.load($(vadd_13_ptr) + r125_z, dtype=tl.float32)))
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(512)])
    tl.store($(vaddcdiv_13_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(512))
    r126_value = tl.load($(input0_ptr) + (r126_z + $(7168)), dtype=tl.bfloat16)
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(512)])
    tl.store($(vto_copy_14_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(512))
    r127_value = (tl.load($(parameter0_ptr) + (r127_z + $(7168)), dtype=tl.float32) * 0.99999)
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(512)])
    tl.store($(vmul_28_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(512))
    r128_value = (0.0 + (tl.load($(vto_copy_14_ptr) + r128_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(512)])
    tl.store($(vlerp_14_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(512))
    r129_value = (0.0 * 0.999)
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(512)])
    tl.store($(vmul_29_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(512))
    r130_value = (tl.load($(vmul_29_ptr) + r130_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_14_ptr) + r130_z, dtype=tl.float32) * tl.load($(vto_copy_14_ptr) + r130_z, dtype=tl.float32))
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(512)])
    tl.store($(vaddcmul_14_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(512))
    r131_value = tl.sqrt(tl.load($(vaddcmul_14_ptr) + r131_z, dtype=tl.float32))
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(512)])
    tl.store($(vsqrt_14_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(512))
    r132_value = (tl.load($(vsqrt_14_ptr) + r132_z, dtype=tl.float32) / 0.031622776601683805)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(512)])
    tl.store($(vdiv_14_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(512))
    r133_value = (tl.load($(vdiv_14_ptr) + r133_z, dtype=tl.float32) + 0.00000001)
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(512)])
    tl.store($(vadd_14_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(512))
    r134_value = (tl.load($(vmul_28_ptr) + r134_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_14_ptr) + r134_z, dtype=tl.float32) / tl.load($(vadd_14_ptr) + r134_z, dtype=tl.float32)))
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(512)])
    tl.store($(vaddcdiv_14_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(512))
    r135_value = tl.load($(input0_ptr) + (r135_z + $(7680)), dtype=tl.bfloat16)
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(512)])
    tl.store($(vto_copy_15_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(512))
    r136_value = (tl.load($(parameter0_ptr) + (r136_z + $(7680)), dtype=tl.float32) * 0.99999)
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(512)])
    tl.store($(vmul_30_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(512))
    r137_value = (0.0 + (tl.load($(vto_copy_15_ptr) + r137_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(512)])
    tl.store($(vlerp_15_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(512))
    r138_value = (0.0 * 0.999)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(512)])
    tl.store($(vmul_31_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(512))
    r139_value = (tl.load($(vmul_31_ptr) + r139_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_15_ptr) + r139_z, dtype=tl.float32) * tl.load($(vto_copy_15_ptr) + r139_z, dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(512)])
    tl.store($(vaddcmul_15_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(512))
    r140_value = tl.sqrt(tl.load($(vaddcmul_15_ptr) + r140_z, dtype=tl.float32))
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(512)])
    tl.store($(vsqrt_15_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(512))
    r141_value = (tl.load($(vsqrt_15_ptr) + r141_z, dtype=tl.float32) / 0.031622776601683805)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(512)])
    tl.store($(vdiv_15_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(512))
    r142_value = (tl.load($(vdiv_15_ptr) + r142_z, dtype=tl.float32) + 0.00000001)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(512)])
    tl.store($(vadd_15_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(512))
    r143_value = (tl.load($(vmul_30_ptr) + r143_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_15_ptr) + r143_z, dtype=tl.float32) / tl.load($(vadd_15_ptr) + r143_z, dtype=tl.float32)))
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(512)])
    tl.store($(vaddcdiv_15_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(512))
    r144_value = tl.load($(input0_ptr) + (r144_z + $(8192)), dtype=tl.bfloat16)
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(512)])
    tl.store($(vto_copy_16_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(512))
    r145_value = (tl.load($(parameter0_ptr) + (r145_z + $(8192)), dtype=tl.float32) * 0.99999)
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(512)])
    tl.store($(vmul_32_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(512))
    r146_value = (0.0 + (tl.load($(vto_copy_16_ptr) + r146_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r146_value = r146_value.to(tl.float32)
    r146_value = tl.broadcast_to(r146_value, [$(512)])
    tl.store($(vlerp_16_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(512))
    r147_value = (0.0 * 0.999)
    r147_value = r147_value.to(tl.float32)
    r147_value = tl.broadcast_to(r147_value, [$(512)])
    tl.store($(vmul_33_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(512))
    r148_value = (tl.load($(vmul_33_ptr) + r148_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_16_ptr) + r148_z, dtype=tl.float32) * tl.load($(vto_copy_16_ptr) + r148_z, dtype=tl.float32))
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(512)])
    tl.store($(vaddcmul_16_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(512))
    r149_value = tl.sqrt(tl.load($(vaddcmul_16_ptr) + r149_z, dtype=tl.float32))
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(512)])
    tl.store($(vsqrt_16_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(512))
    r150_value = (tl.load($(vsqrt_16_ptr) + r150_z, dtype=tl.float32) / 0.031622776601683805)
    r150_value = r150_value.to(tl.float32)
    r150_value = tl.broadcast_to(r150_value, [$(512)])
    tl.store($(vdiv_16_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(512))
    r151_value = (tl.load($(vdiv_16_ptr) + r151_z, dtype=tl.float32) + 0.00000001)
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(512)])
    tl.store($(vadd_16_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(512))
    r152_value = (tl.load($(vmul_32_ptr) + r152_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_16_ptr) + r152_z, dtype=tl.float32) / tl.load($(vadd_16_ptr) + r152_z, dtype=tl.float32)))
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(512)])
    tl.store($(vaddcdiv_16_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(512))
    r153_value = tl.load($(input0_ptr) + (r153_z + $(8704)), dtype=tl.bfloat16)
    r153_value = r153_value.to(tl.float32)
    r153_value = tl.broadcast_to(r153_value, [$(512)])
    tl.store($(vto_copy_17_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(512))
    r154_value = (tl.load($(parameter0_ptr) + (r154_z + $(8704)), dtype=tl.float32) * 0.99999)
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(512)])
    tl.store($(vmul_34_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(512))
    r155_value = (0.0 + (tl.load($(vto_copy_17_ptr) + r155_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(512)])
    tl.store($(vlerp_17_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(512))
    r156_value = (0.0 * 0.999)
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(512)])
    tl.store($(vmul_35_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(512))
    r157_value = (tl.load($(vmul_35_ptr) + r157_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_17_ptr) + r157_z, dtype=tl.float32) * tl.load($(vto_copy_17_ptr) + r157_z, dtype=tl.float32))
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(512)])
    tl.store($(vaddcmul_17_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(512))
    r158_value = tl.sqrt(tl.load($(vaddcmul_17_ptr) + r158_z, dtype=tl.float32))
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(512)])
    tl.store($(vsqrt_17_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(512))
    r159_value = (tl.load($(vsqrt_17_ptr) + r159_z, dtype=tl.float32) / 0.031622776601683805)
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(512)])
    tl.store($(vdiv_17_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(512))
    r160_value = (tl.load($(vdiv_17_ptr) + r160_z, dtype=tl.float32) + 0.00000001)
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(512)])
    tl.store($(vadd_17_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(512))
    r161_value = (tl.load($(vmul_34_ptr) + r161_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_17_ptr) + r161_z, dtype=tl.float32) / tl.load($(vadd_17_ptr) + r161_z, dtype=tl.float32)))
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(512)])
    tl.store($(vaddcdiv_17_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(512))
    r162_value = tl.load($(input0_ptr) + (r162_z + $(9216)), dtype=tl.bfloat16)
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(512)])
    tl.store($(vto_copy_18_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(512))
    r163_value = (tl.load($(parameter0_ptr) + (r163_z + $(9216)), dtype=tl.float32) * 0.99999)
    r163_value = r163_value.to(tl.float32)
    r163_value = tl.broadcast_to(r163_value, [$(512)])
    tl.store($(vmul_36_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(512))
    r164_value = (0.0 + (tl.load($(vto_copy_18_ptr) + r164_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(512)])
    tl.store($(vlerp_18_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(512))
    r165_value = (0.0 * 0.999)
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(512)])
    tl.store($(vmul_37_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(512))
    r166_value = (tl.load($(vmul_37_ptr) + r166_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_18_ptr) + r166_z, dtype=tl.float32) * tl.load($(vto_copy_18_ptr) + r166_z, dtype=tl.float32))
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(512)])
    tl.store($(vaddcmul_18_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(512))
    r167_value = tl.sqrt(tl.load($(vaddcmul_18_ptr) + r167_z, dtype=tl.float32))
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(512)])
    tl.store($(vsqrt_18_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(512))
    r168_value = (tl.load($(vsqrt_18_ptr) + r168_z, dtype=tl.float32) / 0.031622776601683805)
    r168_value = r168_value.to(tl.float32)
    r168_value = tl.broadcast_to(r168_value, [$(512)])
    tl.store($(vdiv_18_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(512))
    r169_value = (tl.load($(vdiv_18_ptr) + r169_z, dtype=tl.float32) + 0.00000001)
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(512)])
    tl.store($(vadd_18_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(512))
    r170_value = (tl.load($(vmul_36_ptr) + r170_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_18_ptr) + r170_z, dtype=tl.float32) / tl.load($(vadd_18_ptr) + r170_z, dtype=tl.float32)))
    r170_value = r170_value.to(tl.float32)
    r170_value = tl.broadcast_to(r170_value, [$(512)])
    tl.store($(vaddcdiv_18_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(512))
    r171_value = tl.load($(input0_ptr) + (r171_z + $(9728)), dtype=tl.bfloat16)
    r171_value = r171_value.to(tl.float32)
    r171_value = tl.broadcast_to(r171_value, [$(512)])
    tl.store($(vto_copy_19_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(512))
    r172_value = (tl.load($(parameter0_ptr) + (r172_z + $(9728)), dtype=tl.float32) * 0.99999)
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(512)])
    tl.store($(vmul_38_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(512))
    r173_value = (0.0 + (tl.load($(vto_copy_19_ptr) + r173_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r173_value = r173_value.to(tl.float32)
    r173_value = tl.broadcast_to(r173_value, [$(512)])
    tl.store($(vlerp_19_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(512))
    r174_value = (0.0 * 0.999)
    r174_value = r174_value.to(tl.float32)
    r174_value = tl.broadcast_to(r174_value, [$(512)])
    tl.store($(vmul_39_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(512))
    r175_value = (tl.load($(vmul_39_ptr) + r175_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_19_ptr) + r175_z, dtype=tl.float32) * tl.load($(vto_copy_19_ptr) + r175_z, dtype=tl.float32))
    r175_value = r175_value.to(tl.float32)
    r175_value = tl.broadcast_to(r175_value, [$(512)])
    tl.store($(vaddcmul_19_ptr) + r175_z, r175_value)
    r176_z = tl.arange($(512))
    r176_value = tl.sqrt(tl.load($(vaddcmul_19_ptr) + r176_z, dtype=tl.float32))
    r176_value = r176_value.to(tl.float32)
    r176_value = tl.broadcast_to(r176_value, [$(512)])
    tl.store($(vsqrt_19_ptr) + r176_z, r176_value)
    r177_z = tl.arange($(512))
    r177_value = (tl.load($(vsqrt_19_ptr) + r177_z, dtype=tl.float32) / 0.031622776601683805)
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(512)])
    tl.store($(vdiv_19_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(512))
    r178_value = (tl.load($(vdiv_19_ptr) + r178_z, dtype=tl.float32) + 0.00000001)
    r178_value = r178_value.to(tl.float32)
    r178_value = tl.broadcast_to(r178_value, [$(512)])
    tl.store($(vadd_19_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(512))
    r179_value = (tl.load($(vmul_38_ptr) + r179_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_19_ptr) + r179_z, dtype=tl.float32) / tl.load($(vadd_19_ptr) + r179_z, dtype=tl.float32)))
    r179_value = r179_value.to(tl.float32)
    r179_value = tl.broadcast_to(r179_value, [$(512)])
    tl.store($(vaddcdiv_19_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(512))
    r180_value = tl.load($(input0_ptr) + (r180_z + $(10240)), dtype=tl.bfloat16)
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(512)])
    tl.store($(vto_copy_20_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(512))
    r181_value = (tl.load($(parameter0_ptr) + (r181_z + $(10240)), dtype=tl.float32) * 0.99999)
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(512)])
    tl.store($(vmul_40_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(512))
    r182_value = (0.0 + (tl.load($(vto_copy_20_ptr) + r182_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(512)])
    tl.store($(vlerp_20_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(512))
    r183_value = (0.0 * 0.999)
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(512)])
    tl.store($(vmul_41_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(512))
    r184_value = (tl.load($(vmul_41_ptr) + r184_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_20_ptr) + r184_z, dtype=tl.float32) * tl.load($(vto_copy_20_ptr) + r184_z, dtype=tl.float32))
    r184_value = r184_value.to(tl.float32)
    r184_value = tl.broadcast_to(r184_value, [$(512)])
    tl.store($(vaddcmul_20_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(512))
    r185_value = tl.sqrt(tl.load($(vaddcmul_20_ptr) + r185_z, dtype=tl.float32))
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(512)])
    tl.store($(vsqrt_20_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(512))
    r186_value = (tl.load($(vsqrt_20_ptr) + r186_z, dtype=tl.float32) / 0.031622776601683805)
    r186_value = r186_value.to(tl.float32)
    r186_value = tl.broadcast_to(r186_value, [$(512)])
    tl.store($(vdiv_20_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(512))
    r187_value = (tl.load($(vdiv_20_ptr) + r187_z, dtype=tl.float32) + 0.00000001)
    r187_value = r187_value.to(tl.float32)
    r187_value = tl.broadcast_to(r187_value, [$(512)])
    tl.store($(vadd_20_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(512))
    r188_value = (tl.load($(vmul_40_ptr) + r188_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_20_ptr) + r188_z, dtype=tl.float32) / tl.load($(vadd_20_ptr) + r188_z, dtype=tl.float32)))
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(512)])
    tl.store($(vaddcdiv_20_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(512))
    r189_value = tl.load($(input0_ptr) + (r189_z + $(10752)), dtype=tl.bfloat16)
    r189_value = r189_value.to(tl.float32)
    r189_value = tl.broadcast_to(r189_value, [$(512)])
    tl.store($(vto_copy_21_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(512))
    r190_value = (tl.load($(parameter0_ptr) + (r190_z + $(10752)), dtype=tl.float32) * 0.99999)
    r190_value = r190_value.to(tl.float32)
    r190_value = tl.broadcast_to(r190_value, [$(512)])
    tl.store($(vmul_42_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(512))
    r191_value = (0.0 + (tl.load($(vto_copy_21_ptr) + r191_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r191_value = r191_value.to(tl.float32)
    r191_value = tl.broadcast_to(r191_value, [$(512)])
    tl.store($(vlerp_21_ptr) + r191_z, r191_value)
    r192_z = tl.arange($(512))
    r192_value = (0.0 * 0.999)
    r192_value = r192_value.to(tl.float32)
    r192_value = tl.broadcast_to(r192_value, [$(512)])
    tl.store($(vmul_43_ptr) + r192_z, r192_value)
    r193_z = tl.arange($(512))
    r193_value = (tl.load($(vmul_43_ptr) + r193_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_21_ptr) + r193_z, dtype=tl.float32) * tl.load($(vto_copy_21_ptr) + r193_z, dtype=tl.float32))
    r193_value = r193_value.to(tl.float32)
    r193_value = tl.broadcast_to(r193_value, [$(512)])
    tl.store($(vaddcmul_21_ptr) + r193_z, r193_value)
    r194_z = tl.arange($(512))
    r194_value = tl.sqrt(tl.load($(vaddcmul_21_ptr) + r194_z, dtype=tl.float32))
    r194_value = r194_value.to(tl.float32)
    r194_value = tl.broadcast_to(r194_value, [$(512)])
    tl.store($(vsqrt_21_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(512))
    r195_value = (tl.load($(vsqrt_21_ptr) + r195_z, dtype=tl.float32) / 0.031622776601683805)
    r195_value = r195_value.to(tl.float32)
    r195_value = tl.broadcast_to(r195_value, [$(512)])
    tl.store($(vdiv_21_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(512))
    r196_value = (tl.load($(vdiv_21_ptr) + r196_z, dtype=tl.float32) + 0.00000001)
    r196_value = r196_value.to(tl.float32)
    r196_value = tl.broadcast_to(r196_value, [$(512)])
    tl.store($(vadd_21_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(512))
    r197_value = (tl.load($(vmul_42_ptr) + r197_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_21_ptr) + r197_z, dtype=tl.float32) / tl.load($(vadd_21_ptr) + r197_z, dtype=tl.float32)))
    r197_value = r197_value.to(tl.float32)
    r197_value = tl.broadcast_to(r197_value, [$(512)])
    tl.store($(vaddcdiv_21_ptr) + r197_z, r197_value)
    r198_z = tl.arange($(512))
    r198_value = tl.load($(input0_ptr) + (r198_z + $(11264)), dtype=tl.bfloat16)
    r198_value = r198_value.to(tl.float32)
    r198_value = tl.broadcast_to(r198_value, [$(512)])
    tl.store($(vto_copy_22_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(512))
    r199_value = (tl.load($(parameter0_ptr) + (r199_z + $(11264)), dtype=tl.float32) * 0.99999)
    r199_value = r199_value.to(tl.float32)
    r199_value = tl.broadcast_to(r199_value, [$(512)])
    tl.store($(vmul_44_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(512))
    r200_value = (0.0 + (tl.load($(vto_copy_22_ptr) + r200_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r200_value = r200_value.to(tl.float32)
    r200_value = tl.broadcast_to(r200_value, [$(512)])
    tl.store($(vlerp_22_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(512))
    r201_value = (0.0 * 0.999)
    r201_value = r201_value.to(tl.float32)
    r201_value = tl.broadcast_to(r201_value, [$(512)])
    tl.store($(vmul_45_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(512))
    r202_value = (tl.load($(vmul_45_ptr) + r202_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_22_ptr) + r202_z, dtype=tl.float32) * tl.load($(vto_copy_22_ptr) + r202_z, dtype=tl.float32))
    r202_value = r202_value.to(tl.float32)
    r202_value = tl.broadcast_to(r202_value, [$(512)])
    tl.store($(vaddcmul_22_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(512))
    r203_value = tl.sqrt(tl.load($(vaddcmul_22_ptr) + r203_z, dtype=tl.float32))
    r203_value = r203_value.to(tl.float32)
    r203_value = tl.broadcast_to(r203_value, [$(512)])
    tl.store($(vsqrt_22_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(512))
    r204_value = (tl.load($(vsqrt_22_ptr) + r204_z, dtype=tl.float32) / 0.031622776601683805)
    r204_value = r204_value.to(tl.float32)
    r204_value = tl.broadcast_to(r204_value, [$(512)])
    tl.store($(vdiv_22_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(512))
    r205_value = (tl.load($(vdiv_22_ptr) + r205_z, dtype=tl.float32) + 0.00000001)
    r205_value = r205_value.to(tl.float32)
    r205_value = tl.broadcast_to(r205_value, [$(512)])
    tl.store($(vadd_22_ptr) + r205_z, r205_value)
    r206_z = tl.arange($(512))
    r206_value = (tl.load($(vmul_44_ptr) + r206_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_22_ptr) + r206_z, dtype=tl.float32) / tl.load($(vadd_22_ptr) + r206_z, dtype=tl.float32)))
    r206_value = r206_value.to(tl.float32)
    r206_value = tl.broadcast_to(r206_value, [$(512)])
    tl.store($(vaddcdiv_22_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(512))
    r207_value = tl.load($(input0_ptr) + (r207_z + $(11776)), dtype=tl.bfloat16)
    r207_value = r207_value.to(tl.float32)
    r207_value = tl.broadcast_to(r207_value, [$(512)])
    tl.store($(vto_copy_23_ptr) + r207_z, r207_value)
    r208_z = tl.arange($(512))
    r208_value = (tl.load($(parameter0_ptr) + (r208_z + $(11776)), dtype=tl.float32) * 0.99999)
    r208_value = r208_value.to(tl.float32)
    r208_value = tl.broadcast_to(r208_value, [$(512)])
    tl.store($(vmul_46_ptr) + r208_z, r208_value)
    r209_z = tl.arange($(512))
    r209_value = (0.0 + (tl.load($(vto_copy_23_ptr) + r209_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r209_value = r209_value.to(tl.float32)
    r209_value = tl.broadcast_to(r209_value, [$(512)])
    tl.store($(vlerp_23_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(512))
    r210_value = (0.0 * 0.999)
    r210_value = r210_value.to(tl.float32)
    r210_value = tl.broadcast_to(r210_value, [$(512)])
    tl.store($(vmul_47_ptr) + r210_z, r210_value)
    r211_z = tl.arange($(512))
    r211_value = (tl.load($(vmul_47_ptr) + r211_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_23_ptr) + r211_z, dtype=tl.float32) * tl.load($(vto_copy_23_ptr) + r211_z, dtype=tl.float32))
    r211_value = r211_value.to(tl.float32)
    r211_value = tl.broadcast_to(r211_value, [$(512)])
    tl.store($(vaddcmul_23_ptr) + r211_z, r211_value)
    r212_z = tl.arange($(512))
    r212_value = tl.sqrt(tl.load($(vaddcmul_23_ptr) + r212_z, dtype=tl.float32))
    r212_value = r212_value.to(tl.float32)
    r212_value = tl.broadcast_to(r212_value, [$(512)])
    tl.store($(vsqrt_23_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(512))
    r213_value = (tl.load($(vsqrt_23_ptr) + r213_z, dtype=tl.float32) / 0.031622776601683805)
    r213_value = r213_value.to(tl.float32)
    r213_value = tl.broadcast_to(r213_value, [$(512)])
    tl.store($(vdiv_23_ptr) + r213_z, r213_value)
    r214_z = tl.arange($(512))
    r214_value = (tl.load($(vdiv_23_ptr) + r214_z, dtype=tl.float32) + 0.00000001)
    r214_value = r214_value.to(tl.float32)
    r214_value = tl.broadcast_to(r214_value, [$(512)])
    tl.store($(vadd_23_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(512))
    r215_value = (tl.load($(vmul_46_ptr) + r215_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_23_ptr) + r215_z, dtype=tl.float32) / tl.load($(vadd_23_ptr) + r215_z, dtype=tl.float32)))
    r215_value = r215_value.to(tl.float32)
    r215_value = tl.broadcast_to(r215_value, [$(512)])
    tl.store($(vaddcdiv_23_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(512))
    r216_value = tl.load($(input0_ptr) + (r216_z + $(12288)), dtype=tl.bfloat16)
    r216_value = r216_value.to(tl.float32)
    r216_value = tl.broadcast_to(r216_value, [$(512)])
    tl.store($(vto_copy_24_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(512))
    r217_value = (tl.load($(parameter0_ptr) + (r217_z + $(12288)), dtype=tl.float32) * 0.99999)
    r217_value = r217_value.to(tl.float32)
    r217_value = tl.broadcast_to(r217_value, [$(512)])
    tl.store($(vmul_48_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(512))
    r218_value = (0.0 + (tl.load($(vto_copy_24_ptr) + r218_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r218_value = r218_value.to(tl.float32)
    r218_value = tl.broadcast_to(r218_value, [$(512)])
    tl.store($(vlerp_24_ptr) + r218_z, r218_value)
    r219_z = tl.arange($(512))
    r219_value = (0.0 * 0.999)
    r219_value = r219_value.to(tl.float32)
    r219_value = tl.broadcast_to(r219_value, [$(512)])
    tl.store($(vmul_49_ptr) + r219_z, r219_value)
    r220_z = tl.arange($(512))
    r220_value = (tl.load($(vmul_49_ptr) + r220_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_24_ptr) + r220_z, dtype=tl.float32) * tl.load($(vto_copy_24_ptr) + r220_z, dtype=tl.float32))
    r220_value = r220_value.to(tl.float32)
    r220_value = tl.broadcast_to(r220_value, [$(512)])
    tl.store($(vaddcmul_24_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(512))
    r221_value = tl.sqrt(tl.load($(vaddcmul_24_ptr) + r221_z, dtype=tl.float32))
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(512)])
    tl.store($(vsqrt_24_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(512))
    r222_value = (tl.load($(vsqrt_24_ptr) + r222_z, dtype=tl.float32) / 0.031622776601683805)
    r222_value = r222_value.to(tl.float32)
    r222_value = tl.broadcast_to(r222_value, [$(512)])
    tl.store($(vdiv_24_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(512))
    r223_value = (tl.load($(vdiv_24_ptr) + r223_z, dtype=tl.float32) + 0.00000001)
    r223_value = r223_value.to(tl.float32)
    r223_value = tl.broadcast_to(r223_value, [$(512)])
    tl.store($(vadd_24_ptr) + r223_z, r223_value)
    r224_z = tl.arange($(512))
    r224_value = (tl.load($(vmul_48_ptr) + r224_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_24_ptr) + r224_z, dtype=tl.float32) / tl.load($(vadd_24_ptr) + r224_z, dtype=tl.float32)))
    r224_value = r224_value.to(tl.float32)
    r224_value = tl.broadcast_to(r224_value, [$(512)])
    tl.store($(vaddcdiv_24_ptr) + r224_z, r224_value)
    r225_z = tl.arange($(512))
    r225_value = tl.load($(input0_ptr) + (r225_z + $(12800)), dtype=tl.bfloat16)
    r225_value = r225_value.to(tl.float32)
    r225_value = tl.broadcast_to(r225_value, [$(512)])
    tl.store($(vto_copy_25_ptr) + r225_z, r225_value)
    r226_z = tl.arange($(512))
    r226_value = (tl.load($(parameter0_ptr) + (r226_z + $(12800)), dtype=tl.float32) * 0.99999)
    r226_value = r226_value.to(tl.float32)
    r226_value = tl.broadcast_to(r226_value, [$(512)])
    tl.store($(vmul_50_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(512))
    r227_value = (0.0 + (tl.load($(vto_copy_25_ptr) + r227_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r227_value = r227_value.to(tl.float32)
    r227_value = tl.broadcast_to(r227_value, [$(512)])
    tl.store($(vlerp_25_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(512))
    r228_value = (0.0 * 0.999)
    r228_value = r228_value.to(tl.float32)
    r228_value = tl.broadcast_to(r228_value, [$(512)])
    tl.store($(vmul_51_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(512))
    r229_value = (tl.load($(vmul_51_ptr) + r229_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_25_ptr) + r229_z, dtype=tl.float32) * tl.load($(vto_copy_25_ptr) + r229_z, dtype=tl.float32))
    r229_value = r229_value.to(tl.float32)
    r229_value = tl.broadcast_to(r229_value, [$(512)])
    tl.store($(vaddcmul_25_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(512))
    r230_value = tl.sqrt(tl.load($(vaddcmul_25_ptr) + r230_z, dtype=tl.float32))
    r230_value = r230_value.to(tl.float32)
    r230_value = tl.broadcast_to(r230_value, [$(512)])
    tl.store($(vsqrt_25_ptr) + r230_z, r230_value)
    r231_z = tl.arange($(512))
    r231_value = (tl.load($(vsqrt_25_ptr) + r231_z, dtype=tl.float32) / 0.031622776601683805)
    r231_value = r231_value.to(tl.float32)
    r231_value = tl.broadcast_to(r231_value, [$(512)])
    tl.store($(vdiv_25_ptr) + r231_z, r231_value)
    r232_z = tl.arange($(512))
    r232_value = (tl.load($(vdiv_25_ptr) + r232_z, dtype=tl.float32) + 0.00000001)
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(512)])
    tl.store($(vadd_25_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(512))
    r233_value = (tl.load($(vmul_50_ptr) + r233_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_25_ptr) + r233_z, dtype=tl.float32) / tl.load($(vadd_25_ptr) + r233_z, dtype=tl.float32)))
    r233_value = r233_value.to(tl.float32)
    r233_value = tl.broadcast_to(r233_value, [$(512)])
    tl.store($(vaddcdiv_25_ptr) + r233_z, r233_value)
    r234_z = tl.arange($(512))
    r234_value = tl.load($(input0_ptr) + (r234_z + $(13312)), dtype=tl.bfloat16)
    r234_value = r234_value.to(tl.float32)
    r234_value = tl.broadcast_to(r234_value, [$(512)])
    tl.store($(vto_copy_26_ptr) + r234_z, r234_value)
    r235_z = tl.arange($(512))
    r235_value = (tl.load($(parameter0_ptr) + (r235_z + $(13312)), dtype=tl.float32) * 0.99999)
    r235_value = r235_value.to(tl.float32)
    r235_value = tl.broadcast_to(r235_value, [$(512)])
    tl.store($(vmul_52_ptr) + r235_z, r235_value)
    r236_z = tl.arange($(512))
    r236_value = (0.0 + (tl.load($(vto_copy_26_ptr) + r236_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r236_value = r236_value.to(tl.float32)
    r236_value = tl.broadcast_to(r236_value, [$(512)])
    tl.store($(vlerp_26_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(512))
    r237_value = (0.0 * 0.999)
    r237_value = r237_value.to(tl.float32)
    r237_value = tl.broadcast_to(r237_value, [$(512)])
    tl.store($(vmul_53_ptr) + r237_z, r237_value)
    r238_z = tl.arange($(512))
    r238_value = (tl.load($(vmul_53_ptr) + r238_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_26_ptr) + r238_z, dtype=tl.float32) * tl.load($(vto_copy_26_ptr) + r238_z, dtype=tl.float32))
    r238_value = r238_value.to(tl.float32)
    r238_value = tl.broadcast_to(r238_value, [$(512)])
    tl.store($(vaddcmul_26_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(512))
    r239_value = tl.sqrt(tl.load($(vaddcmul_26_ptr) + r239_z, dtype=tl.float32))
    r239_value = r239_value.to(tl.float32)
    r239_value = tl.broadcast_to(r239_value, [$(512)])
    tl.store($(vsqrt_26_ptr) + r239_z, r239_value)
    r240_z = tl.arange($(512))
    r240_value = (tl.load($(vsqrt_26_ptr) + r240_z, dtype=tl.float32) / 0.031622776601683805)
    r240_value = r240_value.to(tl.float32)
    r240_value = tl.broadcast_to(r240_value, [$(512)])
    tl.store($(vdiv_26_ptr) + r240_z, r240_value)
    r241_z = tl.arange($(512))
    r241_value = (tl.load($(vdiv_26_ptr) + r241_z, dtype=tl.float32) + 0.00000001)
    r241_value = r241_value.to(tl.float32)
    r241_value = tl.broadcast_to(r241_value, [$(512)])
    tl.store($(vadd_26_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(512))
    r242_value = (tl.load($(vmul_52_ptr) + r242_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_26_ptr) + r242_z, dtype=tl.float32) / tl.load($(vadd_26_ptr) + r242_z, dtype=tl.float32)))
    r242_value = r242_value.to(tl.float32)
    r242_value = tl.broadcast_to(r242_value, [$(512)])
    tl.store($(vaddcdiv_26_ptr) + r242_z, r242_value)
    r243_z = tl.arange($(512))
    r243_value = tl.load($(input0_ptr) + (r243_z + $(13824)), dtype=tl.bfloat16)
    r243_value = r243_value.to(tl.float32)
    r243_value = tl.broadcast_to(r243_value, [$(512)])
    tl.store($(vto_copy_27_ptr) + r243_z, r243_value)
    r244_z = tl.arange($(512))
    r244_value = (tl.load($(parameter0_ptr) + (r244_z + $(13824)), dtype=tl.float32) * 0.99999)
    r244_value = r244_value.to(tl.float32)
    r244_value = tl.broadcast_to(r244_value, [$(512)])
    tl.store($(vmul_54_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(512))
    r245_value = (0.0 + (tl.load($(vto_copy_27_ptr) + r245_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r245_value = r245_value.to(tl.float32)
    r245_value = tl.broadcast_to(r245_value, [$(512)])
    tl.store($(vlerp_27_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(512))
    r246_value = (0.0 * 0.999)
    r246_value = r246_value.to(tl.float32)
    r246_value = tl.broadcast_to(r246_value, [$(512)])
    tl.store($(vmul_55_ptr) + r246_z, r246_value)
    r247_z = tl.arange($(512))
    r247_value = (tl.load($(vmul_55_ptr) + r247_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_27_ptr) + r247_z, dtype=tl.float32) * tl.load($(vto_copy_27_ptr) + r247_z, dtype=tl.float32))
    r247_value = r247_value.to(tl.float32)
    r247_value = tl.broadcast_to(r247_value, [$(512)])
    tl.store($(vaddcmul_27_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(512))
    r248_value = tl.sqrt(tl.load($(vaddcmul_27_ptr) + r248_z, dtype=tl.float32))
    r248_value = r248_value.to(tl.float32)
    r248_value = tl.broadcast_to(r248_value, [$(512)])
    tl.store($(vsqrt_27_ptr) + r248_z, r248_value)
    r249_z = tl.arange($(512))
    r249_value = (tl.load($(vsqrt_27_ptr) + r249_z, dtype=tl.float32) / 0.031622776601683805)
    r249_value = r249_value.to(tl.float32)
    r249_value = tl.broadcast_to(r249_value, [$(512)])
    tl.store($(vdiv_27_ptr) + r249_z, r249_value)
    r250_z = tl.arange($(512))
    r250_value = (tl.load($(vdiv_27_ptr) + r250_z, dtype=tl.float32) + 0.00000001)
    r250_value = r250_value.to(tl.float32)
    r250_value = tl.broadcast_to(r250_value, [$(512)])
    tl.store($(vadd_27_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(512))
    r251_value = (tl.load($(vmul_54_ptr) + r251_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_27_ptr) + r251_z, dtype=tl.float32) / tl.load($(vadd_27_ptr) + r251_z, dtype=tl.float32)))
    r251_value = r251_value.to(tl.float32)
    r251_value = tl.broadcast_to(r251_value, [$(512)])
    tl.store($(vaddcdiv_27_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(512))
    r252_value = tl.load($(input0_ptr) + (r252_z + $(14336)), dtype=tl.bfloat16)
    r252_value = r252_value.to(tl.float32)
    r252_value = tl.broadcast_to(r252_value, [$(512)])
    tl.store($(vto_copy_28_ptr) + r252_z, r252_value)
    r253_z = tl.arange($(512))
    r253_value = (tl.load($(parameter0_ptr) + (r253_z + $(14336)), dtype=tl.float32) * 0.99999)
    r253_value = r253_value.to(tl.float32)
    r253_value = tl.broadcast_to(r253_value, [$(512)])
    tl.store($(vmul_56_ptr) + r253_z, r253_value)
    r254_z = tl.arange($(512))
    r254_value = (0.0 + (tl.load($(vto_copy_28_ptr) + r254_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r254_value = r254_value.to(tl.float32)
    r254_value = tl.broadcast_to(r254_value, [$(512)])
    tl.store($(vlerp_28_ptr) + r254_z, r254_value)
    r255_z = tl.arange($(512))
    r255_value = (0.0 * 0.999)
    r255_value = r255_value.to(tl.float32)
    r255_value = tl.broadcast_to(r255_value, [$(512)])
    tl.store($(vmul_57_ptr) + r255_z, r255_value)
    r256_z = tl.arange($(512))
    r256_value = (tl.load($(vmul_57_ptr) + r256_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_28_ptr) + r256_z, dtype=tl.float32) * tl.load($(vto_copy_28_ptr) + r256_z, dtype=tl.float32))
    r256_value = r256_value.to(tl.float32)
    r256_value = tl.broadcast_to(r256_value, [$(512)])
    tl.store($(vaddcmul_28_ptr) + r256_z, r256_value)
    r257_z = tl.arange($(512))
    r257_value = tl.sqrt(tl.load($(vaddcmul_28_ptr) + r257_z, dtype=tl.float32))
    r257_value = r257_value.to(tl.float32)
    r257_value = tl.broadcast_to(r257_value, [$(512)])
    tl.store($(vsqrt_28_ptr) + r257_z, r257_value)
    r258_z = tl.arange($(512))
    r258_value = (tl.load($(vsqrt_28_ptr) + r258_z, dtype=tl.float32) / 0.031622776601683805)
    r258_value = r258_value.to(tl.float32)
    r258_value = tl.broadcast_to(r258_value, [$(512)])
    tl.store($(vdiv_28_ptr) + r258_z, r258_value)
    r259_z = tl.arange($(512))
    r259_value = (tl.load($(vdiv_28_ptr) + r259_z, dtype=tl.float32) + 0.00000001)
    r259_value = r259_value.to(tl.float32)
    r259_value = tl.broadcast_to(r259_value, [$(512)])
    tl.store($(vadd_28_ptr) + r259_z, r259_value)
    r260_z = tl.arange($(512))
    r260_value = (tl.load($(vmul_56_ptr) + r260_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_28_ptr) + r260_z, dtype=tl.float32) / tl.load($(vadd_28_ptr) + r260_z, dtype=tl.float32)))
    r260_value = r260_value.to(tl.float32)
    r260_value = tl.broadcast_to(r260_value, [$(512)])
    tl.store($(vaddcdiv_28_ptr) + r260_z, r260_value)
    r261_z = tl.arange($(512))
    r261_value = tl.load($(input0_ptr) + (r261_z + $(14848)), dtype=tl.bfloat16)
    r261_value = r261_value.to(tl.float32)
    r261_value = tl.broadcast_to(r261_value, [$(512)])
    tl.store($(vto_copy_29_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(512))
    r262_value = (tl.load($(parameter0_ptr) + (r262_z + $(14848)), dtype=tl.float32) * 0.99999)
    r262_value = r262_value.to(tl.float32)
    r262_value = tl.broadcast_to(r262_value, [$(512)])
    tl.store($(vmul_58_ptr) + r262_z, r262_value)
    r263_z = tl.arange($(512))
    r263_value = (0.0 + (tl.load($(vto_copy_29_ptr) + r263_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r263_value = r263_value.to(tl.float32)
    r263_value = tl.broadcast_to(r263_value, [$(512)])
    tl.store($(vlerp_29_ptr) + r263_z, r263_value)
    r264_z = tl.arange($(512))
    r264_value = (0.0 * 0.999)
    r264_value = r264_value.to(tl.float32)
    r264_value = tl.broadcast_to(r264_value, [$(512)])
    tl.store($(vmul_59_ptr) + r264_z, r264_value)
    r265_z = tl.arange($(512))
    r265_value = (tl.load($(vmul_59_ptr) + r265_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_29_ptr) + r265_z, dtype=tl.float32) * tl.load($(vto_copy_29_ptr) + r265_z, dtype=tl.float32))
    r265_value = r265_value.to(tl.float32)
    r265_value = tl.broadcast_to(r265_value, [$(512)])
    tl.store($(vaddcmul_29_ptr) + r265_z, r265_value)
    r266_z = tl.arange($(512))
    r266_value = tl.sqrt(tl.load($(vaddcmul_29_ptr) + r266_z, dtype=tl.float32))
    r266_value = r266_value.to(tl.float32)
    r266_value = tl.broadcast_to(r266_value, [$(512)])
    tl.store($(vsqrt_29_ptr) + r266_z, r266_value)
    r267_z = tl.arange($(512))
    r267_value = (tl.load($(vsqrt_29_ptr) + r267_z, dtype=tl.float32) / 0.031622776601683805)
    r267_value = r267_value.to(tl.float32)
    r267_value = tl.broadcast_to(r267_value, [$(512)])
    tl.store($(vdiv_29_ptr) + r267_z, r267_value)
    r268_z = tl.arange($(512))
    r268_value = (tl.load($(vdiv_29_ptr) + r268_z, dtype=tl.float32) + 0.00000001)
    r268_value = r268_value.to(tl.float32)
    r268_value = tl.broadcast_to(r268_value, [$(512)])
    tl.store($(vadd_29_ptr) + r268_z, r268_value)
    r269_z = tl.arange($(512))
    r269_value = (tl.load($(vmul_58_ptr) + r269_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_29_ptr) + r269_z, dtype=tl.float32) / tl.load($(vadd_29_ptr) + r269_z, dtype=tl.float32)))
    r269_value = r269_value.to(tl.float32)
    r269_value = tl.broadcast_to(r269_value, [$(512)])
    tl.store($(vaddcdiv_29_ptr) + r269_z, r269_value)
    r270_z = tl.arange($(512))
    r270_value = tl.load($(input0_ptr) + (r270_z + $(15360)), dtype=tl.bfloat16)
    r270_value = r270_value.to(tl.float32)
    r270_value = tl.broadcast_to(r270_value, [$(512)])
    tl.store($(vto_copy_30_ptr) + r270_z, r270_value)
    r271_z = tl.arange($(512))
    r271_value = (tl.load($(parameter0_ptr) + (r271_z + $(15360)), dtype=tl.float32) * 0.99999)
    r271_value = r271_value.to(tl.float32)
    r271_value = tl.broadcast_to(r271_value, [$(512)])
    tl.store($(vmul_60_ptr) + r271_z, r271_value)
    r272_z = tl.arange($(512))
    r272_value = (0.0 + (tl.load($(vto_copy_30_ptr) + r272_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r272_value = r272_value.to(tl.float32)
    r272_value = tl.broadcast_to(r272_value, [$(512)])
    tl.store($(vlerp_30_ptr) + r272_z, r272_value)
    r273_z = tl.arange($(512))
    r273_value = (0.0 * 0.999)
    r273_value = r273_value.to(tl.float32)
    r273_value = tl.broadcast_to(r273_value, [$(512)])
    tl.store($(vmul_61_ptr) + r273_z, r273_value)
    r274_z = tl.arange($(512))
    r274_value = (tl.load($(vmul_61_ptr) + r274_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_30_ptr) + r274_z, dtype=tl.float32) * tl.load($(vto_copy_30_ptr) + r274_z, dtype=tl.float32))
    r274_value = r274_value.to(tl.float32)
    r274_value = tl.broadcast_to(r274_value, [$(512)])
    tl.store($(vaddcmul_30_ptr) + r274_z, r274_value)
    r275_z = tl.arange($(512))
    r275_value = tl.sqrt(tl.load($(vaddcmul_30_ptr) + r275_z, dtype=tl.float32))
    r275_value = r275_value.to(tl.float32)
    r275_value = tl.broadcast_to(r275_value, [$(512)])
    tl.store($(vsqrt_30_ptr) + r275_z, r275_value)
    r276_z = tl.arange($(512))
    r276_value = (tl.load($(vsqrt_30_ptr) + r276_z, dtype=tl.float32) / 0.031622776601683805)
    r276_value = r276_value.to(tl.float32)
    r276_value = tl.broadcast_to(r276_value, [$(512)])
    tl.store($(vdiv_30_ptr) + r276_z, r276_value)
    r277_z = tl.arange($(512))
    r277_value = (tl.load($(vdiv_30_ptr) + r277_z, dtype=tl.float32) + 0.00000001)
    r277_value = r277_value.to(tl.float32)
    r277_value = tl.broadcast_to(r277_value, [$(512)])
    tl.store($(vadd_30_ptr) + r277_z, r277_value)
    r278_z = tl.arange($(512))
    r278_value = (tl.load($(vmul_60_ptr) + r278_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_30_ptr) + r278_z, dtype=tl.float32) / tl.load($(vadd_30_ptr) + r278_z, dtype=tl.float32)))
    r278_value = r278_value.to(tl.float32)
    r278_value = tl.broadcast_to(r278_value, [$(512)])
    tl.store($(vaddcdiv_30_ptr) + r278_z, r278_value)
    r279_z = tl.arange($(512))
    r279_value = tl.load($(input0_ptr) + (r279_z + $(15872)), dtype=tl.bfloat16)
    r279_value = r279_value.to(tl.float32)
    r279_value = tl.broadcast_to(r279_value, [$(512)])
    tl.store($(vto_copy_31_ptr) + r279_z, r279_value)
    r280_z = tl.arange($(512))
    r280_value = (tl.load($(parameter0_ptr) + (r280_z + $(15872)), dtype=tl.float32) * 0.99999)
    r280_value = r280_value.to(tl.float32)
    r280_value = tl.broadcast_to(r280_value, [$(512)])
    tl.store($(vmul_62_ptr) + r280_z, r280_value)
    r281_z = tl.arange($(512))
    r281_value = (0.0 + (tl.load($(vto_copy_31_ptr) + r281_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r281_value = r281_value.to(tl.float32)
    r281_value = tl.broadcast_to(r281_value, [$(512)])
    tl.store($(vlerp_31_ptr) + r281_z, r281_value)
    r282_z = tl.arange($(512))
    r282_value = (0.0 * 0.999)
    r282_value = r282_value.to(tl.float32)
    r282_value = tl.broadcast_to(r282_value, [$(512)])
    tl.store($(vmul_63_ptr) + r282_z, r282_value)
    r283_z = tl.arange($(512))
    r283_value = (tl.load($(vmul_63_ptr) + r283_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(vto_copy_31_ptr) + r283_z, dtype=tl.float32) * tl.load($(vto_copy_31_ptr) + r283_z, dtype=tl.float32))
    r283_value = r283_value.to(tl.float32)
    r283_value = tl.broadcast_to(r283_value, [$(512)])
    tl.store($(vaddcmul_31_ptr) + r283_z, r283_value)
    r284_z = tl.arange($(512))
    r284_value = tl.sqrt(tl.load($(vaddcmul_31_ptr) + r284_z, dtype=tl.float32))
    r284_value = r284_value.to(tl.float32)
    r284_value = tl.broadcast_to(r284_value, [$(512)])
    tl.store($(vsqrt_31_ptr) + r284_z, r284_value)
    r285_z = tl.arange($(512))
    r285_value = (tl.load($(vsqrt_31_ptr) + r285_z, dtype=tl.float32) / 0.031622776601683805)
    r285_value = r285_value.to(tl.float32)
    r285_value = tl.broadcast_to(r285_value, [$(512)])
    tl.store($(vdiv_31_ptr) + r285_z, r285_value)
    r286_z = tl.arange($(512))
    r286_value = (tl.load($(vdiv_31_ptr) + r286_z, dtype=tl.float32) + 0.00000001)
    r286_value = r286_value.to(tl.float32)
    r286_value = tl.broadcast_to(r286_value, [$(512)])
    tl.store($(vadd_31_ptr) + r286_z, r286_value)
    r287_z = tl.arange($(512))
    r287_value = (tl.load($(vmul_62_ptr) + r287_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(vlerp_31_ptr) + r287_z, dtype=tl.float32) / tl.load($(vadd_31_ptr) + r287_z, dtype=tl.float32)))
    r287_value = r287_value.to(tl.float32)
    r287_value = tl.broadcast_to(r287_value, [$(512)])
    tl.store($(vaddcdiv_31_ptr) + r287_z, r287_value)
    r288_z = tl.arange($(16384))
    r288_value = tl.where(((r288_z / $(512)) % $(32)) < $(1), tl.load($(vaddcdiv_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(2), tl.load($(vaddcdiv_1_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(3), tl.load($(vaddcdiv_2_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(4), tl.load($(vaddcdiv_3_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(5), tl.load($(vaddcdiv_4_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(6), tl.load($(vaddcdiv_5_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(7), tl.load($(vaddcdiv_6_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(8), tl.load($(vaddcdiv_7_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(9), tl.load($(vaddcdiv_8_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(10), tl.load($(vaddcdiv_9_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(11), tl.load($(vaddcdiv_10_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(12), tl.load($(vaddcdiv_11_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(13), tl.load($(vaddcdiv_12_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(14), tl.load($(vaddcdiv_13_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(15), tl.load($(vaddcdiv_14_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(16), tl.load($(vaddcdiv_15_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(17), tl.load($(vaddcdiv_16_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(18), tl.load($(vaddcdiv_17_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(19), tl.load($(vaddcdiv_18_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(20), tl.load($(vaddcdiv_19_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(21), tl.load($(vaddcdiv_20_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(22), tl.load($(vaddcdiv_21_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(23), tl.load($(vaddcdiv_22_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(24), tl.load($(vaddcdiv_23_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(25), tl.load($(vaddcdiv_24_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(26), tl.load($(vaddcdiv_25_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(27), tl.load($(vaddcdiv_26_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(28), tl.load($(vaddcdiv_27_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(29), tl.load($(vaddcdiv_28_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(30), tl.load($(vaddcdiv_29_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(31), tl.load($(vaddcdiv_30_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.load($(vaddcdiv_31_ptr) + (r288_z % $(512)), dtype=tl.float32))))))))))))))))))))))))))))))))
    r288_value = r288_value.to(tl.float32)
    r288_value = tl.broadcast_to(r288_value, [$(16384)])
    tl.store($(vstack_ptr) + r288_z, r288_value)
    z = tl.arange($(16384))
    value = tl.load($(vstack_ptr) + ((((z / $(512)) % $(32)) * $(512)) + (z % $(512))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(16384)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vmul_ptr : RegionName := "vmul"
  let vlerp_ptr : RegionName := "vlerp"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vaddcmul_ptr : RegionName := "vaddcmul"
  let vsqrt_ptr : RegionName := "vsqrt"
  let vdiv_ptr : RegionName := "vdiv"
  let vadd_ptr : RegionName := "vadd"
  let vaddcdiv_ptr : RegionName := "vaddcdiv"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vlerp_1_ptr : RegionName := "vlerp_1"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vaddcmul_1_ptr : RegionName := "vaddcmul_1"
  let vsqrt_1_ptr : RegionName := "vsqrt_1"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vaddcdiv_1_ptr : RegionName := "vaddcdiv_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vlerp_2_ptr : RegionName := "vlerp_2"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vaddcmul_2_ptr : RegionName := "vaddcmul_2"
  let vsqrt_2_ptr : RegionName := "vsqrt_2"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vaddcdiv_2_ptr : RegionName := "vaddcdiv_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vlerp_3_ptr : RegionName := "vlerp_3"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vaddcmul_3_ptr : RegionName := "vaddcmul_3"
  let vsqrt_3_ptr : RegionName := "vsqrt_3"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vaddcdiv_3_ptr : RegionName := "vaddcdiv_3"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vlerp_4_ptr : RegionName := "vlerp_4"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vaddcmul_4_ptr : RegionName := "vaddcmul_4"
  let vsqrt_4_ptr : RegionName := "vsqrt_4"
  let vdiv_4_ptr : RegionName := "vdiv_4"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vaddcdiv_4_ptr : RegionName := "vaddcdiv_4"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vlerp_5_ptr : RegionName := "vlerp_5"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vaddcmul_5_ptr : RegionName := "vaddcmul_5"
  let vsqrt_5_ptr : RegionName := "vsqrt_5"
  let vdiv_5_ptr : RegionName := "vdiv_5"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vaddcdiv_5_ptr : RegionName := "vaddcdiv_5"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vlerp_6_ptr : RegionName := "vlerp_6"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vaddcmul_6_ptr : RegionName := "vaddcmul_6"
  let vsqrt_6_ptr : RegionName := "vsqrt_6"
  let vdiv_6_ptr : RegionName := "vdiv_6"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vaddcdiv_6_ptr : RegionName := "vaddcdiv_6"
  let vto_copy_7_ptr : RegionName := "vto_copy_7"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vlerp_7_ptr : RegionName := "vlerp_7"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vaddcmul_7_ptr : RegionName := "vaddcmul_7"
  let vsqrt_7_ptr : RegionName := "vsqrt_7"
  let vdiv_7_ptr : RegionName := "vdiv_7"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vaddcdiv_7_ptr : RegionName := "vaddcdiv_7"
  let vto_copy_8_ptr : RegionName := "vto_copy_8"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vlerp_8_ptr : RegionName := "vlerp_8"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vaddcmul_8_ptr : RegionName := "vaddcmul_8"
  let vsqrt_8_ptr : RegionName := "vsqrt_8"
  let vdiv_8_ptr : RegionName := "vdiv_8"
  let vadd_8_ptr : RegionName := "vadd_8"
  let vaddcdiv_8_ptr : RegionName := "vaddcdiv_8"
  let vto_copy_9_ptr : RegionName := "vto_copy_9"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vlerp_9_ptr : RegionName := "vlerp_9"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vaddcmul_9_ptr : RegionName := "vaddcmul_9"
  let vsqrt_9_ptr : RegionName := "vsqrt_9"
  let vdiv_9_ptr : RegionName := "vdiv_9"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vaddcdiv_9_ptr : RegionName := "vaddcdiv_9"
  let vto_copy_10_ptr : RegionName := "vto_copy_10"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vlerp_10_ptr : RegionName := "vlerp_10"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vaddcmul_10_ptr : RegionName := "vaddcmul_10"
  let vsqrt_10_ptr : RegionName := "vsqrt_10"
  let vdiv_10_ptr : RegionName := "vdiv_10"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vaddcdiv_10_ptr : RegionName := "vaddcdiv_10"
  let vto_copy_11_ptr : RegionName := "vto_copy_11"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vlerp_11_ptr : RegionName := "vlerp_11"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vaddcmul_11_ptr : RegionName := "vaddcmul_11"
  let vsqrt_11_ptr : RegionName := "vsqrt_11"
  let vdiv_11_ptr : RegionName := "vdiv_11"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vaddcdiv_11_ptr : RegionName := "vaddcdiv_11"
  let vto_copy_12_ptr : RegionName := "vto_copy_12"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vlerp_12_ptr : RegionName := "vlerp_12"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vaddcmul_12_ptr : RegionName := "vaddcmul_12"
  let vsqrt_12_ptr : RegionName := "vsqrt_12"
  let vdiv_12_ptr : RegionName := "vdiv_12"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vaddcdiv_12_ptr : RegionName := "vaddcdiv_12"
  let vto_copy_13_ptr : RegionName := "vto_copy_13"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vlerp_13_ptr : RegionName := "vlerp_13"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vaddcmul_13_ptr : RegionName := "vaddcmul_13"
  let vsqrt_13_ptr : RegionName := "vsqrt_13"
  let vdiv_13_ptr : RegionName := "vdiv_13"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vaddcdiv_13_ptr : RegionName := "vaddcdiv_13"
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vlerp_14_ptr : RegionName := "vlerp_14"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vaddcmul_14_ptr : RegionName := "vaddcmul_14"
  let vsqrt_14_ptr : RegionName := "vsqrt_14"
  let vdiv_14_ptr : RegionName := "vdiv_14"
  let vadd_14_ptr : RegionName := "vadd_14"
  let vaddcdiv_14_ptr : RegionName := "vaddcdiv_14"
  let vto_copy_15_ptr : RegionName := "vto_copy_15"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vlerp_15_ptr : RegionName := "vlerp_15"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vaddcmul_15_ptr : RegionName := "vaddcmul_15"
  let vsqrt_15_ptr : RegionName := "vsqrt_15"
  let vdiv_15_ptr : RegionName := "vdiv_15"
  let vadd_15_ptr : RegionName := "vadd_15"
  let vaddcdiv_15_ptr : RegionName := "vaddcdiv_15"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vlerp_16_ptr : RegionName := "vlerp_16"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vaddcmul_16_ptr : RegionName := "vaddcmul_16"
  let vsqrt_16_ptr : RegionName := "vsqrt_16"
  let vdiv_16_ptr : RegionName := "vdiv_16"
  let vadd_16_ptr : RegionName := "vadd_16"
  let vaddcdiv_16_ptr : RegionName := "vaddcdiv_16"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vlerp_17_ptr : RegionName := "vlerp_17"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vaddcmul_17_ptr : RegionName := "vaddcmul_17"
  let vsqrt_17_ptr : RegionName := "vsqrt_17"
  let vdiv_17_ptr : RegionName := "vdiv_17"
  let vadd_17_ptr : RegionName := "vadd_17"
  let vaddcdiv_17_ptr : RegionName := "vaddcdiv_17"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vlerp_18_ptr : RegionName := "vlerp_18"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vaddcmul_18_ptr : RegionName := "vaddcmul_18"
  let vsqrt_18_ptr : RegionName := "vsqrt_18"
  let vdiv_18_ptr : RegionName := "vdiv_18"
  let vadd_18_ptr : RegionName := "vadd_18"
  let vaddcdiv_18_ptr : RegionName := "vaddcdiv_18"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vlerp_19_ptr : RegionName := "vlerp_19"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vaddcmul_19_ptr : RegionName := "vaddcmul_19"
  let vsqrt_19_ptr : RegionName := "vsqrt_19"
  let vdiv_19_ptr : RegionName := "vdiv_19"
  let vadd_19_ptr : RegionName := "vadd_19"
  let vaddcdiv_19_ptr : RegionName := "vaddcdiv_19"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vlerp_20_ptr : RegionName := "vlerp_20"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vaddcmul_20_ptr : RegionName := "vaddcmul_20"
  let vsqrt_20_ptr : RegionName := "vsqrt_20"
  let vdiv_20_ptr : RegionName := "vdiv_20"
  let vadd_20_ptr : RegionName := "vadd_20"
  let vaddcdiv_20_ptr : RegionName := "vaddcdiv_20"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vlerp_21_ptr : RegionName := "vlerp_21"
  let vmul_43_ptr : RegionName := "vmul_43"
  let vaddcmul_21_ptr : RegionName := "vaddcmul_21"
  let vsqrt_21_ptr : RegionName := "vsqrt_21"
  let vdiv_21_ptr : RegionName := "vdiv_21"
  let vadd_21_ptr : RegionName := "vadd_21"
  let vaddcdiv_21_ptr : RegionName := "vaddcdiv_21"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vmul_44_ptr : RegionName := "vmul_44"
  let vlerp_22_ptr : RegionName := "vlerp_22"
  let vmul_45_ptr : RegionName := "vmul_45"
  let vaddcmul_22_ptr : RegionName := "vaddcmul_22"
  let vsqrt_22_ptr : RegionName := "vsqrt_22"
  let vdiv_22_ptr : RegionName := "vdiv_22"
  let vadd_22_ptr : RegionName := "vadd_22"
  let vaddcdiv_22_ptr : RegionName := "vaddcdiv_22"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmul_46_ptr : RegionName := "vmul_46"
  let vlerp_23_ptr : RegionName := "vlerp_23"
  let vmul_47_ptr : RegionName := "vmul_47"
  let vaddcmul_23_ptr : RegionName := "vaddcmul_23"
  let vsqrt_23_ptr : RegionName := "vsqrt_23"
  let vdiv_23_ptr : RegionName := "vdiv_23"
  let vadd_23_ptr : RegionName := "vadd_23"
  let vaddcdiv_23_ptr : RegionName := "vaddcdiv_23"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vmul_48_ptr : RegionName := "vmul_48"
  let vlerp_24_ptr : RegionName := "vlerp_24"
  let vmul_49_ptr : RegionName := "vmul_49"
  let vaddcmul_24_ptr : RegionName := "vaddcmul_24"
  let vsqrt_24_ptr : RegionName := "vsqrt_24"
  let vdiv_24_ptr : RegionName := "vdiv_24"
  let vadd_24_ptr : RegionName := "vadd_24"
  let vaddcdiv_24_ptr : RegionName := "vaddcdiv_24"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vmul_50_ptr : RegionName := "vmul_50"
  let vlerp_25_ptr : RegionName := "vlerp_25"
  let vmul_51_ptr : RegionName := "vmul_51"
  let vaddcmul_25_ptr : RegionName := "vaddcmul_25"
  let vsqrt_25_ptr : RegionName := "vsqrt_25"
  let vdiv_25_ptr : RegionName := "vdiv_25"
  let vadd_25_ptr : RegionName := "vadd_25"
  let vaddcdiv_25_ptr : RegionName := "vaddcdiv_25"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmul_52_ptr : RegionName := "vmul_52"
  let vlerp_26_ptr : RegionName := "vlerp_26"
  let vmul_53_ptr : RegionName := "vmul_53"
  let vaddcmul_26_ptr : RegionName := "vaddcmul_26"
  let vsqrt_26_ptr : RegionName := "vsqrt_26"
  let vdiv_26_ptr : RegionName := "vdiv_26"
  let vadd_26_ptr : RegionName := "vadd_26"
  let vaddcdiv_26_ptr : RegionName := "vaddcdiv_26"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vmul_54_ptr : RegionName := "vmul_54"
  let vlerp_27_ptr : RegionName := "vlerp_27"
  let vmul_55_ptr : RegionName := "vmul_55"
  let vaddcmul_27_ptr : RegionName := "vaddcmul_27"
  let vsqrt_27_ptr : RegionName := "vsqrt_27"
  let vdiv_27_ptr : RegionName := "vdiv_27"
  let vadd_27_ptr : RegionName := "vadd_27"
  let vaddcdiv_27_ptr : RegionName := "vaddcdiv_27"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_56_ptr : RegionName := "vmul_56"
  let vlerp_28_ptr : RegionName := "vlerp_28"
  let vmul_57_ptr : RegionName := "vmul_57"
  let vaddcmul_28_ptr : RegionName := "vaddcmul_28"
  let vsqrt_28_ptr : RegionName := "vsqrt_28"
  let vdiv_28_ptr : RegionName := "vdiv_28"
  let vadd_28_ptr : RegionName := "vadd_28"
  let vaddcdiv_28_ptr : RegionName := "vaddcdiv_28"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vmul_58_ptr : RegionName := "vmul_58"
  let vlerp_29_ptr : RegionName := "vlerp_29"
  let vmul_59_ptr : RegionName := "vmul_59"
  let vaddcmul_29_ptr : RegionName := "vaddcmul_29"
  let vsqrt_29_ptr : RegionName := "vsqrt_29"
  let vdiv_29_ptr : RegionName := "vdiv_29"
  let vadd_29_ptr : RegionName := "vadd_29"
  let vaddcdiv_29_ptr : RegionName := "vaddcdiv_29"
  let vto_copy_30_ptr : RegionName := "vto_copy_30"
  let vmul_60_ptr : RegionName := "vmul_60"
  let vlerp_30_ptr : RegionName := "vlerp_30"
  let vmul_61_ptr : RegionName := "vmul_61"
  let vaddcmul_30_ptr : RegionName := "vaddcmul_30"
  let vsqrt_30_ptr : RegionName := "vsqrt_30"
  let vdiv_30_ptr : RegionName := "vdiv_30"
  let vadd_30_ptr : RegionName := "vadd_30"
  let vaddcdiv_30_ptr : RegionName := "vaddcdiv_30"
  let vto_copy_31_ptr : RegionName := "vto_copy_31"
  let vmul_62_ptr : RegionName := "vmul_62"
  let vlerp_31_ptr : RegionName := "vlerp_31"
  let vmul_63_ptr : RegionName := "vmul_63"
  let vaddcmul_31_ptr : RegionName := "vaddcmul_31"
  let vsqrt_31_ptr : RegionName := "vsqrt_31"
  let vdiv_31_ptr : RegionName := "vdiv_31"
  let vadd_31_ptr : RegionName := "vadd_31"
  let vaddcdiv_31_ptr : RegionName := "vaddcdiv_31"
  let vstack_ptr : RegionName := "vstack"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vto_copy_7_ptr : RegionName := "candidate_vto_copy_7"
  let candidate_vto_copy_8_ptr : RegionName := "candidate_vto_copy_8"
  let candidate_vto_copy_9_ptr : RegionName := "candidate_vto_copy_9"
  let candidate_vto_copy_10_ptr : RegionName := "candidate_vto_copy_10"
  let candidate_vto_copy_11_ptr : RegionName := "candidate_vto_copy_11"
  let candidate_vto_copy_12_ptr : RegionName := "candidate_vto_copy_12"
  let candidate_vto_copy_13_ptr : RegionName := "candidate_vto_copy_13"
  let candidate_vto_copy_14_ptr : RegionName := "candidate_vto_copy_14"
  let candidate_vto_copy_15_ptr : RegionName := "candidate_vto_copy_15"
  let candidate_vto_copy_16_ptr : RegionName := "candidate_vto_copy_16"
  let candidate_vto_copy_17_ptr : RegionName := "candidate_vto_copy_17"
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vmul_6_ptr : RegionName := "candidate_vmul_6"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vlerp_ptr : RegionName := "candidate_vlerp"
  let candidate_vlerp_1_ptr : RegionName := "candidate_vlerp_1"
  let candidate_vlerp_2_ptr : RegionName := "candidate_vlerp_2"
  let candidate_vlerp_3_ptr : RegionName := "candidate_vlerp_3"
  let candidate_vlerp_4_ptr : RegionName := "candidate_vlerp_4"
  let candidate_vlerp_5_ptr : RegionName := "candidate_vlerp_5"
  let candidate_vlerp_6_ptr : RegionName := "candidate_vlerp_6"
  let candidate_vlerp_7_ptr : RegionName := "candidate_vlerp_7"
  let candidate_vlerp_8_ptr : RegionName := "candidate_vlerp_8"
  let candidate_vlerp_9_ptr : RegionName := "candidate_vlerp_9"
  let candidate_vlerp_10_ptr : RegionName := "candidate_vlerp_10"
  let candidate_vlerp_11_ptr : RegionName := "candidate_vlerp_11"
  let candidate_vlerp_12_ptr : RegionName := "candidate_vlerp_12"
  let candidate_vlerp_13_ptr : RegionName := "candidate_vlerp_13"
  let candidate_vlerp_14_ptr : RegionName := "candidate_vlerp_14"
  let candidate_vlerp_15_ptr : RegionName := "candidate_vlerp_15"
  let candidate_vlerp_16_ptr : RegionName := "candidate_vlerp_16"
  let candidate_vlerp_17_ptr : RegionName := "candidate_vlerp_17"
  let candidate_vlerp_18_ptr : RegionName := "candidate_vlerp_18"
  let candidate_vlerp_19_ptr : RegionName := "candidate_vlerp_19"
  let candidate_vlerp_20_ptr : RegionName := "candidate_vlerp_20"
  let candidate_vlerp_21_ptr : RegionName := "candidate_vlerp_21"
  let candidate_vlerp_22_ptr : RegionName := "candidate_vlerp_22"
  let candidate_vlerp_23_ptr : RegionName := "candidate_vlerp_23"
  let candidate_vlerp_24_ptr : RegionName := "candidate_vlerp_24"
  let candidate_vlerp_25_ptr : RegionName := "candidate_vlerp_25"
  let candidate_vlerp_26_ptr : RegionName := "candidate_vlerp_26"
  let candidate_vlerp_27_ptr : RegionName := "candidate_vlerp_27"
  let candidate_vlerp_28_ptr : RegionName := "candidate_vlerp_28"
  let candidate_vlerp_29_ptr : RegionName := "candidate_vlerp_29"
  let candidate_vlerp_30_ptr : RegionName := "candidate_vlerp_30"
  let candidate_vlerp_31_ptr : RegionName := "candidate_vlerp_31"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vmul_50_ptr : RegionName := "candidate_vmul_50"
  let candidate_vmul_51_ptr : RegionName := "candidate_vmul_51"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vmul_56_ptr : RegionName := "candidate_vmul_56"
  let candidate_vmul_57_ptr : RegionName := "candidate_vmul_57"
  let candidate_vmul_58_ptr : RegionName := "candidate_vmul_58"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vmul_61_ptr : RegionName := "candidate_vmul_61"
  let candidate_vmul_62_ptr : RegionName := "candidate_vmul_62"
  let candidate_vmul_63_ptr : RegionName := "candidate_vmul_63"
  let candidate_vaddcmul_ptr : RegionName := "candidate_vaddcmul"
  let candidate_vaddcmul_1_ptr : RegionName := "candidate_vaddcmul_1"
  let candidate_vaddcmul_2_ptr : RegionName := "candidate_vaddcmul_2"
  let candidate_vaddcmul_3_ptr : RegionName := "candidate_vaddcmul_3"
  let candidate_vaddcmul_4_ptr : RegionName := "candidate_vaddcmul_4"
  let candidate_vaddcmul_5_ptr : RegionName := "candidate_vaddcmul_5"
  let candidate_vaddcmul_6_ptr : RegionName := "candidate_vaddcmul_6"
  let candidate_vaddcmul_7_ptr : RegionName := "candidate_vaddcmul_7"
  let candidate_vaddcmul_8_ptr : RegionName := "candidate_vaddcmul_8"
  let candidate_vaddcmul_9_ptr : RegionName := "candidate_vaddcmul_9"
  let candidate_vaddcmul_10_ptr : RegionName := "candidate_vaddcmul_10"
  let candidate_vaddcmul_11_ptr : RegionName := "candidate_vaddcmul_11"
  let candidate_vaddcmul_12_ptr : RegionName := "candidate_vaddcmul_12"
  let candidate_vaddcmul_13_ptr : RegionName := "candidate_vaddcmul_13"
  let candidate_vaddcmul_14_ptr : RegionName := "candidate_vaddcmul_14"
  let candidate_vaddcmul_15_ptr : RegionName := "candidate_vaddcmul_15"
  let candidate_vaddcmul_16_ptr : RegionName := "candidate_vaddcmul_16"
  let candidate_vaddcmul_17_ptr : RegionName := "candidate_vaddcmul_17"
  let candidate_vaddcmul_18_ptr : RegionName := "candidate_vaddcmul_18"
  let candidate_vaddcmul_19_ptr : RegionName := "candidate_vaddcmul_19"
  let candidate_vaddcmul_20_ptr : RegionName := "candidate_vaddcmul_20"
  let candidate_vaddcmul_21_ptr : RegionName := "candidate_vaddcmul_21"
  let candidate_vaddcmul_22_ptr : RegionName := "candidate_vaddcmul_22"
  let candidate_vaddcmul_23_ptr : RegionName := "candidate_vaddcmul_23"
  let candidate_vaddcmul_24_ptr : RegionName := "candidate_vaddcmul_24"
  let candidate_vaddcmul_25_ptr : RegionName := "candidate_vaddcmul_25"
  let candidate_vaddcmul_26_ptr : RegionName := "candidate_vaddcmul_26"
  let candidate_vaddcmul_27_ptr : RegionName := "candidate_vaddcmul_27"
  let candidate_vaddcmul_28_ptr : RegionName := "candidate_vaddcmul_28"
  let candidate_vaddcmul_29_ptr : RegionName := "candidate_vaddcmul_29"
  let candidate_vaddcmul_30_ptr : RegionName := "candidate_vaddcmul_30"
  let candidate_vaddcmul_31_ptr : RegionName := "candidate_vaddcmul_31"
  let candidate_vsqrt_ptr : RegionName := "candidate_vsqrt"
  let candidate_vsqrt_1_ptr : RegionName := "candidate_vsqrt_1"
  let candidate_vsqrt_2_ptr : RegionName := "candidate_vsqrt_2"
  let candidate_vsqrt_3_ptr : RegionName := "candidate_vsqrt_3"
  let candidate_vsqrt_4_ptr : RegionName := "candidate_vsqrt_4"
  let candidate_vsqrt_5_ptr : RegionName := "candidate_vsqrt_5"
  let candidate_vsqrt_6_ptr : RegionName := "candidate_vsqrt_6"
  let candidate_vsqrt_7_ptr : RegionName := "candidate_vsqrt_7"
  let candidate_vsqrt_8_ptr : RegionName := "candidate_vsqrt_8"
  let candidate_vsqrt_9_ptr : RegionName := "candidate_vsqrt_9"
  let candidate_vsqrt_10_ptr : RegionName := "candidate_vsqrt_10"
  let candidate_vsqrt_11_ptr : RegionName := "candidate_vsqrt_11"
  let candidate_vsqrt_12_ptr : RegionName := "candidate_vsqrt_12"
  let candidate_vsqrt_13_ptr : RegionName := "candidate_vsqrt_13"
  let candidate_vsqrt_14_ptr : RegionName := "candidate_vsqrt_14"
  let candidate_vsqrt_15_ptr : RegionName := "candidate_vsqrt_15"
  let candidate_vsqrt_16_ptr : RegionName := "candidate_vsqrt_16"
  let candidate_vsqrt_17_ptr : RegionName := "candidate_vsqrt_17"
  let candidate_vsqrt_18_ptr : RegionName := "candidate_vsqrt_18"
  let candidate_vsqrt_19_ptr : RegionName := "candidate_vsqrt_19"
  let candidate_vsqrt_20_ptr : RegionName := "candidate_vsqrt_20"
  let candidate_vsqrt_21_ptr : RegionName := "candidate_vsqrt_21"
  let candidate_vsqrt_22_ptr : RegionName := "candidate_vsqrt_22"
  let candidate_vsqrt_23_ptr : RegionName := "candidate_vsqrt_23"
  let candidate_vsqrt_24_ptr : RegionName := "candidate_vsqrt_24"
  let candidate_vsqrt_25_ptr : RegionName := "candidate_vsqrt_25"
  let candidate_vsqrt_26_ptr : RegionName := "candidate_vsqrt_26"
  let candidate_vsqrt_27_ptr : RegionName := "candidate_vsqrt_27"
  let candidate_vsqrt_28_ptr : RegionName := "candidate_vsqrt_28"
  let candidate_vsqrt_29_ptr : RegionName := "candidate_vsqrt_29"
  let candidate_vsqrt_30_ptr : RegionName := "candidate_vsqrt_30"
  let candidate_vsqrt_31_ptr : RegionName := "candidate_vsqrt_31"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vdiv_4_ptr : RegionName := "candidate_vdiv_4"
  let candidate_vdiv_5_ptr : RegionName := "candidate_vdiv_5"
  let candidate_vdiv_6_ptr : RegionName := "candidate_vdiv_6"
  let candidate_vdiv_7_ptr : RegionName := "candidate_vdiv_7"
  let candidate_vdiv_8_ptr : RegionName := "candidate_vdiv_8"
  let candidate_vdiv_9_ptr : RegionName := "candidate_vdiv_9"
  let candidate_vdiv_10_ptr : RegionName := "candidate_vdiv_10"
  let candidate_vdiv_11_ptr : RegionName := "candidate_vdiv_11"
  let candidate_vdiv_12_ptr : RegionName := "candidate_vdiv_12"
  let candidate_vdiv_13_ptr : RegionName := "candidate_vdiv_13"
  let candidate_vdiv_14_ptr : RegionName := "candidate_vdiv_14"
  let candidate_vdiv_15_ptr : RegionName := "candidate_vdiv_15"
  let candidate_vdiv_16_ptr : RegionName := "candidate_vdiv_16"
  let candidate_vdiv_17_ptr : RegionName := "candidate_vdiv_17"
  let candidate_vdiv_18_ptr : RegionName := "candidate_vdiv_18"
  let candidate_vdiv_19_ptr : RegionName := "candidate_vdiv_19"
  let candidate_vdiv_20_ptr : RegionName := "candidate_vdiv_20"
  let candidate_vdiv_21_ptr : RegionName := "candidate_vdiv_21"
  let candidate_vdiv_22_ptr : RegionName := "candidate_vdiv_22"
  let candidate_vdiv_23_ptr : RegionName := "candidate_vdiv_23"
  let candidate_vdiv_24_ptr : RegionName := "candidate_vdiv_24"
  let candidate_vdiv_25_ptr : RegionName := "candidate_vdiv_25"
  let candidate_vdiv_26_ptr : RegionName := "candidate_vdiv_26"
  let candidate_vdiv_27_ptr : RegionName := "candidate_vdiv_27"
  let candidate_vdiv_28_ptr : RegionName := "candidate_vdiv_28"
  let candidate_vdiv_29_ptr : RegionName := "candidate_vdiv_29"
  let candidate_vdiv_30_ptr : RegionName := "candidate_vdiv_30"
  let candidate_vdiv_31_ptr : RegionName := "candidate_vdiv_31"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vadd_25_ptr : RegionName := "candidate_vadd_25"
  let candidate_vadd_26_ptr : RegionName := "candidate_vadd_26"
  let candidate_vadd_27_ptr : RegionName := "candidate_vadd_27"
  let candidate_vadd_28_ptr : RegionName := "candidate_vadd_28"
  let candidate_vadd_29_ptr : RegionName := "candidate_vadd_29"
  let candidate_vadd_30_ptr : RegionName := "candidate_vadd_30"
  let candidate_vadd_31_ptr : RegionName := "candidate_vadd_31"
  let candidate_vaddcdiv_ptr : RegionName := "candidate_vaddcdiv"
  let candidate_vaddcdiv_1_ptr : RegionName := "candidate_vaddcdiv_1"
  let candidate_vaddcdiv_2_ptr : RegionName := "candidate_vaddcdiv_2"
  let candidate_vaddcdiv_3_ptr : RegionName := "candidate_vaddcdiv_3"
  let candidate_vaddcdiv_4_ptr : RegionName := "candidate_vaddcdiv_4"
  let candidate_vaddcdiv_5_ptr : RegionName := "candidate_vaddcdiv_5"
  let candidate_vaddcdiv_6_ptr : RegionName := "candidate_vaddcdiv_6"
  let candidate_vaddcdiv_7_ptr : RegionName := "candidate_vaddcdiv_7"
  let candidate_vaddcdiv_8_ptr : RegionName := "candidate_vaddcdiv_8"
  let candidate_vaddcdiv_9_ptr : RegionName := "candidate_vaddcdiv_9"
  let candidate_vaddcdiv_10_ptr : RegionName := "candidate_vaddcdiv_10"
  let candidate_vaddcdiv_11_ptr : RegionName := "candidate_vaddcdiv_11"
  let candidate_vaddcdiv_12_ptr : RegionName := "candidate_vaddcdiv_12"
  let candidate_vaddcdiv_13_ptr : RegionName := "candidate_vaddcdiv_13"
  let candidate_vaddcdiv_14_ptr : RegionName := "candidate_vaddcdiv_14"
  let candidate_vaddcdiv_15_ptr : RegionName := "candidate_vaddcdiv_15"
  let candidate_vaddcdiv_16_ptr : RegionName := "candidate_vaddcdiv_16"
  let candidate_vaddcdiv_17_ptr : RegionName := "candidate_vaddcdiv_17"
  let candidate_vaddcdiv_18_ptr : RegionName := "candidate_vaddcdiv_18"
  let candidate_vaddcdiv_19_ptr : RegionName := "candidate_vaddcdiv_19"
  let candidate_vaddcdiv_20_ptr : RegionName := "candidate_vaddcdiv_20"
  let candidate_vaddcdiv_21_ptr : RegionName := "candidate_vaddcdiv_21"
  let candidate_vaddcdiv_22_ptr : RegionName := "candidate_vaddcdiv_22"
  let candidate_vaddcdiv_23_ptr : RegionName := "candidate_vaddcdiv_23"
  let candidate_vaddcdiv_24_ptr : RegionName := "candidate_vaddcdiv_24"
  let candidate_vaddcdiv_25_ptr : RegionName := "candidate_vaddcdiv_25"
  let candidate_vaddcdiv_26_ptr : RegionName := "candidate_vaddcdiv_26"
  let candidate_vaddcdiv_27_ptr : RegionName := "candidate_vaddcdiv_27"
  let candidate_vaddcdiv_28_ptr : RegionName := "candidate_vaddcdiv_28"
  let candidate_vaddcdiv_29_ptr : RegionName := "candidate_vaddcdiv_29"
  let candidate_vaddcdiv_30_ptr : RegionName := "candidate_vaddcdiv_30"
  let candidate_vaddcdiv_31_ptr : RegionName := "candidate_vaddcdiv_31"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  triton {
    r0_z = tl.arange($(512))
    r0_value = tl.load($(input0_ptr) + r0_z, dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(512)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(512))
    r1_value = tl.load($(input0_ptr) + (r1_z + $(512)), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(512)])
    tl.store($(candidate_vto_copy_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = tl.load($(input0_ptr) + (r2_z + $(1024)), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(candidate_vto_copy_2_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(512))
    r3_value = tl.load($(input0_ptr) + (r3_z + $(1536)), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(512)])
    tl.store($(candidate_vto_copy_3_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(512))
    r4_value = tl.load($(input0_ptr) + (r4_z + $(2048)), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(512)])
    tl.store($(candidate_vto_copy_4_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = tl.load($(input0_ptr) + (r5_z + $(2560)), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(candidate_vto_copy_5_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(512))
    r6_value = tl.load($(input0_ptr) + (r6_z + $(3072)), dtype=tl.bfloat16)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(512)])
    tl.store($(candidate_vto_copy_6_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(512))
    r7_value = tl.load($(input0_ptr) + (r7_z + $(3584)), dtype=tl.bfloat16)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(512)])
    tl.store($(candidate_vto_copy_7_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = tl.load($(input0_ptr) + (r8_z + $(4096)), dtype=tl.bfloat16)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(candidate_vto_copy_8_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(512))
    r9_value = tl.load($(input0_ptr) + (r9_z + $(4608)), dtype=tl.bfloat16)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(512)])
    tl.store($(candidate_vto_copy_9_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_value = tl.load($(input0_ptr) + (r10_z + $(5120)), dtype=tl.bfloat16)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(candidate_vto_copy_10_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(512))
    r11_value = tl.load($(input0_ptr) + (r11_z + $(5632)), dtype=tl.bfloat16)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(512)])
    tl.store($(candidate_vto_copy_11_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_value = tl.load($(input0_ptr) + (r12_z + $(6144)), dtype=tl.bfloat16)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(candidate_vto_copy_12_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = tl.load($(input0_ptr) + (r13_z + $(6656)), dtype=tl.bfloat16)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(candidate_vto_copy_13_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(512))
    r14_value = tl.load($(input0_ptr) + (r14_z + $(7168)), dtype=tl.bfloat16)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(512)])
    tl.store($(candidate_vto_copy_14_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(512))
    r15_value = tl.load($(input0_ptr) + (r15_z + $(7680)), dtype=tl.bfloat16)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(candidate_vto_copy_15_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(512))
    r16_value = tl.load($(input0_ptr) + (r16_z + $(8192)), dtype=tl.bfloat16)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(512)])
    tl.store($(candidate_vto_copy_16_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(512))
    r17_value = tl.load($(input0_ptr) + (r17_z + $(8704)), dtype=tl.bfloat16)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(512)])
    tl.store($(candidate_vto_copy_17_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(512))
    r18_value = tl.load($(input0_ptr) + (r18_z + $(9216)), dtype=tl.bfloat16)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(512)])
    tl.store($(candidate_vto_copy_18_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(512))
    r19_value = tl.load($(input0_ptr) + (r19_z + $(9728)), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(512)])
    tl.store($(candidate_vto_copy_19_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(512))
    r20_value = tl.load($(input0_ptr) + (r20_z + $(10240)), dtype=tl.bfloat16)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(512)])
    tl.store($(candidate_vto_copy_20_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(512))
    r21_value = tl.load($(input0_ptr) + (r21_z + $(10752)), dtype=tl.bfloat16)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(512)])
    tl.store($(candidate_vto_copy_21_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(512))
    r22_value = tl.load($(input0_ptr) + (r22_z + $(11264)), dtype=tl.bfloat16)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(512)])
    tl.store($(candidate_vto_copy_22_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(512))
    r23_value = tl.load($(input0_ptr) + (r23_z + $(11776)), dtype=tl.bfloat16)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(512)])
    tl.store($(candidate_vto_copy_23_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(512))
    r24_value = tl.load($(input0_ptr) + (r24_z + $(12288)), dtype=tl.bfloat16)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(512)])
    tl.store($(candidate_vto_copy_24_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(512))
    r25_value = tl.load($(input0_ptr) + (r25_z + $(12800)), dtype=tl.bfloat16)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(512)])
    tl.store($(candidate_vto_copy_25_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(512))
    r26_value = tl.load($(input0_ptr) + (r26_z + $(13312)), dtype=tl.bfloat16)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(512)])
    tl.store($(candidate_vto_copy_26_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(512))
    r27_value = tl.load($(input0_ptr) + (r27_z + $(13824)), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(512)])
    tl.store($(candidate_vto_copy_27_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(512))
    r28_value = tl.load($(input0_ptr) + (r28_z + $(14336)), dtype=tl.bfloat16)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(512)])
    tl.store($(candidate_vto_copy_28_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(512))
    r29_value = tl.load($(input0_ptr) + (r29_z + $(14848)), dtype=tl.bfloat16)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(512)])
    tl.store($(candidate_vto_copy_29_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(512))
    r30_value = tl.load($(input0_ptr) + (r30_z + $(15360)), dtype=tl.bfloat16)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(512)])
    tl.store($(candidate_vto_copy_30_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(512))
    r31_value = tl.load($(input0_ptr) + (r31_z + $(15872)), dtype=tl.bfloat16)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(512)])
    tl.store($(candidate_vto_copy_31_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(512))
    r32_value = (tl.load($(parameter0_ptr) + r32_z, dtype=tl.float32) * 0.99999)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(512)])
    tl.store($(candidate_vmul_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(512))
    r33_value = (tl.load($(parameter0_ptr) + (r33_z + $(512)), dtype=tl.float32) * 0.99999)
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(512)])
    tl.store($(candidate_vmul_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(512))
    r34_value = (tl.load($(parameter0_ptr) + (r34_z + $(1024)), dtype=tl.float32) * 0.99999)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(512)])
    tl.store($(candidate_vmul_2_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(512))
    r35_value = (tl.load($(parameter0_ptr) + (r35_z + $(1536)), dtype=tl.float32) * 0.99999)
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(512)])
    tl.store($(candidate_vmul_3_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(512))
    r36_value = (tl.load($(parameter0_ptr) + (r36_z + $(2048)), dtype=tl.float32) * 0.99999)
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(512)])
    tl.store($(candidate_vmul_4_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(512))
    r37_value = (tl.load($(parameter0_ptr) + (r37_z + $(2560)), dtype=tl.float32) * 0.99999)
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(512)])
    tl.store($(candidate_vmul_5_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(512))
    r38_value = (tl.load($(parameter0_ptr) + (r38_z + $(3072)), dtype=tl.float32) * 0.99999)
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(512)])
    tl.store($(candidate_vmul_6_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(512))
    r39_value = (tl.load($(parameter0_ptr) + (r39_z + $(3584)), dtype=tl.float32) * 0.99999)
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(512)])
    tl.store($(candidate_vmul_7_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(512))
    r40_value = (tl.load($(parameter0_ptr) + (r40_z + $(4096)), dtype=tl.float32) * 0.99999)
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(512)])
    tl.store($(candidate_vmul_8_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(512))
    r41_value = (tl.load($(parameter0_ptr) + (r41_z + $(4608)), dtype=tl.float32) * 0.99999)
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(512)])
    tl.store($(candidate_vmul_9_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(512))
    r42_value = (tl.load($(parameter0_ptr) + (r42_z + $(5120)), dtype=tl.float32) * 0.99999)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(512)])
    tl.store($(candidate_vmul_10_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(512))
    r43_value = (tl.load($(parameter0_ptr) + (r43_z + $(5632)), dtype=tl.float32) * 0.99999)
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(512)])
    tl.store($(candidate_vmul_11_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(512))
    r44_value = (tl.load($(parameter0_ptr) + (r44_z + $(6144)), dtype=tl.float32) * 0.99999)
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(512)])
    tl.store($(candidate_vmul_12_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(512))
    r45_value = (tl.load($(parameter0_ptr) + (r45_z + $(6656)), dtype=tl.float32) * 0.99999)
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(512)])
    tl.store($(candidate_vmul_13_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(512))
    r46_value = (tl.load($(parameter0_ptr) + (r46_z + $(7168)), dtype=tl.float32) * 0.99999)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(512)])
    tl.store($(candidate_vmul_14_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(512))
    r47_value = (tl.load($(parameter0_ptr) + (r47_z + $(7680)), dtype=tl.float32) * 0.99999)
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(512)])
    tl.store($(candidate_vmul_15_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(512))
    r48_value = (tl.load($(parameter0_ptr) + (r48_z + $(8192)), dtype=tl.float32) * 0.99999)
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(512)])
    tl.store($(candidate_vmul_16_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(512))
    r49_value = (tl.load($(parameter0_ptr) + (r49_z + $(8704)), dtype=tl.float32) * 0.99999)
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(512)])
    tl.store($(candidate_vmul_17_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(512))
    r50_value = (tl.load($(parameter0_ptr) + (r50_z + $(9216)), dtype=tl.float32) * 0.99999)
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(512)])
    tl.store($(candidate_vmul_18_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(512))
    r51_value = (tl.load($(parameter0_ptr) + (r51_z + $(9728)), dtype=tl.float32) * 0.99999)
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(512)])
    tl.store($(candidate_vmul_19_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(512))
    r52_value = (tl.load($(parameter0_ptr) + (r52_z + $(10240)), dtype=tl.float32) * 0.99999)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(512)])
    tl.store($(candidate_vmul_20_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(512))
    r53_value = (tl.load($(parameter0_ptr) + (r53_z + $(10752)), dtype=tl.float32) * 0.99999)
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(512)])
    tl.store($(candidate_vmul_21_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(512))
    r54_value = (tl.load($(parameter0_ptr) + (r54_z + $(11264)), dtype=tl.float32) * 0.99999)
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(512)])
    tl.store($(candidate_vmul_22_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(512))
    r55_value = (tl.load($(parameter0_ptr) + (r55_z + $(11776)), dtype=tl.float32) * 0.99999)
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(512)])
    tl.store($(candidate_vmul_23_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(512))
    r56_value = (tl.load($(parameter0_ptr) + (r56_z + $(12288)), dtype=tl.float32) * 0.99999)
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(512)])
    tl.store($(candidate_vmul_24_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(512))
    r57_value = (tl.load($(parameter0_ptr) + (r57_z + $(12800)), dtype=tl.float32) * 0.99999)
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(512)])
    tl.store($(candidate_vmul_25_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(512))
    r58_value = (tl.load($(parameter0_ptr) + (r58_z + $(13312)), dtype=tl.float32) * 0.99999)
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(512)])
    tl.store($(candidate_vmul_26_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(512))
    r59_value = (tl.load($(parameter0_ptr) + (r59_z + $(13824)), dtype=tl.float32) * 0.99999)
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(512)])
    tl.store($(candidate_vmul_27_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(512))
    r60_value = (tl.load($(parameter0_ptr) + (r60_z + $(14336)), dtype=tl.float32) * 0.99999)
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(512)])
    tl.store($(candidate_vmul_28_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(512))
    r61_value = (tl.load($(parameter0_ptr) + (r61_z + $(14848)), dtype=tl.float32) * 0.99999)
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(512)])
    tl.store($(candidate_vmul_29_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(512))
    r62_value = (tl.load($(parameter0_ptr) + (r62_z + $(15360)), dtype=tl.float32) * 0.99999)
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(512)])
    tl.store($(candidate_vmul_30_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(512))
    r63_value = (tl.load($(parameter0_ptr) + (r63_z + $(15872)), dtype=tl.float32) * 0.99999)
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(512)])
    tl.store($(candidate_vmul_31_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(512))
    r64_value = (0.0 + (tl.load($(candidate_vto_copy_ptr) + r64_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(512)])
    tl.store($(candidate_vlerp_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(512))
    r65_value = (0.0 + (tl.load($(candidate_vto_copy_1_ptr) + r65_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(512)])
    tl.store($(candidate_vlerp_1_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(512))
    r66_value = (0.0 + (tl.load($(candidate_vto_copy_2_ptr) + r66_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(512)])
    tl.store($(candidate_vlerp_2_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(512))
    r67_value = (0.0 + (tl.load($(candidate_vto_copy_3_ptr) + r67_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(512)])
    tl.store($(candidate_vlerp_3_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(512))
    r68_value = (0.0 + (tl.load($(candidate_vto_copy_4_ptr) + r68_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(512)])
    tl.store($(candidate_vlerp_4_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(512))
    r69_value = (0.0 + (tl.load($(candidate_vto_copy_5_ptr) + r69_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(512)])
    tl.store($(candidate_vlerp_5_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(512))
    r70_value = (0.0 + (tl.load($(candidate_vto_copy_6_ptr) + r70_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(512)])
    tl.store($(candidate_vlerp_6_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(512))
    r71_value = (0.0 + (tl.load($(candidate_vto_copy_7_ptr) + r71_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(512)])
    tl.store($(candidate_vlerp_7_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(512))
    r72_value = (0.0 + (tl.load($(candidate_vto_copy_8_ptr) + r72_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(512)])
    tl.store($(candidate_vlerp_8_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(512))
    r73_value = (0.0 + (tl.load($(candidate_vto_copy_9_ptr) + r73_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(512)])
    tl.store($(candidate_vlerp_9_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(512))
    r74_value = (0.0 + (tl.load($(candidate_vto_copy_10_ptr) + r74_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(512)])
    tl.store($(candidate_vlerp_10_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(512))
    r75_value = (0.0 + (tl.load($(candidate_vto_copy_11_ptr) + r75_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(512)])
    tl.store($(candidate_vlerp_11_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(512))
    r76_value = (0.0 + (tl.load($(candidate_vto_copy_12_ptr) + r76_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(512)])
    tl.store($(candidate_vlerp_12_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(512))
    r77_value = (0.0 + (tl.load($(candidate_vto_copy_13_ptr) + r77_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(512)])
    tl.store($(candidate_vlerp_13_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(512))
    r78_value = (0.0 + (tl.load($(candidate_vto_copy_14_ptr) + r78_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(512)])
    tl.store($(candidate_vlerp_14_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(512))
    r79_value = (0.0 + (tl.load($(candidate_vto_copy_15_ptr) + r79_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(512)])
    tl.store($(candidate_vlerp_15_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(512))
    r80_value = (0.0 + (tl.load($(candidate_vto_copy_16_ptr) + r80_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(512)])
    tl.store($(candidate_vlerp_16_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(512))
    r81_value = (0.0 + (tl.load($(candidate_vto_copy_17_ptr) + r81_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(512)])
    tl.store($(candidate_vlerp_17_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(512))
    r82_value = (0.0 + (tl.load($(candidate_vto_copy_18_ptr) + r82_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(512)])
    tl.store($(candidate_vlerp_18_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(512))
    r83_value = (0.0 + (tl.load($(candidate_vto_copy_19_ptr) + r83_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(512)])
    tl.store($(candidate_vlerp_19_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(512))
    r84_value = (0.0 + (tl.load($(candidate_vto_copy_20_ptr) + r84_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(512)])
    tl.store($(candidate_vlerp_20_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(512))
    r85_value = (0.0 + (tl.load($(candidate_vto_copy_21_ptr) + r85_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(512)])
    tl.store($(candidate_vlerp_21_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(512))
    r86_value = (0.0 + (tl.load($(candidate_vto_copy_22_ptr) + r86_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(512)])
    tl.store($(candidate_vlerp_22_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(512))
    r87_value = (0.0 + (tl.load($(candidate_vto_copy_23_ptr) + r87_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(512)])
    tl.store($(candidate_vlerp_23_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(512))
    r88_value = (0.0 + (tl.load($(candidate_vto_copy_24_ptr) + r88_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(512)])
    tl.store($(candidate_vlerp_24_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(512))
    r89_value = (0.0 + (tl.load($(candidate_vto_copy_25_ptr) + r89_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(512)])
    tl.store($(candidate_vlerp_25_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(512))
    r90_value = (0.0 + (tl.load($(candidate_vto_copy_26_ptr) + r90_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(512)])
    tl.store($(candidate_vlerp_26_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(512))
    r91_value = (0.0 + (tl.load($(candidate_vto_copy_27_ptr) + r91_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(512)])
    tl.store($(candidate_vlerp_27_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(512))
    r92_value = (0.0 + (tl.load($(candidate_vto_copy_28_ptr) + r92_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(512)])
    tl.store($(candidate_vlerp_28_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(512))
    r93_value = (0.0 + (tl.load($(candidate_vto_copy_29_ptr) + r93_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(512)])
    tl.store($(candidate_vlerp_29_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(512))
    r94_value = (0.0 + (tl.load($(candidate_vto_copy_30_ptr) + r94_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(512)])
    tl.store($(candidate_vlerp_30_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(512))
    r95_value = (0.0 + (tl.load($(candidate_vto_copy_31_ptr) + r95_z, dtype=tl.float32) - 0.0) * 0.09999999999999998)
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(512)])
    tl.store($(candidate_vlerp_31_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(512))
    r96_value = (0.0 * 0.999)
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(512)])
    tl.store($(candidate_vmul_32_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(512))
    r97_value = (0.0 * 0.999)
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(512)])
    tl.store($(candidate_vmul_33_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(512))
    r98_value = (0.0 * 0.999)
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(512)])
    tl.store($(candidate_vmul_34_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(512))
    r99_value = (0.0 * 0.999)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(512)])
    tl.store($(candidate_vmul_35_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(512))
    r100_value = (0.0 * 0.999)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(512)])
    tl.store($(candidate_vmul_36_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(512))
    r101_value = (0.0 * 0.999)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(512)])
    tl.store($(candidate_vmul_37_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(512))
    r102_value = (0.0 * 0.999)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(512)])
    tl.store($(candidate_vmul_38_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(512))
    r103_value = (0.0 * 0.999)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(512)])
    tl.store($(candidate_vmul_39_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(512))
    r104_value = (0.0 * 0.999)
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(512)])
    tl.store($(candidate_vmul_40_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(512))
    r105_value = (0.0 * 0.999)
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(512)])
    tl.store($(candidate_vmul_41_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(512))
    r106_value = (0.0 * 0.999)
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(512)])
    tl.store($(candidate_vmul_42_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(512))
    r107_value = (0.0 * 0.999)
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(512)])
    tl.store($(candidate_vmul_43_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(512))
    r108_value = (0.0 * 0.999)
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(512)])
    tl.store($(candidate_vmul_44_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(512))
    r109_value = (0.0 * 0.999)
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(512)])
    tl.store($(candidate_vmul_45_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(512))
    r110_value = (0.0 * 0.999)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(512)])
    tl.store($(candidate_vmul_46_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(512))
    r111_value = (0.0 * 0.999)
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(512)])
    tl.store($(candidate_vmul_47_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(512))
    r112_value = (0.0 * 0.999)
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(512)])
    tl.store($(candidate_vmul_48_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(512))
    r113_value = (0.0 * 0.999)
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(512)])
    tl.store($(candidate_vmul_49_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(512))
    r114_value = (0.0 * 0.999)
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(512)])
    tl.store($(candidate_vmul_50_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(512))
    r115_value = (0.0 * 0.999)
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(512)])
    tl.store($(candidate_vmul_51_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(512))
    r116_value = (0.0 * 0.999)
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(512)])
    tl.store($(candidate_vmul_52_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(512))
    r117_value = (0.0 * 0.999)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(512)])
    tl.store($(candidate_vmul_53_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(512))
    r118_value = (0.0 * 0.999)
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(512)])
    tl.store($(candidate_vmul_54_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(512))
    r119_value = (0.0 * 0.999)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(512)])
    tl.store($(candidate_vmul_55_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(512))
    r120_value = (0.0 * 0.999)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(512)])
    tl.store($(candidate_vmul_56_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(512))
    r121_value = (0.0 * 0.999)
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(512)])
    tl.store($(candidate_vmul_57_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(512))
    r122_value = (0.0 * 0.999)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(512)])
    tl.store($(candidate_vmul_58_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(512))
    r123_value = (0.0 * 0.999)
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(512)])
    tl.store($(candidate_vmul_59_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(512))
    r124_value = (0.0 * 0.999)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(512)])
    tl.store($(candidate_vmul_60_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(512))
    r125_value = (0.0 * 0.999)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(512)])
    tl.store($(candidate_vmul_61_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(512))
    r126_value = (0.0 * 0.999)
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(512)])
    tl.store($(candidate_vmul_62_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(512))
    r127_value = (0.0 * 0.999)
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(512)])
    tl.store($(candidate_vmul_63_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(512))
    r128_value = (tl.load($(candidate_vmul_32_ptr) + r128_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_ptr) + r128_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + r128_z, dtype=tl.float32))
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(512)])
    tl.store($(candidate_vaddcmul_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(512))
    r129_value = (tl.load($(candidate_vmul_33_ptr) + r129_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_1_ptr) + r129_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_1_ptr) + r129_z, dtype=tl.float32))
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(512)])
    tl.store($(candidate_vaddcmul_1_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(512))
    r130_value = (tl.load($(candidate_vmul_34_ptr) + r130_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_2_ptr) + r130_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_2_ptr) + r130_z, dtype=tl.float32))
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(512)])
    tl.store($(candidate_vaddcmul_2_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(512))
    r131_value = (tl.load($(candidate_vmul_35_ptr) + r131_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_3_ptr) + r131_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_3_ptr) + r131_z, dtype=tl.float32))
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(512)])
    tl.store($(candidate_vaddcmul_3_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(512))
    r132_value = (tl.load($(candidate_vmul_36_ptr) + r132_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_4_ptr) + r132_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_4_ptr) + r132_z, dtype=tl.float32))
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(512)])
    tl.store($(candidate_vaddcmul_4_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(512))
    r133_value = (tl.load($(candidate_vmul_37_ptr) + r133_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_5_ptr) + r133_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_5_ptr) + r133_z, dtype=tl.float32))
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(512)])
    tl.store($(candidate_vaddcmul_5_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(512))
    r134_value = (tl.load($(candidate_vmul_38_ptr) + r134_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_6_ptr) + r134_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_6_ptr) + r134_z, dtype=tl.float32))
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(512)])
    tl.store($(candidate_vaddcmul_6_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(512))
    r135_value = (tl.load($(candidate_vmul_39_ptr) + r135_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_7_ptr) + r135_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_7_ptr) + r135_z, dtype=tl.float32))
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(512)])
    tl.store($(candidate_vaddcmul_7_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(512))
    r136_value = (tl.load($(candidate_vmul_40_ptr) + r136_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_8_ptr) + r136_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_8_ptr) + r136_z, dtype=tl.float32))
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(512)])
    tl.store($(candidate_vaddcmul_8_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(512))
    r137_value = (tl.load($(candidate_vmul_41_ptr) + r137_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_9_ptr) + r137_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_9_ptr) + r137_z, dtype=tl.float32))
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(512)])
    tl.store($(candidate_vaddcmul_9_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(512))
    r138_value = (tl.load($(candidate_vmul_42_ptr) + r138_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_10_ptr) + r138_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_10_ptr) + r138_z, dtype=tl.float32))
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(512)])
    tl.store($(candidate_vaddcmul_10_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(512))
    r139_value = (tl.load($(candidate_vmul_43_ptr) + r139_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_11_ptr) + r139_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_11_ptr) + r139_z, dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(512)])
    tl.store($(candidate_vaddcmul_11_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(512))
    r140_value = (tl.load($(candidate_vmul_44_ptr) + r140_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_12_ptr) + r140_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_12_ptr) + r140_z, dtype=tl.float32))
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(512)])
    tl.store($(candidate_vaddcmul_12_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(512))
    r141_value = (tl.load($(candidate_vmul_45_ptr) + r141_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_13_ptr) + r141_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_13_ptr) + r141_z, dtype=tl.float32))
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(512)])
    tl.store($(candidate_vaddcmul_13_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(512))
    r142_value = (tl.load($(candidate_vmul_46_ptr) + r142_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_14_ptr) + r142_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_14_ptr) + r142_z, dtype=tl.float32))
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(512)])
    tl.store($(candidate_vaddcmul_14_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(512))
    r143_value = (tl.load($(candidate_vmul_47_ptr) + r143_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_15_ptr) + r143_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_15_ptr) + r143_z, dtype=tl.float32))
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(512)])
    tl.store($(candidate_vaddcmul_15_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(512))
    r144_value = (tl.load($(candidate_vmul_48_ptr) + r144_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_16_ptr) + r144_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_16_ptr) + r144_z, dtype=tl.float32))
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(512)])
    tl.store($(candidate_vaddcmul_16_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(512))
    r145_value = (tl.load($(candidate_vmul_49_ptr) + r145_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_17_ptr) + r145_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_17_ptr) + r145_z, dtype=tl.float32))
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(512)])
    tl.store($(candidate_vaddcmul_17_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(512))
    r146_value = (tl.load($(candidate_vmul_50_ptr) + r146_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_18_ptr) + r146_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_18_ptr) + r146_z, dtype=tl.float32))
    r146_value = r146_value.to(tl.float32)
    r146_value = tl.broadcast_to(r146_value, [$(512)])
    tl.store($(candidate_vaddcmul_18_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(512))
    r147_value = (tl.load($(candidate_vmul_51_ptr) + r147_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_19_ptr) + r147_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_19_ptr) + r147_z, dtype=tl.float32))
    r147_value = r147_value.to(tl.float32)
    r147_value = tl.broadcast_to(r147_value, [$(512)])
    tl.store($(candidate_vaddcmul_19_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(512))
    r148_value = (tl.load($(candidate_vmul_52_ptr) + r148_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_20_ptr) + r148_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + r148_z, dtype=tl.float32))
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(512)])
    tl.store($(candidate_vaddcmul_20_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(512))
    r149_value = (tl.load($(candidate_vmul_53_ptr) + r149_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_21_ptr) + r149_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_21_ptr) + r149_z, dtype=tl.float32))
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(512)])
    tl.store($(candidate_vaddcmul_21_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(512))
    r150_value = (tl.load($(candidate_vmul_54_ptr) + r150_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_22_ptr) + r150_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_22_ptr) + r150_z, dtype=tl.float32))
    r150_value = r150_value.to(tl.float32)
    r150_value = tl.broadcast_to(r150_value, [$(512)])
    tl.store($(candidate_vaddcmul_22_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(512))
    r151_value = (tl.load($(candidate_vmul_55_ptr) + r151_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_23_ptr) + r151_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_23_ptr) + r151_z, dtype=tl.float32))
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(512)])
    tl.store($(candidate_vaddcmul_23_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(512))
    r152_value = (tl.load($(candidate_vmul_56_ptr) + r152_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_24_ptr) + r152_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_24_ptr) + r152_z, dtype=tl.float32))
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(512)])
    tl.store($(candidate_vaddcmul_24_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(512))
    r153_value = (tl.load($(candidate_vmul_57_ptr) + r153_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_25_ptr) + r153_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_25_ptr) + r153_z, dtype=tl.float32))
    r153_value = r153_value.to(tl.float32)
    r153_value = tl.broadcast_to(r153_value, [$(512)])
    tl.store($(candidate_vaddcmul_25_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(512))
    r154_value = (tl.load($(candidate_vmul_58_ptr) + r154_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_26_ptr) + r154_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_26_ptr) + r154_z, dtype=tl.float32))
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(512)])
    tl.store($(candidate_vaddcmul_26_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(512))
    r155_value = (tl.load($(candidate_vmul_59_ptr) + r155_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_27_ptr) + r155_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_27_ptr) + r155_z, dtype=tl.float32))
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(512)])
    tl.store($(candidate_vaddcmul_27_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(512))
    r156_value = (tl.load($(candidate_vmul_60_ptr) + r156_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_28_ptr) + r156_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_28_ptr) + r156_z, dtype=tl.float32))
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(512)])
    tl.store($(candidate_vaddcmul_28_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(512))
    r157_value = (tl.load($(candidate_vmul_61_ptr) + r157_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_29_ptr) + r157_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_29_ptr) + r157_z, dtype=tl.float32))
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(512)])
    tl.store($(candidate_vaddcmul_29_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(512))
    r158_value = (tl.load($(candidate_vmul_62_ptr) + r158_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_30_ptr) + r158_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_30_ptr) + r158_z, dtype=tl.float32))
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(512)])
    tl.store($(candidate_vaddcmul_30_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(512))
    r159_value = (tl.load($(candidate_vmul_63_ptr) + r159_z, dtype=tl.float32) + 0.0010000000000000009 * tl.load($(candidate_vto_copy_31_ptr) + r159_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_31_ptr) + r159_z, dtype=tl.float32))
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(512)])
    tl.store($(candidate_vaddcmul_31_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(512))
    r160_value = tl.sqrt(tl.load($(candidate_vaddcmul_ptr) + r160_z, dtype=tl.float32))
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(512)])
    tl.store($(candidate_vsqrt_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(512))
    r161_value = tl.sqrt(tl.load($(candidate_vaddcmul_1_ptr) + r161_z, dtype=tl.float32))
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(512)])
    tl.store($(candidate_vsqrt_1_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(512))
    r162_value = tl.sqrt(tl.load($(candidate_vaddcmul_2_ptr) + r162_z, dtype=tl.float32))
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(512)])
    tl.store($(candidate_vsqrt_2_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(512))
    r163_value = tl.sqrt(tl.load($(candidate_vaddcmul_3_ptr) + r163_z, dtype=tl.float32))
    r163_value = r163_value.to(tl.float32)
    r163_value = tl.broadcast_to(r163_value, [$(512)])
    tl.store($(candidate_vsqrt_3_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(512))
    r164_value = tl.sqrt(tl.load($(candidate_vaddcmul_4_ptr) + r164_z, dtype=tl.float32))
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(512)])
    tl.store($(candidate_vsqrt_4_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(512))
    r165_value = tl.sqrt(tl.load($(candidate_vaddcmul_5_ptr) + r165_z, dtype=tl.float32))
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(512)])
    tl.store($(candidate_vsqrt_5_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(512))
    r166_value = tl.sqrt(tl.load($(candidate_vaddcmul_6_ptr) + r166_z, dtype=tl.float32))
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(512)])
    tl.store($(candidate_vsqrt_6_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(512))
    r167_value = tl.sqrt(tl.load($(candidate_vaddcmul_7_ptr) + r167_z, dtype=tl.float32))
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(512)])
    tl.store($(candidate_vsqrt_7_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(512))
    r168_value = tl.sqrt(tl.load($(candidate_vaddcmul_8_ptr) + r168_z, dtype=tl.float32))
    r168_value = r168_value.to(tl.float32)
    r168_value = tl.broadcast_to(r168_value, [$(512)])
    tl.store($(candidate_vsqrt_8_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(512))
    r169_value = tl.sqrt(tl.load($(candidate_vaddcmul_9_ptr) + r169_z, dtype=tl.float32))
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(512)])
    tl.store($(candidate_vsqrt_9_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(512))
    r170_value = tl.sqrt(tl.load($(candidate_vaddcmul_10_ptr) + r170_z, dtype=tl.float32))
    r170_value = r170_value.to(tl.float32)
    r170_value = tl.broadcast_to(r170_value, [$(512)])
    tl.store($(candidate_vsqrt_10_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(512))
    r171_value = tl.sqrt(tl.load($(candidate_vaddcmul_11_ptr) + r171_z, dtype=tl.float32))
    r171_value = r171_value.to(tl.float32)
    r171_value = tl.broadcast_to(r171_value, [$(512)])
    tl.store($(candidate_vsqrt_11_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(512))
    r172_value = tl.sqrt(tl.load($(candidate_vaddcmul_12_ptr) + r172_z, dtype=tl.float32))
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(512)])
    tl.store($(candidate_vsqrt_12_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(512))
    r173_value = tl.sqrt(tl.load($(candidate_vaddcmul_13_ptr) + r173_z, dtype=tl.float32))
    r173_value = r173_value.to(tl.float32)
    r173_value = tl.broadcast_to(r173_value, [$(512)])
    tl.store($(candidate_vsqrt_13_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(512))
    r174_value = tl.sqrt(tl.load($(candidate_vaddcmul_14_ptr) + r174_z, dtype=tl.float32))
    r174_value = r174_value.to(tl.float32)
    r174_value = tl.broadcast_to(r174_value, [$(512)])
    tl.store($(candidate_vsqrt_14_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(512))
    r175_value = tl.sqrt(tl.load($(candidate_vaddcmul_15_ptr) + r175_z, dtype=tl.float32))
    r175_value = r175_value.to(tl.float32)
    r175_value = tl.broadcast_to(r175_value, [$(512)])
    tl.store($(candidate_vsqrt_15_ptr) + r175_z, r175_value)
    r176_z = tl.arange($(512))
    r176_value = tl.sqrt(tl.load($(candidate_vaddcmul_16_ptr) + r176_z, dtype=tl.float32))
    r176_value = r176_value.to(tl.float32)
    r176_value = tl.broadcast_to(r176_value, [$(512)])
    tl.store($(candidate_vsqrt_16_ptr) + r176_z, r176_value)
    r177_z = tl.arange($(512))
    r177_value = tl.sqrt(tl.load($(candidate_vaddcmul_17_ptr) + r177_z, dtype=tl.float32))
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(512)])
    tl.store($(candidate_vsqrt_17_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(512))
    r178_value = tl.sqrt(tl.load($(candidate_vaddcmul_18_ptr) + r178_z, dtype=tl.float32))
    r178_value = r178_value.to(tl.float32)
    r178_value = tl.broadcast_to(r178_value, [$(512)])
    tl.store($(candidate_vsqrt_18_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(512))
    r179_value = tl.sqrt(tl.load($(candidate_vaddcmul_19_ptr) + r179_z, dtype=tl.float32))
    r179_value = r179_value.to(tl.float32)
    r179_value = tl.broadcast_to(r179_value, [$(512)])
    tl.store($(candidate_vsqrt_19_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(512))
    r180_value = tl.sqrt(tl.load($(candidate_vaddcmul_20_ptr) + r180_z, dtype=tl.float32))
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(512)])
    tl.store($(candidate_vsqrt_20_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(512))
    r181_value = tl.sqrt(tl.load($(candidate_vaddcmul_21_ptr) + r181_z, dtype=tl.float32))
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(512)])
    tl.store($(candidate_vsqrt_21_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(512))
    r182_value = tl.sqrt(tl.load($(candidate_vaddcmul_22_ptr) + r182_z, dtype=tl.float32))
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(512)])
    tl.store($(candidate_vsqrt_22_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(512))
    r183_value = tl.sqrt(tl.load($(candidate_vaddcmul_23_ptr) + r183_z, dtype=tl.float32))
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(512)])
    tl.store($(candidate_vsqrt_23_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(512))
    r184_value = tl.sqrt(tl.load($(candidate_vaddcmul_24_ptr) + r184_z, dtype=tl.float32))
    r184_value = r184_value.to(tl.float32)
    r184_value = tl.broadcast_to(r184_value, [$(512)])
    tl.store($(candidate_vsqrt_24_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(512))
    r185_value = tl.sqrt(tl.load($(candidate_vaddcmul_25_ptr) + r185_z, dtype=tl.float32))
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(512)])
    tl.store($(candidate_vsqrt_25_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(512))
    r186_value = tl.sqrt(tl.load($(candidate_vaddcmul_26_ptr) + r186_z, dtype=tl.float32))
    r186_value = r186_value.to(tl.float32)
    r186_value = tl.broadcast_to(r186_value, [$(512)])
    tl.store($(candidate_vsqrt_26_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(512))
    r187_value = tl.sqrt(tl.load($(candidate_vaddcmul_27_ptr) + r187_z, dtype=tl.float32))
    r187_value = r187_value.to(tl.float32)
    r187_value = tl.broadcast_to(r187_value, [$(512)])
    tl.store($(candidate_vsqrt_27_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(512))
    r188_value = tl.sqrt(tl.load($(candidate_vaddcmul_28_ptr) + r188_z, dtype=tl.float32))
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(512)])
    tl.store($(candidate_vsqrt_28_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(512))
    r189_value = tl.sqrt(tl.load($(candidate_vaddcmul_29_ptr) + r189_z, dtype=tl.float32))
    r189_value = r189_value.to(tl.float32)
    r189_value = tl.broadcast_to(r189_value, [$(512)])
    tl.store($(candidate_vsqrt_29_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(512))
    r190_value = tl.sqrt(tl.load($(candidate_vaddcmul_30_ptr) + r190_z, dtype=tl.float32))
    r190_value = r190_value.to(tl.float32)
    r190_value = tl.broadcast_to(r190_value, [$(512)])
    tl.store($(candidate_vsqrt_30_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(512))
    r191_value = tl.sqrt(tl.load($(candidate_vaddcmul_31_ptr) + r191_z, dtype=tl.float32))
    r191_value = r191_value.to(tl.float32)
    r191_value = tl.broadcast_to(r191_value, [$(512)])
    tl.store($(candidate_vsqrt_31_ptr) + r191_z, r191_value)
    r192_z = tl.arange($(512))
    r192_value = (tl.load($(candidate_vsqrt_ptr) + r192_z, dtype=tl.float32) / 0.031622776601683805)
    r192_value = r192_value.to(tl.float32)
    r192_value = tl.broadcast_to(r192_value, [$(512)])
    tl.store($(candidate_vdiv_ptr) + r192_z, r192_value)
    r193_z = tl.arange($(512))
    r193_value = (tl.load($(candidate_vsqrt_1_ptr) + r193_z, dtype=tl.float32) / 0.031622776601683805)
    r193_value = r193_value.to(tl.float32)
    r193_value = tl.broadcast_to(r193_value, [$(512)])
    tl.store($(candidate_vdiv_1_ptr) + r193_z, r193_value)
    r194_z = tl.arange($(512))
    r194_value = (tl.load($(candidate_vsqrt_2_ptr) + r194_z, dtype=tl.float32) / 0.031622776601683805)
    r194_value = r194_value.to(tl.float32)
    r194_value = tl.broadcast_to(r194_value, [$(512)])
    tl.store($(candidate_vdiv_2_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(512))
    r195_value = (tl.load($(candidate_vsqrt_3_ptr) + r195_z, dtype=tl.float32) / 0.031622776601683805)
    r195_value = r195_value.to(tl.float32)
    r195_value = tl.broadcast_to(r195_value, [$(512)])
    tl.store($(candidate_vdiv_3_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(512))
    r196_value = (tl.load($(candidate_vsqrt_4_ptr) + r196_z, dtype=tl.float32) / 0.031622776601683805)
    r196_value = r196_value.to(tl.float32)
    r196_value = tl.broadcast_to(r196_value, [$(512)])
    tl.store($(candidate_vdiv_4_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(512))
    r197_value = (tl.load($(candidate_vsqrt_5_ptr) + r197_z, dtype=tl.float32) / 0.031622776601683805)
    r197_value = r197_value.to(tl.float32)
    r197_value = tl.broadcast_to(r197_value, [$(512)])
    tl.store($(candidate_vdiv_5_ptr) + r197_z, r197_value)
    r198_z = tl.arange($(512))
    r198_value = (tl.load($(candidate_vsqrt_6_ptr) + r198_z, dtype=tl.float32) / 0.031622776601683805)
    r198_value = r198_value.to(tl.float32)
    r198_value = tl.broadcast_to(r198_value, [$(512)])
    tl.store($(candidate_vdiv_6_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(512))
    r199_value = (tl.load($(candidate_vsqrt_7_ptr) + r199_z, dtype=tl.float32) / 0.031622776601683805)
    r199_value = r199_value.to(tl.float32)
    r199_value = tl.broadcast_to(r199_value, [$(512)])
    tl.store($(candidate_vdiv_7_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(512))
    r200_value = (tl.load($(candidate_vsqrt_8_ptr) + r200_z, dtype=tl.float32) / 0.031622776601683805)
    r200_value = r200_value.to(tl.float32)
    r200_value = tl.broadcast_to(r200_value, [$(512)])
    tl.store($(candidate_vdiv_8_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(512))
    r201_value = (tl.load($(candidate_vsqrt_9_ptr) + r201_z, dtype=tl.float32) / 0.031622776601683805)
    r201_value = r201_value.to(tl.float32)
    r201_value = tl.broadcast_to(r201_value, [$(512)])
    tl.store($(candidate_vdiv_9_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(512))
    r202_value = (tl.load($(candidate_vsqrt_10_ptr) + r202_z, dtype=tl.float32) / 0.031622776601683805)
    r202_value = r202_value.to(tl.float32)
    r202_value = tl.broadcast_to(r202_value, [$(512)])
    tl.store($(candidate_vdiv_10_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(512))
    r203_value = (tl.load($(candidate_vsqrt_11_ptr) + r203_z, dtype=tl.float32) / 0.031622776601683805)
    r203_value = r203_value.to(tl.float32)
    r203_value = tl.broadcast_to(r203_value, [$(512)])
    tl.store($(candidate_vdiv_11_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(512))
    r204_value = (tl.load($(candidate_vsqrt_12_ptr) + r204_z, dtype=tl.float32) / 0.031622776601683805)
    r204_value = r204_value.to(tl.float32)
    r204_value = tl.broadcast_to(r204_value, [$(512)])
    tl.store($(candidate_vdiv_12_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(512))
    r205_value = (tl.load($(candidate_vsqrt_13_ptr) + r205_z, dtype=tl.float32) / 0.031622776601683805)
    r205_value = r205_value.to(tl.float32)
    r205_value = tl.broadcast_to(r205_value, [$(512)])
    tl.store($(candidate_vdiv_13_ptr) + r205_z, r205_value)
    r206_z = tl.arange($(512))
    r206_value = (tl.load($(candidate_vsqrt_14_ptr) + r206_z, dtype=tl.float32) / 0.031622776601683805)
    r206_value = r206_value.to(tl.float32)
    r206_value = tl.broadcast_to(r206_value, [$(512)])
    tl.store($(candidate_vdiv_14_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(512))
    r207_value = (tl.load($(candidate_vsqrt_15_ptr) + r207_z, dtype=tl.float32) / 0.031622776601683805)
    r207_value = r207_value.to(tl.float32)
    r207_value = tl.broadcast_to(r207_value, [$(512)])
    tl.store($(candidate_vdiv_15_ptr) + r207_z, r207_value)
    r208_z = tl.arange($(512))
    r208_value = (tl.load($(candidate_vsqrt_16_ptr) + r208_z, dtype=tl.float32) / 0.031622776601683805)
    r208_value = r208_value.to(tl.float32)
    r208_value = tl.broadcast_to(r208_value, [$(512)])
    tl.store($(candidate_vdiv_16_ptr) + r208_z, r208_value)
    r209_z = tl.arange($(512))
    r209_value = (tl.load($(candidate_vsqrt_17_ptr) + r209_z, dtype=tl.float32) / 0.031622776601683805)
    r209_value = r209_value.to(tl.float32)
    r209_value = tl.broadcast_to(r209_value, [$(512)])
    tl.store($(candidate_vdiv_17_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(512))
    r210_value = (tl.load($(candidate_vsqrt_18_ptr) + r210_z, dtype=tl.float32) / 0.031622776601683805)
    r210_value = r210_value.to(tl.float32)
    r210_value = tl.broadcast_to(r210_value, [$(512)])
    tl.store($(candidate_vdiv_18_ptr) + r210_z, r210_value)
    r211_z = tl.arange($(512))
    r211_value = (tl.load($(candidate_vsqrt_19_ptr) + r211_z, dtype=tl.float32) / 0.031622776601683805)
    r211_value = r211_value.to(tl.float32)
    r211_value = tl.broadcast_to(r211_value, [$(512)])
    tl.store($(candidate_vdiv_19_ptr) + r211_z, r211_value)
    r212_z = tl.arange($(512))
    r212_value = (tl.load($(candidate_vsqrt_20_ptr) + r212_z, dtype=tl.float32) / 0.031622776601683805)
    r212_value = r212_value.to(tl.float32)
    r212_value = tl.broadcast_to(r212_value, [$(512)])
    tl.store($(candidate_vdiv_20_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(512))
    r213_value = (tl.load($(candidate_vsqrt_21_ptr) + r213_z, dtype=tl.float32) / 0.031622776601683805)
    r213_value = r213_value.to(tl.float32)
    r213_value = tl.broadcast_to(r213_value, [$(512)])
    tl.store($(candidate_vdiv_21_ptr) + r213_z, r213_value)
    r214_z = tl.arange($(512))
    r214_value = (tl.load($(candidate_vsqrt_22_ptr) + r214_z, dtype=tl.float32) / 0.031622776601683805)
    r214_value = r214_value.to(tl.float32)
    r214_value = tl.broadcast_to(r214_value, [$(512)])
    tl.store($(candidate_vdiv_22_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(512))
    r215_value = (tl.load($(candidate_vsqrt_23_ptr) + r215_z, dtype=tl.float32) / 0.031622776601683805)
    r215_value = r215_value.to(tl.float32)
    r215_value = tl.broadcast_to(r215_value, [$(512)])
    tl.store($(candidate_vdiv_23_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(512))
    r216_value = (tl.load($(candidate_vsqrt_24_ptr) + r216_z, dtype=tl.float32) / 0.031622776601683805)
    r216_value = r216_value.to(tl.float32)
    r216_value = tl.broadcast_to(r216_value, [$(512)])
    tl.store($(candidate_vdiv_24_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(512))
    r217_value = (tl.load($(candidate_vsqrt_25_ptr) + r217_z, dtype=tl.float32) / 0.031622776601683805)
    r217_value = r217_value.to(tl.float32)
    r217_value = tl.broadcast_to(r217_value, [$(512)])
    tl.store($(candidate_vdiv_25_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(512))
    r218_value = (tl.load($(candidate_vsqrt_26_ptr) + r218_z, dtype=tl.float32) / 0.031622776601683805)
    r218_value = r218_value.to(tl.float32)
    r218_value = tl.broadcast_to(r218_value, [$(512)])
    tl.store($(candidate_vdiv_26_ptr) + r218_z, r218_value)
    r219_z = tl.arange($(512))
    r219_value = (tl.load($(candidate_vsqrt_27_ptr) + r219_z, dtype=tl.float32) / 0.031622776601683805)
    r219_value = r219_value.to(tl.float32)
    r219_value = tl.broadcast_to(r219_value, [$(512)])
    tl.store($(candidate_vdiv_27_ptr) + r219_z, r219_value)
    r220_z = tl.arange($(512))
    r220_value = (tl.load($(candidate_vsqrt_28_ptr) + r220_z, dtype=tl.float32) / 0.031622776601683805)
    r220_value = r220_value.to(tl.float32)
    r220_value = tl.broadcast_to(r220_value, [$(512)])
    tl.store($(candidate_vdiv_28_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(512))
    r221_value = (tl.load($(candidate_vsqrt_29_ptr) + r221_z, dtype=tl.float32) / 0.031622776601683805)
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(512)])
    tl.store($(candidate_vdiv_29_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(512))
    r222_value = (tl.load($(candidate_vsqrt_30_ptr) + r222_z, dtype=tl.float32) / 0.031622776601683805)
    r222_value = r222_value.to(tl.float32)
    r222_value = tl.broadcast_to(r222_value, [$(512)])
    tl.store($(candidate_vdiv_30_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(512))
    r223_value = (tl.load($(candidate_vsqrt_31_ptr) + r223_z, dtype=tl.float32) / 0.031622776601683805)
    r223_value = r223_value.to(tl.float32)
    r223_value = tl.broadcast_to(r223_value, [$(512)])
    tl.store($(candidate_vdiv_31_ptr) + r223_z, r223_value)
    r224_z = tl.arange($(512))
    r224_value = (tl.load($(candidate_vdiv_ptr) + r224_z, dtype=tl.float32) + 0.00000001)
    r224_value = r224_value.to(tl.float32)
    r224_value = tl.broadcast_to(r224_value, [$(512)])
    tl.store($(candidate_vadd_ptr) + r224_z, r224_value)
    r225_z = tl.arange($(512))
    r225_value = (tl.load($(candidate_vdiv_1_ptr) + r225_z, dtype=tl.float32) + 0.00000001)
    r225_value = r225_value.to(tl.float32)
    r225_value = tl.broadcast_to(r225_value, [$(512)])
    tl.store($(candidate_vadd_1_ptr) + r225_z, r225_value)
    r226_z = tl.arange($(512))
    r226_value = (tl.load($(candidate_vdiv_2_ptr) + r226_z, dtype=tl.float32) + 0.00000001)
    r226_value = r226_value.to(tl.float32)
    r226_value = tl.broadcast_to(r226_value, [$(512)])
    tl.store($(candidate_vadd_2_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(512))
    r227_value = (tl.load($(candidate_vdiv_3_ptr) + r227_z, dtype=tl.float32) + 0.00000001)
    r227_value = r227_value.to(tl.float32)
    r227_value = tl.broadcast_to(r227_value, [$(512)])
    tl.store($(candidate_vadd_3_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(512))
    r228_value = (tl.load($(candidate_vdiv_4_ptr) + r228_z, dtype=tl.float32) + 0.00000001)
    r228_value = r228_value.to(tl.float32)
    r228_value = tl.broadcast_to(r228_value, [$(512)])
    tl.store($(candidate_vadd_4_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(512))
    r229_value = (tl.load($(candidate_vdiv_5_ptr) + r229_z, dtype=tl.float32) + 0.00000001)
    r229_value = r229_value.to(tl.float32)
    r229_value = tl.broadcast_to(r229_value, [$(512)])
    tl.store($(candidate_vadd_5_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(512))
    r230_value = (tl.load($(candidate_vdiv_6_ptr) + r230_z, dtype=tl.float32) + 0.00000001)
    r230_value = r230_value.to(tl.float32)
    r230_value = tl.broadcast_to(r230_value, [$(512)])
    tl.store($(candidate_vadd_6_ptr) + r230_z, r230_value)
    r231_z = tl.arange($(512))
    r231_value = (tl.load($(candidate_vdiv_7_ptr) + r231_z, dtype=tl.float32) + 0.00000001)
    r231_value = r231_value.to(tl.float32)
    r231_value = tl.broadcast_to(r231_value, [$(512)])
    tl.store($(candidate_vadd_7_ptr) + r231_z, r231_value)
    r232_z = tl.arange($(512))
    r232_value = (tl.load($(candidate_vdiv_8_ptr) + r232_z, dtype=tl.float32) + 0.00000001)
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(512)])
    tl.store($(candidate_vadd_8_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(512))
    r233_value = (tl.load($(candidate_vdiv_9_ptr) + r233_z, dtype=tl.float32) + 0.00000001)
    r233_value = r233_value.to(tl.float32)
    r233_value = tl.broadcast_to(r233_value, [$(512)])
    tl.store($(candidate_vadd_9_ptr) + r233_z, r233_value)
    r234_z = tl.arange($(512))
    r234_value = (tl.load($(candidate_vdiv_10_ptr) + r234_z, dtype=tl.float32) + 0.00000001)
    r234_value = r234_value.to(tl.float32)
    r234_value = tl.broadcast_to(r234_value, [$(512)])
    tl.store($(candidate_vadd_10_ptr) + r234_z, r234_value)
    r235_z = tl.arange($(512))
    r235_value = (tl.load($(candidate_vdiv_11_ptr) + r235_z, dtype=tl.float32) + 0.00000001)
    r235_value = r235_value.to(tl.float32)
    r235_value = tl.broadcast_to(r235_value, [$(512)])
    tl.store($(candidate_vadd_11_ptr) + r235_z, r235_value)
    r236_z = tl.arange($(512))
    r236_value = (tl.load($(candidate_vdiv_12_ptr) + r236_z, dtype=tl.float32) + 0.00000001)
    r236_value = r236_value.to(tl.float32)
    r236_value = tl.broadcast_to(r236_value, [$(512)])
    tl.store($(candidate_vadd_12_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(512))
    r237_value = (tl.load($(candidate_vdiv_13_ptr) + r237_z, dtype=tl.float32) + 0.00000001)
    r237_value = r237_value.to(tl.float32)
    r237_value = tl.broadcast_to(r237_value, [$(512)])
    tl.store($(candidate_vadd_13_ptr) + r237_z, r237_value)
    r238_z = tl.arange($(512))
    r238_value = (tl.load($(candidate_vdiv_14_ptr) + r238_z, dtype=tl.float32) + 0.00000001)
    r238_value = r238_value.to(tl.float32)
    r238_value = tl.broadcast_to(r238_value, [$(512)])
    tl.store($(candidate_vadd_14_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(512))
    r239_value = (tl.load($(candidate_vdiv_15_ptr) + r239_z, dtype=tl.float32) + 0.00000001)
    r239_value = r239_value.to(tl.float32)
    r239_value = tl.broadcast_to(r239_value, [$(512)])
    tl.store($(candidate_vadd_15_ptr) + r239_z, r239_value)
    r240_z = tl.arange($(512))
    r240_value = (tl.load($(candidate_vdiv_16_ptr) + r240_z, dtype=tl.float32) + 0.00000001)
    r240_value = r240_value.to(tl.float32)
    r240_value = tl.broadcast_to(r240_value, [$(512)])
    tl.store($(candidate_vadd_16_ptr) + r240_z, r240_value)
    r241_z = tl.arange($(512))
    r241_value = (tl.load($(candidate_vdiv_17_ptr) + r241_z, dtype=tl.float32) + 0.00000001)
    r241_value = r241_value.to(tl.float32)
    r241_value = tl.broadcast_to(r241_value, [$(512)])
    tl.store($(candidate_vadd_17_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(512))
    r242_value = (tl.load($(candidate_vdiv_18_ptr) + r242_z, dtype=tl.float32) + 0.00000001)
    r242_value = r242_value.to(tl.float32)
    r242_value = tl.broadcast_to(r242_value, [$(512)])
    tl.store($(candidate_vadd_18_ptr) + r242_z, r242_value)
    r243_z = tl.arange($(512))
    r243_value = (tl.load($(candidate_vdiv_19_ptr) + r243_z, dtype=tl.float32) + 0.00000001)
    r243_value = r243_value.to(tl.float32)
    r243_value = tl.broadcast_to(r243_value, [$(512)])
    tl.store($(candidate_vadd_19_ptr) + r243_z, r243_value)
    r244_z = tl.arange($(512))
    r244_value = (tl.load($(candidate_vdiv_20_ptr) + r244_z, dtype=tl.float32) + 0.00000001)
    r244_value = r244_value.to(tl.float32)
    r244_value = tl.broadcast_to(r244_value, [$(512)])
    tl.store($(candidate_vadd_20_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(512))
    r245_value = (tl.load($(candidate_vdiv_21_ptr) + r245_z, dtype=tl.float32) + 0.00000001)
    r245_value = r245_value.to(tl.float32)
    r245_value = tl.broadcast_to(r245_value, [$(512)])
    tl.store($(candidate_vadd_21_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(512))
    r246_value = (tl.load($(candidate_vdiv_22_ptr) + r246_z, dtype=tl.float32) + 0.00000001)
    r246_value = r246_value.to(tl.float32)
    r246_value = tl.broadcast_to(r246_value, [$(512)])
    tl.store($(candidate_vadd_22_ptr) + r246_z, r246_value)
    r247_z = tl.arange($(512))
    r247_value = (tl.load($(candidate_vdiv_23_ptr) + r247_z, dtype=tl.float32) + 0.00000001)
    r247_value = r247_value.to(tl.float32)
    r247_value = tl.broadcast_to(r247_value, [$(512)])
    tl.store($(candidate_vadd_23_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(512))
    r248_value = (tl.load($(candidate_vdiv_24_ptr) + r248_z, dtype=tl.float32) + 0.00000001)
    r248_value = r248_value.to(tl.float32)
    r248_value = tl.broadcast_to(r248_value, [$(512)])
    tl.store($(candidate_vadd_24_ptr) + r248_z, r248_value)
    r249_z = tl.arange($(512))
    r249_value = (tl.load($(candidate_vdiv_25_ptr) + r249_z, dtype=tl.float32) + 0.00000001)
    r249_value = r249_value.to(tl.float32)
    r249_value = tl.broadcast_to(r249_value, [$(512)])
    tl.store($(candidate_vadd_25_ptr) + r249_z, r249_value)
    r250_z = tl.arange($(512))
    r250_value = (tl.load($(candidate_vdiv_26_ptr) + r250_z, dtype=tl.float32) + 0.00000001)
    r250_value = r250_value.to(tl.float32)
    r250_value = tl.broadcast_to(r250_value, [$(512)])
    tl.store($(candidate_vadd_26_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(512))
    r251_value = (tl.load($(candidate_vdiv_27_ptr) + r251_z, dtype=tl.float32) + 0.00000001)
    r251_value = r251_value.to(tl.float32)
    r251_value = tl.broadcast_to(r251_value, [$(512)])
    tl.store($(candidate_vadd_27_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(512))
    r252_value = (tl.load($(candidate_vdiv_28_ptr) + r252_z, dtype=tl.float32) + 0.00000001)
    r252_value = r252_value.to(tl.float32)
    r252_value = tl.broadcast_to(r252_value, [$(512)])
    tl.store($(candidate_vadd_28_ptr) + r252_z, r252_value)
    r253_z = tl.arange($(512))
    r253_value = (tl.load($(candidate_vdiv_29_ptr) + r253_z, dtype=tl.float32) + 0.00000001)
    r253_value = r253_value.to(tl.float32)
    r253_value = tl.broadcast_to(r253_value, [$(512)])
    tl.store($(candidate_vadd_29_ptr) + r253_z, r253_value)
    r254_z = tl.arange($(512))
    r254_value = (tl.load($(candidate_vdiv_30_ptr) + r254_z, dtype=tl.float32) + 0.00000001)
    r254_value = r254_value.to(tl.float32)
    r254_value = tl.broadcast_to(r254_value, [$(512)])
    tl.store($(candidate_vadd_30_ptr) + r254_z, r254_value)
    r255_z = tl.arange($(512))
    r255_value = (tl.load($(candidate_vdiv_31_ptr) + r255_z, dtype=tl.float32) + 0.00000001)
    r255_value = r255_value.to(tl.float32)
    r255_value = tl.broadcast_to(r255_value, [$(512)])
    tl.store($(candidate_vadd_31_ptr) + r255_z, r255_value)
    r256_z = tl.arange($(512))
    r256_value = (tl.load($(candidate_vmul_ptr) + r256_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_ptr) + r256_z, dtype=tl.float32) / tl.load($(candidate_vadd_ptr) + r256_z, dtype=tl.float32)))
    r256_value = r256_value.to(tl.float32)
    r256_value = tl.broadcast_to(r256_value, [$(512)])
    tl.store($(candidate_vaddcdiv_ptr) + r256_z, r256_value)
    r257_z = tl.arange($(512))
    r257_value = (tl.load($(candidate_vmul_1_ptr) + r257_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_1_ptr) + r257_z, dtype=tl.float32) / tl.load($(candidate_vadd_1_ptr) + r257_z, dtype=tl.float32)))
    r257_value = r257_value.to(tl.float32)
    r257_value = tl.broadcast_to(r257_value, [$(512)])
    tl.store($(candidate_vaddcdiv_1_ptr) + r257_z, r257_value)
    r258_z = tl.arange($(512))
    r258_value = (tl.load($(candidate_vmul_2_ptr) + r258_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_2_ptr) + r258_z, dtype=tl.float32) / tl.load($(candidate_vadd_2_ptr) + r258_z, dtype=tl.float32)))
    r258_value = r258_value.to(tl.float32)
    r258_value = tl.broadcast_to(r258_value, [$(512)])
    tl.store($(candidate_vaddcdiv_2_ptr) + r258_z, r258_value)
    r259_z = tl.arange($(512))
    r259_value = (tl.load($(candidate_vmul_3_ptr) + r259_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_3_ptr) + r259_z, dtype=tl.float32) / tl.load($(candidate_vadd_3_ptr) + r259_z, dtype=tl.float32)))
    r259_value = r259_value.to(tl.float32)
    r259_value = tl.broadcast_to(r259_value, [$(512)])
    tl.store($(candidate_vaddcdiv_3_ptr) + r259_z, r259_value)
    r260_z = tl.arange($(512))
    r260_value = (tl.load($(candidate_vmul_4_ptr) + r260_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_4_ptr) + r260_z, dtype=tl.float32) / tl.load($(candidate_vadd_4_ptr) + r260_z, dtype=tl.float32)))
    r260_value = r260_value.to(tl.float32)
    r260_value = tl.broadcast_to(r260_value, [$(512)])
    tl.store($(candidate_vaddcdiv_4_ptr) + r260_z, r260_value)
    r261_z = tl.arange($(512))
    r261_value = (tl.load($(candidate_vmul_5_ptr) + r261_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_5_ptr) + r261_z, dtype=tl.float32) / tl.load($(candidate_vadd_5_ptr) + r261_z, dtype=tl.float32)))
    r261_value = r261_value.to(tl.float32)
    r261_value = tl.broadcast_to(r261_value, [$(512)])
    tl.store($(candidate_vaddcdiv_5_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(512))
    r262_value = (tl.load($(candidate_vmul_6_ptr) + r262_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_6_ptr) + r262_z, dtype=tl.float32) / tl.load($(candidate_vadd_6_ptr) + r262_z, dtype=tl.float32)))
    r262_value = r262_value.to(tl.float32)
    r262_value = tl.broadcast_to(r262_value, [$(512)])
    tl.store($(candidate_vaddcdiv_6_ptr) + r262_z, r262_value)
    r263_z = tl.arange($(512))
    r263_value = (tl.load($(candidate_vmul_7_ptr) + r263_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_7_ptr) + r263_z, dtype=tl.float32) / tl.load($(candidate_vadd_7_ptr) + r263_z, dtype=tl.float32)))
    r263_value = r263_value.to(tl.float32)
    r263_value = tl.broadcast_to(r263_value, [$(512)])
    tl.store($(candidate_vaddcdiv_7_ptr) + r263_z, r263_value)
    r264_z = tl.arange($(512))
    r264_value = (tl.load($(candidate_vmul_8_ptr) + r264_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_8_ptr) + r264_z, dtype=tl.float32) / tl.load($(candidate_vadd_8_ptr) + r264_z, dtype=tl.float32)))
    r264_value = r264_value.to(tl.float32)
    r264_value = tl.broadcast_to(r264_value, [$(512)])
    tl.store($(candidate_vaddcdiv_8_ptr) + r264_z, r264_value)
    r265_z = tl.arange($(512))
    r265_value = (tl.load($(candidate_vmul_9_ptr) + r265_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_9_ptr) + r265_z, dtype=tl.float32) / tl.load($(candidate_vadd_9_ptr) + r265_z, dtype=tl.float32)))
    r265_value = r265_value.to(tl.float32)
    r265_value = tl.broadcast_to(r265_value, [$(512)])
    tl.store($(candidate_vaddcdiv_9_ptr) + r265_z, r265_value)
    r266_z = tl.arange($(512))
    r266_value = (tl.load($(candidate_vmul_10_ptr) + r266_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_10_ptr) + r266_z, dtype=tl.float32) / tl.load($(candidate_vadd_10_ptr) + r266_z, dtype=tl.float32)))
    r266_value = r266_value.to(tl.float32)
    r266_value = tl.broadcast_to(r266_value, [$(512)])
    tl.store($(candidate_vaddcdiv_10_ptr) + r266_z, r266_value)
    r267_z = tl.arange($(512))
    r267_value = (tl.load($(candidate_vmul_11_ptr) + r267_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_11_ptr) + r267_z, dtype=tl.float32) / tl.load($(candidate_vadd_11_ptr) + r267_z, dtype=tl.float32)))
    r267_value = r267_value.to(tl.float32)
    r267_value = tl.broadcast_to(r267_value, [$(512)])
    tl.store($(candidate_vaddcdiv_11_ptr) + r267_z, r267_value)
    r268_z = tl.arange($(512))
    r268_value = (tl.load($(candidate_vmul_12_ptr) + r268_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_12_ptr) + r268_z, dtype=tl.float32) / tl.load($(candidate_vadd_12_ptr) + r268_z, dtype=tl.float32)))
    r268_value = r268_value.to(tl.float32)
    r268_value = tl.broadcast_to(r268_value, [$(512)])
    tl.store($(candidate_vaddcdiv_12_ptr) + r268_z, r268_value)
    r269_z = tl.arange($(512))
    r269_value = (tl.load($(candidate_vmul_13_ptr) + r269_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_13_ptr) + r269_z, dtype=tl.float32) / tl.load($(candidate_vadd_13_ptr) + r269_z, dtype=tl.float32)))
    r269_value = r269_value.to(tl.float32)
    r269_value = tl.broadcast_to(r269_value, [$(512)])
    tl.store($(candidate_vaddcdiv_13_ptr) + r269_z, r269_value)
    r270_z = tl.arange($(512))
    r270_value = (tl.load($(candidate_vmul_14_ptr) + r270_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_14_ptr) + r270_z, dtype=tl.float32) / tl.load($(candidate_vadd_14_ptr) + r270_z, dtype=tl.float32)))
    r270_value = r270_value.to(tl.float32)
    r270_value = tl.broadcast_to(r270_value, [$(512)])
    tl.store($(candidate_vaddcdiv_14_ptr) + r270_z, r270_value)
    r271_z = tl.arange($(512))
    r271_value = (tl.load($(candidate_vmul_15_ptr) + r271_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_15_ptr) + r271_z, dtype=tl.float32) / tl.load($(candidate_vadd_15_ptr) + r271_z, dtype=tl.float32)))
    r271_value = r271_value.to(tl.float32)
    r271_value = tl.broadcast_to(r271_value, [$(512)])
    tl.store($(candidate_vaddcdiv_15_ptr) + r271_z, r271_value)
    r272_z = tl.arange($(512))
    r272_value = (tl.load($(candidate_vmul_16_ptr) + r272_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_16_ptr) + r272_z, dtype=tl.float32) / tl.load($(candidate_vadd_16_ptr) + r272_z, dtype=tl.float32)))
    r272_value = r272_value.to(tl.float32)
    r272_value = tl.broadcast_to(r272_value, [$(512)])
    tl.store($(candidate_vaddcdiv_16_ptr) + r272_z, r272_value)
    r273_z = tl.arange($(512))
    r273_value = (tl.load($(candidate_vmul_17_ptr) + r273_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_17_ptr) + r273_z, dtype=tl.float32) / tl.load($(candidate_vadd_17_ptr) + r273_z, dtype=tl.float32)))
    r273_value = r273_value.to(tl.float32)
    r273_value = tl.broadcast_to(r273_value, [$(512)])
    tl.store($(candidate_vaddcdiv_17_ptr) + r273_z, r273_value)
    r274_z = tl.arange($(512))
    r274_value = (tl.load($(candidate_vmul_18_ptr) + r274_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_18_ptr) + r274_z, dtype=tl.float32) / tl.load($(candidate_vadd_18_ptr) + r274_z, dtype=tl.float32)))
    r274_value = r274_value.to(tl.float32)
    r274_value = tl.broadcast_to(r274_value, [$(512)])
    tl.store($(candidate_vaddcdiv_18_ptr) + r274_z, r274_value)
    r275_z = tl.arange($(512))
    r275_value = (tl.load($(candidate_vmul_19_ptr) + r275_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_19_ptr) + r275_z, dtype=tl.float32) / tl.load($(candidate_vadd_19_ptr) + r275_z, dtype=tl.float32)))
    r275_value = r275_value.to(tl.float32)
    r275_value = tl.broadcast_to(r275_value, [$(512)])
    tl.store($(candidate_vaddcdiv_19_ptr) + r275_z, r275_value)
    r276_z = tl.arange($(512))
    r276_value = (tl.load($(candidate_vmul_20_ptr) + r276_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_20_ptr) + r276_z, dtype=tl.float32) / tl.load($(candidate_vadd_20_ptr) + r276_z, dtype=tl.float32)))
    r276_value = r276_value.to(tl.float32)
    r276_value = tl.broadcast_to(r276_value, [$(512)])
    tl.store($(candidate_vaddcdiv_20_ptr) + r276_z, r276_value)
    r277_z = tl.arange($(512))
    r277_value = (tl.load($(candidate_vmul_21_ptr) + r277_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_21_ptr) + r277_z, dtype=tl.float32) / tl.load($(candidate_vadd_21_ptr) + r277_z, dtype=tl.float32)))
    r277_value = r277_value.to(tl.float32)
    r277_value = tl.broadcast_to(r277_value, [$(512)])
    tl.store($(candidate_vaddcdiv_21_ptr) + r277_z, r277_value)
    r278_z = tl.arange($(512))
    r278_value = (tl.load($(candidate_vmul_22_ptr) + r278_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_22_ptr) + r278_z, dtype=tl.float32) / tl.load($(candidate_vadd_22_ptr) + r278_z, dtype=tl.float32)))
    r278_value = r278_value.to(tl.float32)
    r278_value = tl.broadcast_to(r278_value, [$(512)])
    tl.store($(candidate_vaddcdiv_22_ptr) + r278_z, r278_value)
    r279_z = tl.arange($(512))
    r279_value = (tl.load($(candidate_vmul_23_ptr) + r279_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_23_ptr) + r279_z, dtype=tl.float32) / tl.load($(candidate_vadd_23_ptr) + r279_z, dtype=tl.float32)))
    r279_value = r279_value.to(tl.float32)
    r279_value = tl.broadcast_to(r279_value, [$(512)])
    tl.store($(candidate_vaddcdiv_23_ptr) + r279_z, r279_value)
    r280_z = tl.arange($(512))
    r280_value = (tl.load($(candidate_vmul_24_ptr) + r280_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_24_ptr) + r280_z, dtype=tl.float32) / tl.load($(candidate_vadd_24_ptr) + r280_z, dtype=tl.float32)))
    r280_value = r280_value.to(tl.float32)
    r280_value = tl.broadcast_to(r280_value, [$(512)])
    tl.store($(candidate_vaddcdiv_24_ptr) + r280_z, r280_value)
    r281_z = tl.arange($(512))
    r281_value = (tl.load($(candidate_vmul_25_ptr) + r281_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_25_ptr) + r281_z, dtype=tl.float32) / tl.load($(candidate_vadd_25_ptr) + r281_z, dtype=tl.float32)))
    r281_value = r281_value.to(tl.float32)
    r281_value = tl.broadcast_to(r281_value, [$(512)])
    tl.store($(candidate_vaddcdiv_25_ptr) + r281_z, r281_value)
    r282_z = tl.arange($(512))
    r282_value = (tl.load($(candidate_vmul_26_ptr) + r282_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_26_ptr) + r282_z, dtype=tl.float32) / tl.load($(candidate_vadd_26_ptr) + r282_z, dtype=tl.float32)))
    r282_value = r282_value.to(tl.float32)
    r282_value = tl.broadcast_to(r282_value, [$(512)])
    tl.store($(candidate_vaddcdiv_26_ptr) + r282_z, r282_value)
    r283_z = tl.arange($(512))
    r283_value = (tl.load($(candidate_vmul_27_ptr) + r283_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_27_ptr) + r283_z, dtype=tl.float32) / tl.load($(candidate_vadd_27_ptr) + r283_z, dtype=tl.float32)))
    r283_value = r283_value.to(tl.float32)
    r283_value = tl.broadcast_to(r283_value, [$(512)])
    tl.store($(candidate_vaddcdiv_27_ptr) + r283_z, r283_value)
    r284_z = tl.arange($(512))
    r284_value = (tl.load($(candidate_vmul_28_ptr) + r284_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_28_ptr) + r284_z, dtype=tl.float32) / tl.load($(candidate_vadd_28_ptr) + r284_z, dtype=tl.float32)))
    r284_value = r284_value.to(tl.float32)
    r284_value = tl.broadcast_to(r284_value, [$(512)])
    tl.store($(candidate_vaddcdiv_28_ptr) + r284_z, r284_value)
    r285_z = tl.arange($(512))
    r285_value = (tl.load($(candidate_vmul_29_ptr) + r285_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_29_ptr) + r285_z, dtype=tl.float32) / tl.load($(candidate_vadd_29_ptr) + r285_z, dtype=tl.float32)))
    r285_value = r285_value.to(tl.float32)
    r285_value = tl.broadcast_to(r285_value, [$(512)])
    tl.store($(candidate_vaddcdiv_29_ptr) + r285_z, r285_value)
    r286_z = tl.arange($(512))
    r286_value = (tl.load($(candidate_vmul_30_ptr) + r286_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_30_ptr) + r286_z, dtype=tl.float32) / tl.load($(candidate_vadd_30_ptr) + r286_z, dtype=tl.float32)))
    r286_value = r286_value.to(tl.float32)
    r286_value = tl.broadcast_to(r286_value, [$(512)])
    tl.store($(candidate_vaddcdiv_30_ptr) + r286_z, r286_value)
    r287_z = tl.arange($(512))
    r287_value = (tl.load($(candidate_vmul_31_ptr) + r287_z, dtype=tl.float32) + -0.010000000000000002 * (tl.load($(candidate_vlerp_31_ptr) + r287_z, dtype=tl.float32) / tl.load($(candidate_vadd_31_ptr) + r287_z, dtype=tl.float32)))
    r287_value = r287_value.to(tl.float32)
    r287_value = tl.broadcast_to(r287_value, [$(512)])
    tl.store($(candidate_vaddcdiv_31_ptr) + r287_z, r287_value)
    r288_z = tl.arange($(16384))
    r288_value = tl.where(((r288_z / $(512)) % $(32)) < $(1), tl.load($(candidate_vaddcdiv_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(2), tl.load($(candidate_vaddcdiv_1_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(3), tl.load($(candidate_vaddcdiv_2_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(4), tl.load($(candidate_vaddcdiv_3_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(5), tl.load($(candidate_vaddcdiv_4_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(6), tl.load($(candidate_vaddcdiv_5_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(7), tl.load($(candidate_vaddcdiv_6_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(8), tl.load($(candidate_vaddcdiv_7_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(9), tl.load($(candidate_vaddcdiv_8_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(10), tl.load($(candidate_vaddcdiv_9_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(11), tl.load($(candidate_vaddcdiv_10_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(12), tl.load($(candidate_vaddcdiv_11_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(13), tl.load($(candidate_vaddcdiv_12_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(14), tl.load($(candidate_vaddcdiv_13_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(15), tl.load($(candidate_vaddcdiv_14_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(16), tl.load($(candidate_vaddcdiv_15_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(17), tl.load($(candidate_vaddcdiv_16_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(18), tl.load($(candidate_vaddcdiv_17_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(19), tl.load($(candidate_vaddcdiv_18_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(20), tl.load($(candidate_vaddcdiv_19_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(21), tl.load($(candidate_vaddcdiv_20_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(22), tl.load($(candidate_vaddcdiv_21_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(23), tl.load($(candidate_vaddcdiv_22_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(24), tl.load($(candidate_vaddcdiv_23_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(25), tl.load($(candidate_vaddcdiv_24_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(26), tl.load($(candidate_vaddcdiv_25_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(27), tl.load($(candidate_vaddcdiv_26_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(28), tl.load($(candidate_vaddcdiv_27_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(29), tl.load($(candidate_vaddcdiv_28_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(30), tl.load($(candidate_vaddcdiv_29_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.where(((r288_z / $(512)) % $(32)) < $(31), tl.load($(candidate_vaddcdiv_30_ptr) + (r288_z % $(512)), dtype=tl.float32), tl.load($(candidate_vaddcdiv_31_ptr) + (r288_z % $(512)), dtype=tl.float32))))))))))))))))))))))))))))))))
    r288_value = r288_value.to(tl.float32)
    r288_value = tl.broadcast_to(r288_value, [$(16384)])
    tl.store($(candidate_vstack_ptr) + r288_z, r288_value)
    z = tl.arange($(16384))
    value = tl.load($(candidate_vstack_ptr) + ((((z / $(512)) % $(32)) * $(512)) + (z % $(512))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(16384)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_adamw_multi_foreach_vs_loop
