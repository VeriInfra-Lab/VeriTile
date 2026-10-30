import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_muon_ns_schedule_bmm_vs_loop
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
  let vlerp_ptr : RegionName := "vlerp"
  let vselect_scatter_ptr : RegionName := "vselect_scatter"
  let vlerp_1_ptr : RegionName := "vlerp_1"
  let vlinalg_vector_norm_ptr : RegionName := "vlinalg_vector_norm"
  let vclamp_ptr : RegionName := "vclamp"
  let vdiv_ptr : RegionName := "vdiv"
  let vmm_ptr : RegionName := "vmm"
  let vaddmm_ptr : RegionName := "vaddmm"
  let vaddmm_1_ptr : RegionName := "vaddmm_1"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vaddmm_2_ptr : RegionName := "vaddmm_2"
  let vaddmm_3_ptr : RegionName := "vaddmm_3"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vaddmm_4_ptr : RegionName := "vaddmm_4"
  let vaddmm_5_ptr : RegionName := "vaddmm_5"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vaddmm_6_ptr : RegionName := "vaddmm_6"
  let vaddmm_7_ptr : RegionName := "vaddmm_7"
  let vmm_4_ptr : RegionName := "vmm_4"
  let vaddmm_8_ptr : RegionName := "vaddmm_8"
  let vaddmm_9_ptr : RegionName := "vaddmm_9"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vaddmm_10_ptr : RegionName := "vaddmm_10"
  let vaddmm_11_ptr : RegionName := "vaddmm_11"
  let vmm_6_ptr : RegionName := "vmm_6"
  let vaddmm_12_ptr : RegionName := "vaddmm_12"
  let vaddmm_13_ptr : RegionName := "vaddmm_13"
  let vmm_7_ptr : RegionName := "vmm_7"
  let vaddmm_14_ptr : RegionName := "vaddmm_14"
  let vaddmm_15_ptr : RegionName := "vaddmm_15"
  let vmm_8_ptr : RegionName := "vmm_8"
  let vaddmm_16_ptr : RegionName := "vaddmm_16"
  let vaddmm_17_ptr : RegionName := "vaddmm_17"
  let vmm_9_ptr : RegionName := "vmm_9"
  let vaddmm_18_ptr : RegionName := "vaddmm_18"
  let vaddmm_19_ptr : RegionName := "vaddmm_19"
  let vmul_ptr : RegionName := "vmul"
  let vselect_scatter_1_ptr : RegionName := "vselect_scatter_1"
  let vadd_ptr : RegionName := "vadd"
  let vselect_scatter_2_ptr : RegionName := "vselect_scatter_2"
  let vlerp_2_ptr : RegionName := "vlerp_2"
  let vselect_scatter_3_ptr : RegionName := "vselect_scatter_3"
  let vlerp_3_ptr : RegionName := "vlerp_3"
  let vlinalg_vector_norm_1_ptr : RegionName := "vlinalg_vector_norm_1"
  let vclamp_1_ptr : RegionName := "vclamp_1"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vmm_10_ptr : RegionName := "vmm_10"
  let vaddmm_20_ptr : RegionName := "vaddmm_20"
  let vaddmm_21_ptr : RegionName := "vaddmm_21"
  let vmm_11_ptr : RegionName := "vmm_11"
  let vaddmm_22_ptr : RegionName := "vaddmm_22"
  let vaddmm_23_ptr : RegionName := "vaddmm_23"
  let vmm_12_ptr : RegionName := "vmm_12"
  let vaddmm_24_ptr : RegionName := "vaddmm_24"
  let vaddmm_25_ptr : RegionName := "vaddmm_25"
  let vmm_13_ptr : RegionName := "vmm_13"
  let vaddmm_26_ptr : RegionName := "vaddmm_26"
  let vaddmm_27_ptr : RegionName := "vaddmm_27"
  let vmm_14_ptr : RegionName := "vmm_14"
  let vaddmm_28_ptr : RegionName := "vaddmm_28"
  let vaddmm_29_ptr : RegionName := "vaddmm_29"
  let vmm_15_ptr : RegionName := "vmm_15"
  let vaddmm_30_ptr : RegionName := "vaddmm_30"
  let vaddmm_31_ptr : RegionName := "vaddmm_31"
  let vmm_16_ptr : RegionName := "vmm_16"
  let vaddmm_32_ptr : RegionName := "vaddmm_32"
  let vaddmm_33_ptr : RegionName := "vaddmm_33"
  let vmm_17_ptr : RegionName := "vmm_17"
  let vaddmm_34_ptr : RegionName := "vaddmm_34"
  let vaddmm_35_ptr : RegionName := "vaddmm_35"
  let vmm_18_ptr : RegionName := "vmm_18"
  let vaddmm_36_ptr : RegionName := "vaddmm_36"
  let vaddmm_37_ptr : RegionName := "vaddmm_37"
  let vmm_19_ptr : RegionName := "vmm_19"
  let vaddmm_38_ptr : RegionName := "vaddmm_38"
  let vaddmm_39_ptr : RegionName := "vaddmm_39"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vselect_scatter_4_ptr : RegionName := "vselect_scatter_4"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vselect_scatter_5_ptr : RegionName := "vselect_scatter_5"
  let vlerp_4_ptr : RegionName := "vlerp_4"
  let vselect_scatter_6_ptr : RegionName := "vselect_scatter_6"
  let vlerp_5_ptr : RegionName := "vlerp_5"
  let vlinalg_vector_norm_2_ptr : RegionName := "vlinalg_vector_norm_2"
  let vclamp_2_ptr : RegionName := "vclamp_2"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vmm_20_ptr : RegionName := "vmm_20"
  let vaddmm_40_ptr : RegionName := "vaddmm_40"
  let vaddmm_41_ptr : RegionName := "vaddmm_41"
  let vmm_21_ptr : RegionName := "vmm_21"
  let vaddmm_42_ptr : RegionName := "vaddmm_42"
  let vaddmm_43_ptr : RegionName := "vaddmm_43"
  let vmm_22_ptr : RegionName := "vmm_22"
  let vaddmm_44_ptr : RegionName := "vaddmm_44"
  let vaddmm_45_ptr : RegionName := "vaddmm_45"
  let vmm_23_ptr : RegionName := "vmm_23"
  let vaddmm_46_ptr : RegionName := "vaddmm_46"
  let vaddmm_47_ptr : RegionName := "vaddmm_47"
  let vmm_24_ptr : RegionName := "vmm_24"
  let vaddmm_48_ptr : RegionName := "vaddmm_48"
  let vaddmm_49_ptr : RegionName := "vaddmm_49"
  let vmm_25_ptr : RegionName := "vmm_25"
  let vaddmm_50_ptr : RegionName := "vaddmm_50"
  let vaddmm_51_ptr : RegionName := "vaddmm_51"
  let vmm_26_ptr : RegionName := "vmm_26"
  let vaddmm_52_ptr : RegionName := "vaddmm_52"
  let vaddmm_53_ptr : RegionName := "vaddmm_53"
  let vmm_27_ptr : RegionName := "vmm_27"
  let vaddmm_54_ptr : RegionName := "vaddmm_54"
  let vaddmm_55_ptr : RegionName := "vaddmm_55"
  let vmm_28_ptr : RegionName := "vmm_28"
  let vaddmm_56_ptr : RegionName := "vaddmm_56"
  let vaddmm_57_ptr : RegionName := "vaddmm_57"
  let vmm_29_ptr : RegionName := "vmm_29"
  let vaddmm_58_ptr : RegionName := "vaddmm_58"
  let vaddmm_59_ptr : RegionName := "vaddmm_59"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vselect_scatter_7_ptr : RegionName := "vselect_scatter_7"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vselect_scatter_8_ptr : RegionName := "vselect_scatter_8"
  let vlerp_6_ptr : RegionName := "vlerp_6"
  let vselect_scatter_9_ptr : RegionName := "vselect_scatter_9"
  let vlerp_7_ptr : RegionName := "vlerp_7"
  let vlinalg_vector_norm_3_ptr : RegionName := "vlinalg_vector_norm_3"
  let vclamp_3_ptr : RegionName := "vclamp_3"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vmm_30_ptr : RegionName := "vmm_30"
  let vaddmm_60_ptr : RegionName := "vaddmm_60"
  let vaddmm_61_ptr : RegionName := "vaddmm_61"
  let vmm_31_ptr : RegionName := "vmm_31"
  let vaddmm_62_ptr : RegionName := "vaddmm_62"
  let vaddmm_63_ptr : RegionName := "vaddmm_63"
  let vmm_32_ptr : RegionName := "vmm_32"
  let vaddmm_64_ptr : RegionName := "vaddmm_64"
  let vaddmm_65_ptr : RegionName := "vaddmm_65"
  let vmm_33_ptr : RegionName := "vmm_33"
  let vaddmm_66_ptr : RegionName := "vaddmm_66"
  let vaddmm_67_ptr : RegionName := "vaddmm_67"
  let vmm_34_ptr : RegionName := "vmm_34"
  let vaddmm_68_ptr : RegionName := "vaddmm_68"
  let vaddmm_69_ptr : RegionName := "vaddmm_69"
  let vmm_35_ptr : RegionName := "vmm_35"
  let vaddmm_70_ptr : RegionName := "vaddmm_70"
  let vaddmm_71_ptr : RegionName := "vaddmm_71"
  let vmm_36_ptr : RegionName := "vmm_36"
  let vaddmm_72_ptr : RegionName := "vaddmm_72"
  let vaddmm_73_ptr : RegionName := "vaddmm_73"
  let vmm_37_ptr : RegionName := "vmm_37"
  let vaddmm_74_ptr : RegionName := "vaddmm_74"
  let vaddmm_75_ptr : RegionName := "vaddmm_75"
  let vmm_38_ptr : RegionName := "vmm_38"
  let vaddmm_76_ptr : RegionName := "vaddmm_76"
  let vaddmm_77_ptr : RegionName := "vaddmm_77"
  let vmm_39_ptr : RegionName := "vmm_39"
  let vaddmm_78_ptr : RegionName := "vaddmm_78"
  let vaddmm_79_ptr : RegionName := "vaddmm_79"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vselect_scatter_10_ptr : RegionName := "vselect_scatter_10"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vselect_scatter_11_ptr : RegionName := "vselect_scatter_11"
  triton {
    r0_z = tl.arange($(32768))
    r0_value = (tl.cast(0.0, tl.bfloat16) + (tl.load($(input0_ptr) + ((((r0_z / $(128)) % $(256)) * $(128)) + (r0_z % $(128))), dtype=tl.bfloat16) - tl.cast(0.0, tl.bfloat16)) * 0.050000000000000044)
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(32768)])
    tl.store($(vlerp_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(131072))
    r1_value = tl.where((((r1_z / $(32768)) % $(4)) == $(0)), tl.load($(vlerp_ptr) + ((((r1_z / $(128)) % $(256)) * $(128)) + (r1_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(131072)])
    tl.store($(vselect_scatter_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32768))
    r2_value = (tl.load($(input0_ptr) + ((((r2_z / $(128)) % $(256)) * $(128)) + (r2_z % $(128))), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_ptr) + ((((r2_z / $(128)) % $(256)) * $(128)) + (r2_z % $(128))), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r2_z / $(128)) % $(256)) * $(128)) + (r2_z % $(128))), dtype=tl.bfloat16)) * 0.95)
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(32768)])
    tl.store($(vlerp_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1))
    r3_lane = tl.arange($(32768))
    r3_value = tl.sqrt(tl.sum(tl.load($(vlerp_1_ptr) + (((r3_lane[None, :] % $(256)) * $(128)) + ((r3_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16) * tl.load($(vlerp_1_ptr) + (((r3_lane[None, :] % $(256)) * $(128)) + ((r3_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16), axis=1))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(1)])
    tl.store($(vlinalg_vector_norm_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(1))
    r4_value = tl.maximum(tl.load($(vlinalg_vector_norm_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(1)])
    tl.store($(vclamp_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_value = (tl.load($(vlerp_1_ptr) + (((r5_z % $(256)) * $(128)) + ((r5_z / $(256)) % $(128))), dtype=tl.bfloat16) / tl.load($(vclamp_ptr) + $(0), dtype=tl.bfloat16))
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(vdiv_ptr) + r5_z, r5_value)
    r6_row = tl.arange($(128))
    r6_col = tl.arange($(128))
    r6_lane = tl.arange($(256))
    r6_left = tl.load($(vdiv_ptr) + (((r6_row[:, None] % $(128)) * $(256)) + (r6_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r6_right = tl.load($(vdiv_ptr) + (((r6_col[None, :] % $(128)) * $(256)) + (r6_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r6_value = tl.dot(r6_left, r6_right)
    r6_value = r6_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r6_row[:, None] * $(128)) + r6_col[None, :]), r6_value)
    r7_row = tl.arange($(128))
    r7_col = tl.arange($(128))
    r7_lane = tl.arange($(128))
    r7_left = tl.load($(vmm_ptr) + (((r7_row[:, None] % $(128)) * $(128)) + (r7_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r7_right = tl.load($(vmm_ptr) + (((r7_lane[:, None] % $(128)) * $(128)) + (r7_col[None, :] % $(128))), dtype=tl.bfloat16)
    r7_value = tl.dot(r7_left, r7_right)
    r7_value = r7_value * 2.0315
    r7_value = r7_value + tl.load($(vmm_ptr) + (((r7_row[:, None] % $(128)) * $(128)) + (r7_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r7_value = r7_value.to(tl.bfloat16)
    tl.store($(vaddmm_ptr) + ((r7_row[:, None] * $(128)) + r7_col[None, :]), r7_value)
    r8_row = tl.arange($(128))
    r8_col = tl.arange($(256))
    r8_lane = tl.arange($(128))
    r8_left = tl.load($(vaddmm_ptr) + (((r8_row[:, None] % $(128)) * $(128)) + (r8_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r8_right = tl.load($(vdiv_ptr) + (((r8_lane[:, None] % $(128)) * $(256)) + (r8_col[None, :] % $(256))), dtype=tl.bfloat16)
    r8_value = tl.dot(r8_left, r8_right)
    r8_value = r8_value + tl.load($(vdiv_ptr) + (((r8_row[:, None] % $(128)) * $(256)) + (r8_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r8_value = r8_value.to(tl.bfloat16)
    tl.store($(vaddmm_1_ptr) + ((r8_row[:, None] * $(256)) + r8_col[None, :]), r8_value)
    r9_row = tl.arange($(128))
    r9_col = tl.arange($(128))
    r9_lane = tl.arange($(256))
    r9_left = tl.load($(vaddmm_1_ptr) + (((r9_row[:, None] % $(128)) * $(256)) + (r9_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r9_right = tl.load($(vaddmm_1_ptr) + (((r9_col[None, :] % $(128)) * $(256)) + (r9_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r9_value = tl.dot(r9_left, r9_right)
    r9_value = r9_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r9_row[:, None] * $(128)) + r9_col[None, :]), r9_value)
    r10_row = tl.arange($(128))
    r10_col = tl.arange($(128))
    r10_lane = tl.arange($(128))
    r10_left = tl.load($(vmm_1_ptr) + (((r10_row[:, None] % $(128)) * $(128)) + (r10_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r10_right = tl.load($(vmm_1_ptr) + (((r10_lane[:, None] % $(128)) * $(128)) + (r10_col[None, :] % $(128))), dtype=tl.bfloat16)
    r10_value = tl.dot(r10_left, r10_right)
    r10_value = r10_value * 2.0315
    r10_value = r10_value + tl.load($(vmm_1_ptr) + (((r10_row[:, None] % $(128)) * $(128)) + (r10_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r10_value = r10_value.to(tl.bfloat16)
    tl.store($(vaddmm_2_ptr) + ((r10_row[:, None] * $(128)) + r10_col[None, :]), r10_value)
    r11_row = tl.arange($(128))
    r11_col = tl.arange($(256))
    r11_lane = tl.arange($(128))
    r11_left = tl.load($(vaddmm_2_ptr) + (((r11_row[:, None] % $(128)) * $(128)) + (r11_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r11_right = tl.load($(vaddmm_1_ptr) + (((r11_lane[:, None] % $(128)) * $(256)) + (r11_col[None, :] % $(256))), dtype=tl.bfloat16)
    r11_value = tl.dot(r11_left, r11_right)
    r11_value = r11_value + tl.load($(vaddmm_1_ptr) + (((r11_row[:, None] % $(128)) * $(256)) + (r11_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r11_value = r11_value.to(tl.bfloat16)
    tl.store($(vaddmm_3_ptr) + ((r11_row[:, None] * $(256)) + r11_col[None, :]), r11_value)
    r12_row = tl.arange($(128))
    r12_col = tl.arange($(128))
    r12_lane = tl.arange($(256))
    r12_left = tl.load($(vaddmm_3_ptr) + (((r12_row[:, None] % $(128)) * $(256)) + (r12_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r12_right = tl.load($(vaddmm_3_ptr) + (((r12_col[None, :] % $(128)) * $(256)) + (r12_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r12_value = tl.dot(r12_left, r12_right)
    r12_value = r12_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r12_row[:, None] * $(128)) + r12_col[None, :]), r12_value)
    r13_row = tl.arange($(128))
    r13_col = tl.arange($(128))
    r13_lane = tl.arange($(128))
    r13_left = tl.load($(vmm_2_ptr) + (((r13_row[:, None] % $(128)) * $(128)) + (r13_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r13_right = tl.load($(vmm_2_ptr) + (((r13_lane[:, None] % $(128)) * $(128)) + (r13_col[None, :] % $(128))), dtype=tl.bfloat16)
    r13_value = tl.dot(r13_left, r13_right)
    r13_value = r13_value * 2.0315
    r13_value = r13_value + tl.load($(vmm_2_ptr) + (((r13_row[:, None] % $(128)) * $(128)) + (r13_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r13_value = r13_value.to(tl.bfloat16)
    tl.store($(vaddmm_4_ptr) + ((r13_row[:, None] * $(128)) + r13_col[None, :]), r13_value)
    r14_row = tl.arange($(128))
    r14_col = tl.arange($(256))
    r14_lane = tl.arange($(128))
    r14_left = tl.load($(vaddmm_4_ptr) + (((r14_row[:, None] % $(128)) * $(128)) + (r14_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r14_right = tl.load($(vaddmm_3_ptr) + (((r14_lane[:, None] % $(128)) * $(256)) + (r14_col[None, :] % $(256))), dtype=tl.bfloat16)
    r14_value = tl.dot(r14_left, r14_right)
    r14_value = r14_value + tl.load($(vaddmm_3_ptr) + (((r14_row[:, None] % $(128)) * $(256)) + (r14_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r14_value = r14_value.to(tl.bfloat16)
    tl.store($(vaddmm_5_ptr) + ((r14_row[:, None] * $(256)) + r14_col[None, :]), r14_value)
    r15_row = tl.arange($(128))
    r15_col = tl.arange($(128))
    r15_lane = tl.arange($(256))
    r15_left = tl.load($(vaddmm_5_ptr) + (((r15_row[:, None] % $(128)) * $(256)) + (r15_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r15_right = tl.load($(vaddmm_5_ptr) + (((r15_col[None, :] % $(128)) * $(256)) + (r15_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r15_value = tl.dot(r15_left, r15_right)
    r15_value = r15_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r15_row[:, None] * $(128)) + r15_col[None, :]), r15_value)
    r16_row = tl.arange($(128))
    r16_col = tl.arange($(128))
    r16_lane = tl.arange($(128))
    r16_left = tl.load($(vmm_3_ptr) + (((r16_row[:, None] % $(128)) * $(128)) + (r16_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r16_right = tl.load($(vmm_3_ptr) + (((r16_lane[:, None] % $(128)) * $(128)) + (r16_col[None, :] % $(128))), dtype=tl.bfloat16)
    r16_value = tl.dot(r16_left, r16_right)
    r16_value = r16_value * 2.0315
    r16_value = r16_value + tl.load($(vmm_3_ptr) + (((r16_row[:, None] % $(128)) * $(128)) + (r16_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r16_value = r16_value.to(tl.bfloat16)
    tl.store($(vaddmm_6_ptr) + ((r16_row[:, None] * $(128)) + r16_col[None, :]), r16_value)
    r17_row = tl.arange($(128))
    r17_col = tl.arange($(256))
    r17_lane = tl.arange($(128))
    r17_left = tl.load($(vaddmm_6_ptr) + (((r17_row[:, None] % $(128)) * $(128)) + (r17_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r17_right = tl.load($(vaddmm_5_ptr) + (((r17_lane[:, None] % $(128)) * $(256)) + (r17_col[None, :] % $(256))), dtype=tl.bfloat16)
    r17_value = tl.dot(r17_left, r17_right)
    r17_value = r17_value + tl.load($(vaddmm_5_ptr) + (((r17_row[:, None] % $(128)) * $(256)) + (r17_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r17_value = r17_value.to(tl.bfloat16)
    tl.store($(vaddmm_7_ptr) + ((r17_row[:, None] * $(256)) + r17_col[None, :]), r17_value)
    r18_row = tl.arange($(128))
    r18_col = tl.arange($(128))
    r18_lane = tl.arange($(256))
    r18_left = tl.load($(vaddmm_7_ptr) + (((r18_row[:, None] % $(128)) * $(256)) + (r18_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r18_right = tl.load($(vaddmm_7_ptr) + (((r18_col[None, :] % $(128)) * $(256)) + (r18_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r18_value = tl.dot(r18_left, r18_right)
    r18_value = r18_value.to(tl.bfloat16)
    tl.store($(vmm_4_ptr) + ((r18_row[:, None] * $(128)) + r18_col[None, :]), r18_value)
    r19_row = tl.arange($(128))
    r19_col = tl.arange($(128))
    r19_lane = tl.arange($(128))
    r19_left = tl.load($(vmm_4_ptr) + (((r19_row[:, None] % $(128)) * $(128)) + (r19_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r19_right = tl.load($(vmm_4_ptr) + (((r19_lane[:, None] % $(128)) * $(128)) + (r19_col[None, :] % $(128))), dtype=tl.bfloat16)
    r19_value = tl.dot(r19_left, r19_right)
    r19_value = r19_value * 2.0315
    r19_value = r19_value + tl.load($(vmm_4_ptr) + (((r19_row[:, None] % $(128)) * $(128)) + (r19_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r19_value = r19_value.to(tl.bfloat16)
    tl.store($(vaddmm_8_ptr) + ((r19_row[:, None] * $(128)) + r19_col[None, :]), r19_value)
    r20_row = tl.arange($(128))
    r20_col = tl.arange($(256))
    r20_lane = tl.arange($(128))
    r20_left = tl.load($(vaddmm_8_ptr) + (((r20_row[:, None] % $(128)) * $(128)) + (r20_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r20_right = tl.load($(vaddmm_7_ptr) + (((r20_lane[:, None] % $(128)) * $(256)) + (r20_col[None, :] % $(256))), dtype=tl.bfloat16)
    r20_value = tl.dot(r20_left, r20_right)
    r20_value = r20_value + tl.load($(vaddmm_7_ptr) + (((r20_row[:, None] % $(128)) * $(256)) + (r20_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r20_value = r20_value.to(tl.bfloat16)
    tl.store($(vaddmm_9_ptr) + ((r20_row[:, None] * $(256)) + r20_col[None, :]), r20_value)
    r21_row = tl.arange($(128))
    r21_col = tl.arange($(128))
    r21_lane = tl.arange($(256))
    r21_left = tl.load($(vaddmm_9_ptr) + (((r21_row[:, None] % $(128)) * $(256)) + (r21_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r21_right = tl.load($(vaddmm_9_ptr) + (((r21_col[None, :] % $(128)) * $(256)) + (r21_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r21_value = tl.dot(r21_left, r21_right)
    r21_value = r21_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r21_row[:, None] * $(128)) + r21_col[None, :]), r21_value)
    r22_row = tl.arange($(128))
    r22_col = tl.arange($(128))
    r22_lane = tl.arange($(128))
    r22_left = tl.load($(vmm_5_ptr) + (((r22_row[:, None] % $(128)) * $(128)) + (r22_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r22_right = tl.load($(vmm_5_ptr) + (((r22_lane[:, None] % $(128)) * $(128)) + (r22_col[None, :] % $(128))), dtype=tl.bfloat16)
    r22_value = tl.dot(r22_left, r22_right)
    r22_value = r22_value * 2.0315
    r22_value = r22_value + tl.load($(vmm_5_ptr) + (((r22_row[:, None] % $(128)) * $(128)) + (r22_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r22_value = r22_value.to(tl.bfloat16)
    tl.store($(vaddmm_10_ptr) + ((r22_row[:, None] * $(128)) + r22_col[None, :]), r22_value)
    r23_row = tl.arange($(128))
    r23_col = tl.arange($(256))
    r23_lane = tl.arange($(128))
    r23_left = tl.load($(vaddmm_10_ptr) + (((r23_row[:, None] % $(128)) * $(128)) + (r23_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r23_right = tl.load($(vaddmm_9_ptr) + (((r23_lane[:, None] % $(128)) * $(256)) + (r23_col[None, :] % $(256))), dtype=tl.bfloat16)
    r23_value = tl.dot(r23_left, r23_right)
    r23_value = r23_value + tl.load($(vaddmm_9_ptr) + (((r23_row[:, None] % $(128)) * $(256)) + (r23_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r23_value = r23_value.to(tl.bfloat16)
    tl.store($(vaddmm_11_ptr) + ((r23_row[:, None] * $(256)) + r23_col[None, :]), r23_value)
    r24_row = tl.arange($(128))
    r24_col = tl.arange($(128))
    r24_lane = tl.arange($(256))
    r24_left = tl.load($(vaddmm_11_ptr) + (((r24_row[:, None] % $(128)) * $(256)) + (r24_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r24_right = tl.load($(vaddmm_11_ptr) + (((r24_col[None, :] % $(128)) * $(256)) + (r24_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r24_value = tl.dot(r24_left, r24_right)
    r24_value = r24_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r24_row[:, None] * $(128)) + r24_col[None, :]), r24_value)
    r25_row = tl.arange($(128))
    r25_col = tl.arange($(128))
    r25_lane = tl.arange($(128))
    r25_left = tl.load($(vmm_6_ptr) + (((r25_row[:, None] % $(128)) * $(128)) + (r25_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r25_right = tl.load($(vmm_6_ptr) + (((r25_lane[:, None] % $(128)) * $(128)) + (r25_col[None, :] % $(128))), dtype=tl.bfloat16)
    r25_value = tl.dot(r25_left, r25_right)
    r25_value = r25_value * 2.0315
    r25_value = r25_value + tl.load($(vmm_6_ptr) + (((r25_row[:, None] % $(128)) * $(128)) + (r25_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r25_value = r25_value.to(tl.bfloat16)
    tl.store($(vaddmm_12_ptr) + ((r25_row[:, None] * $(128)) + r25_col[None, :]), r25_value)
    r26_row = tl.arange($(128))
    r26_col = tl.arange($(256))
    r26_lane = tl.arange($(128))
    r26_left = tl.load($(vaddmm_12_ptr) + (((r26_row[:, None] % $(128)) * $(128)) + (r26_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r26_right = tl.load($(vaddmm_11_ptr) + (((r26_lane[:, None] % $(128)) * $(256)) + (r26_col[None, :] % $(256))), dtype=tl.bfloat16)
    r26_value = tl.dot(r26_left, r26_right)
    r26_value = r26_value + tl.load($(vaddmm_11_ptr) + (((r26_row[:, None] % $(128)) * $(256)) + (r26_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r26_value = r26_value.to(tl.bfloat16)
    tl.store($(vaddmm_13_ptr) + ((r26_row[:, None] * $(256)) + r26_col[None, :]), r26_value)
    r27_row = tl.arange($(128))
    r27_col = tl.arange($(128))
    r27_lane = tl.arange($(256))
    r27_left = tl.load($(vaddmm_13_ptr) + (((r27_row[:, None] % $(128)) * $(256)) + (r27_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r27_right = tl.load($(vaddmm_13_ptr) + (((r27_col[None, :] % $(128)) * $(256)) + (r27_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r27_value = tl.dot(r27_left, r27_right)
    r27_value = r27_value.to(tl.bfloat16)
    tl.store($(vmm_7_ptr) + ((r27_row[:, None] * $(128)) + r27_col[None, :]), r27_value)
    r28_row = tl.arange($(128))
    r28_col = tl.arange($(128))
    r28_lane = tl.arange($(128))
    r28_left = tl.load($(vmm_7_ptr) + (((r28_row[:, None] % $(128)) * $(128)) + (r28_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r28_right = tl.load($(vmm_7_ptr) + (((r28_lane[:, None] % $(128)) * $(128)) + (r28_col[None, :] % $(128))), dtype=tl.bfloat16)
    r28_value = tl.dot(r28_left, r28_right)
    r28_value = r28_value * 2.0315
    r28_value = r28_value + tl.load($(vmm_7_ptr) + (((r28_row[:, None] % $(128)) * $(128)) + (r28_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r28_value = r28_value.to(tl.bfloat16)
    tl.store($(vaddmm_14_ptr) + ((r28_row[:, None] * $(128)) + r28_col[None, :]), r28_value)
    r29_row = tl.arange($(128))
    r29_col = tl.arange($(256))
    r29_lane = tl.arange($(128))
    r29_left = tl.load($(vaddmm_14_ptr) + (((r29_row[:, None] % $(128)) * $(128)) + (r29_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r29_right = tl.load($(vaddmm_13_ptr) + (((r29_lane[:, None] % $(128)) * $(256)) + (r29_col[None, :] % $(256))), dtype=tl.bfloat16)
    r29_value = tl.dot(r29_left, r29_right)
    r29_value = r29_value + tl.load($(vaddmm_13_ptr) + (((r29_row[:, None] % $(128)) * $(256)) + (r29_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r29_value = r29_value.to(tl.bfloat16)
    tl.store($(vaddmm_15_ptr) + ((r29_row[:, None] * $(256)) + r29_col[None, :]), r29_value)
    r30_row = tl.arange($(128))
    r30_col = tl.arange($(128))
    r30_lane = tl.arange($(256))
    r30_left = tl.load($(vaddmm_15_ptr) + (((r30_row[:, None] % $(128)) * $(256)) + (r30_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r30_right = tl.load($(vaddmm_15_ptr) + (((r30_col[None, :] % $(128)) * $(256)) + (r30_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r30_value = tl.dot(r30_left, r30_right)
    r30_value = r30_value.to(tl.bfloat16)
    tl.store($(vmm_8_ptr) + ((r30_row[:, None] * $(128)) + r30_col[None, :]), r30_value)
    r31_row = tl.arange($(128))
    r31_col = tl.arange($(128))
    r31_lane = tl.arange($(128))
    r31_left = tl.load($(vmm_8_ptr) + (((r31_row[:, None] % $(128)) * $(128)) + (r31_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r31_right = tl.load($(vmm_8_ptr) + (((r31_lane[:, None] % $(128)) * $(128)) + (r31_col[None, :] % $(128))), dtype=tl.bfloat16)
    r31_value = tl.dot(r31_left, r31_right)
    r31_value = r31_value * 0.5
    r31_value = r31_value + tl.load($(vmm_8_ptr) + (((r31_row[:, None] % $(128)) * $(128)) + (r31_col[None, :] % $(128))), dtype=tl.bfloat16) * -1.5
    r31_value = r31_value.to(tl.bfloat16)
    tl.store($(vaddmm_16_ptr) + ((r31_row[:, None] * $(128)) + r31_col[None, :]), r31_value)
    r32_row = tl.arange($(128))
    r32_col = tl.arange($(256))
    r32_lane = tl.arange($(128))
    r32_left = tl.load($(vaddmm_16_ptr) + (((r32_row[:, None] % $(128)) * $(128)) + (r32_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r32_right = tl.load($(vaddmm_15_ptr) + (((r32_lane[:, None] % $(128)) * $(256)) + (r32_col[None, :] % $(256))), dtype=tl.bfloat16)
    r32_value = tl.dot(r32_left, r32_right)
    r32_value = r32_value + tl.load($(vaddmm_15_ptr) + (((r32_row[:, None] % $(128)) * $(256)) + (r32_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
    r32_value = r32_value.to(tl.bfloat16)
    tl.store($(vaddmm_17_ptr) + ((r32_row[:, None] * $(256)) + r32_col[None, :]), r32_value)
    r33_row = tl.arange($(128))
    r33_col = tl.arange($(128))
    r33_lane = tl.arange($(256))
    r33_left = tl.load($(vaddmm_17_ptr) + (((r33_row[:, None] % $(128)) * $(256)) + (r33_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r33_right = tl.load($(vaddmm_17_ptr) + (((r33_col[None, :] % $(128)) * $(256)) + (r33_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r33_value = tl.dot(r33_left, r33_right)
    r33_value = r33_value.to(tl.bfloat16)
    tl.store($(vmm_9_ptr) + ((r33_row[:, None] * $(128)) + r33_col[None, :]), r33_value)
    r34_row = tl.arange($(128))
    r34_col = tl.arange($(128))
    r34_lane = tl.arange($(128))
    r34_left = tl.load($(vmm_9_ptr) + (((r34_row[:, None] % $(128)) * $(128)) + (r34_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r34_right = tl.load($(vmm_9_ptr) + (((r34_lane[:, None] % $(128)) * $(128)) + (r34_col[None, :] % $(128))), dtype=tl.bfloat16)
    r34_value = tl.dot(r34_left, r34_right)
    r34_value = r34_value * 0.5
    r34_value = r34_value + tl.load($(vmm_9_ptr) + (((r34_row[:, None] % $(128)) * $(128)) + (r34_col[None, :] % $(128))), dtype=tl.bfloat16) * -1.5
    r34_value = r34_value.to(tl.bfloat16)
    tl.store($(vaddmm_18_ptr) + ((r34_row[:, None] * $(128)) + r34_col[None, :]), r34_value)
    r35_row = tl.arange($(128))
    r35_col = tl.arange($(256))
    r35_lane = tl.arange($(128))
    r35_left = tl.load($(vaddmm_18_ptr) + (((r35_row[:, None] % $(128)) * $(128)) + (r35_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r35_right = tl.load($(vaddmm_17_ptr) + (((r35_lane[:, None] % $(128)) * $(256)) + (r35_col[None, :] % $(256))), dtype=tl.bfloat16)
    r35_value = tl.dot(r35_left, r35_right)
    r35_value = r35_value + tl.load($(vaddmm_17_ptr) + (((r35_row[:, None] % $(128)) * $(256)) + (r35_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
    r35_value = r35_value.to(tl.bfloat16)
    tl.store($(vaddmm_19_ptr) + ((r35_row[:, None] * $(256)) + r35_col[None, :]), r35_value)
    r36_z = tl.arange($(32768))
    r36_value = (tl.load($(parameter0_ptr) + ((((r36_z / $(128)) % $(256)) * $(128)) + (r36_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(32768)])
    tl.store($(vmul_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(131072))
    r37_value = tl.where((((r37_z / $(32768)) % $(4)) == $(0)), tl.load($(vmul_ptr) + ((((r37_z / $(128)) % $(256)) * $(128)) + (r37_z % $(128))), dtype=tl.bfloat16), tl.load($(parameter0_ptr) + ((((r37_z / $(32768)) % $(4)) * $(32768)) + (((r37_z / $(128)) % $(256)) * $(128)) + (r37_z % $(128))), dtype=tl.bfloat16))
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(131072)])
    tl.store($(vselect_scatter_1_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(32768))
    r38_value = (tl.load($(vselect_scatter_1_ptr) + ((((r38_z / $(128)) % $(256)) * $(128)) + (r38_z % $(128))), dtype=tl.bfloat16) + (tl.load($(vaddmm_19_ptr) + (((r38_z % $(128)) * $(256)) + ((r38_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(32768)])
    tl.store($(vadd_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(131072))
    r39_value = tl.where((((r39_z / $(32768)) % $(4)) == $(0)), tl.load($(vadd_ptr) + ((((r39_z / $(128)) % $(256)) * $(128)) + (r39_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_1_ptr) + ((((r39_z / $(32768)) % $(4)) * $(32768)) + (((r39_z / $(128)) % $(256)) * $(128)) + (r39_z % $(128))), dtype=tl.bfloat16))
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(131072)])
    tl.store($(vselect_scatter_2_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(32768))
    r40_value = (tl.load($(vselect_scatter_ptr) + ((((r40_z / $(128)) % $(256)) * $(128)) + (r40_z % $(128)) + $(32768)), dtype=tl.bfloat16) + (tl.load($(input0_ptr) + ((((r40_z / $(128)) % $(256)) * $(128)) + (r40_z % $(128)) + $(32768)), dtype=tl.bfloat16) - tl.load($(vselect_scatter_ptr) + ((((r40_z / $(128)) % $(256)) * $(128)) + (r40_z % $(128)) + $(32768)), dtype=tl.bfloat16)) * 0.050000000000000044)
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(32768)])
    tl.store($(vlerp_2_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(131072))
    r41_value = tl.where((((r41_z / $(32768)) % $(4)) == $(1)), tl.load($(vlerp_2_ptr) + ((((r41_z / $(128)) % $(256)) * $(128)) + (r41_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_ptr) + ((((r41_z / $(32768)) % $(4)) * $(32768)) + (((r41_z / $(128)) % $(256)) * $(128)) + (r41_z % $(128))), dtype=tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(131072)])
    tl.store($(vselect_scatter_3_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(32768))
    r42_value = (tl.load($(input0_ptr) + ((((r42_z / $(128)) % $(256)) * $(128)) + (r42_z % $(128)) + $(32768)), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_3_ptr) + ((((r42_z / $(128)) % $(256)) * $(128)) + (r42_z % $(128)) + $(32768)), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r42_z / $(128)) % $(256)) * $(128)) + (r42_z % $(128)) + $(32768)), dtype=tl.bfloat16)) * 0.95)
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(32768)])
    tl.store($(vlerp_3_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(1))
    r43_lane = tl.arange($(32768))
    r43_value = tl.sqrt(tl.sum(tl.load($(vlerp_3_ptr) + (((r43_lane[None, :] % $(256)) * $(128)) + ((r43_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16) * tl.load($(vlerp_3_ptr) + (((r43_lane[None, :] % $(256)) * $(128)) + ((r43_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16), axis=1))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(1)])
    tl.store($(vlinalg_vector_norm_1_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(1))
    r44_value = tl.maximum(tl.load($(vlinalg_vector_norm_1_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(1)])
    tl.store($(vclamp_1_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(32768))
    r45_value = (tl.load($(vlerp_3_ptr) + (((r45_z % $(256)) * $(128)) + ((r45_z / $(256)) % $(128))), dtype=tl.bfloat16) / tl.load($(vclamp_1_ptr) + $(0), dtype=tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(32768)])
    tl.store($(vdiv_1_ptr) + r45_z, r45_value)
    r46_row = tl.arange($(128))
    r46_col = tl.arange($(128))
    r46_lane = tl.arange($(256))
    r46_left = tl.load($(vdiv_1_ptr) + (((r46_row[:, None] % $(128)) * $(256)) + (r46_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r46_right = tl.load($(vdiv_1_ptr) + (((r46_col[None, :] % $(128)) * $(256)) + (r46_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r46_value = tl.dot(r46_left, r46_right)
    r46_value = r46_value.to(tl.bfloat16)
    tl.store($(vmm_10_ptr) + ((r46_row[:, None] * $(128)) + r46_col[None, :]), r46_value)
    r47_row = tl.arange($(128))
    r47_col = tl.arange($(128))
    r47_lane = tl.arange($(128))
    r47_left = tl.load($(vmm_10_ptr) + (((r47_row[:, None] % $(128)) * $(128)) + (r47_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r47_right = tl.load($(vmm_10_ptr) + (((r47_lane[:, None] % $(128)) * $(128)) + (r47_col[None, :] % $(128))), dtype=tl.bfloat16)
    r47_value = tl.dot(r47_left, r47_right)
    r47_value = r47_value * 2.0315
    r47_value = r47_value + tl.load($(vmm_10_ptr) + (((r47_row[:, None] % $(128)) * $(128)) + (r47_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r47_value = r47_value.to(tl.bfloat16)
    tl.store($(vaddmm_20_ptr) + ((r47_row[:, None] * $(128)) + r47_col[None, :]), r47_value)
    r48_row = tl.arange($(128))
    r48_col = tl.arange($(256))
    r48_lane = tl.arange($(128))
    r48_left = tl.load($(vaddmm_20_ptr) + (((r48_row[:, None] % $(128)) * $(128)) + (r48_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r48_right = tl.load($(vdiv_1_ptr) + (((r48_lane[:, None] % $(128)) * $(256)) + (r48_col[None, :] % $(256))), dtype=tl.bfloat16)
    r48_value = tl.dot(r48_left, r48_right)
    r48_value = r48_value + tl.load($(vdiv_1_ptr) + (((r48_row[:, None] % $(128)) * $(256)) + (r48_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r48_value = r48_value.to(tl.bfloat16)
    tl.store($(vaddmm_21_ptr) + ((r48_row[:, None] * $(256)) + r48_col[None, :]), r48_value)
    r49_row = tl.arange($(128))
    r49_col = tl.arange($(128))
    r49_lane = tl.arange($(256))
    r49_left = tl.load($(vaddmm_21_ptr) + (((r49_row[:, None] % $(128)) * $(256)) + (r49_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r49_right = tl.load($(vaddmm_21_ptr) + (((r49_col[None, :] % $(128)) * $(256)) + (r49_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r49_value = tl.dot(r49_left, r49_right)
    r49_value = r49_value.to(tl.bfloat16)
    tl.store($(vmm_11_ptr) + ((r49_row[:, None] * $(128)) + r49_col[None, :]), r49_value)
    r50_row = tl.arange($(128))
    r50_col = tl.arange($(128))
    r50_lane = tl.arange($(128))
    r50_left = tl.load($(vmm_11_ptr) + (((r50_row[:, None] % $(128)) * $(128)) + (r50_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r50_right = tl.load($(vmm_11_ptr) + (((r50_lane[:, None] % $(128)) * $(128)) + (r50_col[None, :] % $(128))), dtype=tl.bfloat16)
    r50_value = tl.dot(r50_left, r50_right)
    r50_value = r50_value * 2.0315
    r50_value = r50_value + tl.load($(vmm_11_ptr) + (((r50_row[:, None] % $(128)) * $(128)) + (r50_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r50_value = r50_value.to(tl.bfloat16)
    tl.store($(vaddmm_22_ptr) + ((r50_row[:, None] * $(128)) + r50_col[None, :]), r50_value)
    r51_row = tl.arange($(128))
    r51_col = tl.arange($(256))
    r51_lane = tl.arange($(128))
    r51_left = tl.load($(vaddmm_22_ptr) + (((r51_row[:, None] % $(128)) * $(128)) + (r51_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r51_right = tl.load($(vaddmm_21_ptr) + (((r51_lane[:, None] % $(128)) * $(256)) + (r51_col[None, :] % $(256))), dtype=tl.bfloat16)
    r51_value = tl.dot(r51_left, r51_right)
    r51_value = r51_value + tl.load($(vaddmm_21_ptr) + (((r51_row[:, None] % $(128)) * $(256)) + (r51_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r51_value = r51_value.to(tl.bfloat16)
    tl.store($(vaddmm_23_ptr) + ((r51_row[:, None] * $(256)) + r51_col[None, :]), r51_value)
    r52_row = tl.arange($(128))
    r52_col = tl.arange($(128))
    r52_lane = tl.arange($(256))
    r52_left = tl.load($(vaddmm_23_ptr) + (((r52_row[:, None] % $(128)) * $(256)) + (r52_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r52_right = tl.load($(vaddmm_23_ptr) + (((r52_col[None, :] % $(128)) * $(256)) + (r52_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r52_value = tl.dot(r52_left, r52_right)
    r52_value = r52_value.to(tl.bfloat16)
    tl.store($(vmm_12_ptr) + ((r52_row[:, None] * $(128)) + r52_col[None, :]), r52_value)
    r53_row = tl.arange($(128))
    r53_col = tl.arange($(128))
    r53_lane = tl.arange($(128))
    r53_left = tl.load($(vmm_12_ptr) + (((r53_row[:, None] % $(128)) * $(128)) + (r53_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r53_right = tl.load($(vmm_12_ptr) + (((r53_lane[:, None] % $(128)) * $(128)) + (r53_col[None, :] % $(128))), dtype=tl.bfloat16)
    r53_value = tl.dot(r53_left, r53_right)
    r53_value = r53_value * 2.0315
    r53_value = r53_value + tl.load($(vmm_12_ptr) + (((r53_row[:, None] % $(128)) * $(128)) + (r53_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r53_value = r53_value.to(tl.bfloat16)
    tl.store($(vaddmm_24_ptr) + ((r53_row[:, None] * $(128)) + r53_col[None, :]), r53_value)
    r54_row = tl.arange($(128))
    r54_col = tl.arange($(256))
    r54_lane = tl.arange($(128))
    r54_left = tl.load($(vaddmm_24_ptr) + (((r54_row[:, None] % $(128)) * $(128)) + (r54_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r54_right = tl.load($(vaddmm_23_ptr) + (((r54_lane[:, None] % $(128)) * $(256)) + (r54_col[None, :] % $(256))), dtype=tl.bfloat16)
    r54_value = tl.dot(r54_left, r54_right)
    r54_value = r54_value + tl.load($(vaddmm_23_ptr) + (((r54_row[:, None] % $(128)) * $(256)) + (r54_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r54_value = r54_value.to(tl.bfloat16)
    tl.store($(vaddmm_25_ptr) + ((r54_row[:, None] * $(256)) + r54_col[None, :]), r54_value)
    r55_row = tl.arange($(128))
    r55_col = tl.arange($(128))
    r55_lane = tl.arange($(256))
    r55_left = tl.load($(vaddmm_25_ptr) + (((r55_row[:, None] % $(128)) * $(256)) + (r55_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r55_right = tl.load($(vaddmm_25_ptr) + (((r55_col[None, :] % $(128)) * $(256)) + (r55_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r55_value = tl.dot(r55_left, r55_right)
    r55_value = r55_value.to(tl.bfloat16)
    tl.store($(vmm_13_ptr) + ((r55_row[:, None] * $(128)) + r55_col[None, :]), r55_value)
    r56_row = tl.arange($(128))
    r56_col = tl.arange($(128))
    r56_lane = tl.arange($(128))
    r56_left = tl.load($(vmm_13_ptr) + (((r56_row[:, None] % $(128)) * $(128)) + (r56_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r56_right = tl.load($(vmm_13_ptr) + (((r56_lane[:, None] % $(128)) * $(128)) + (r56_col[None, :] % $(128))), dtype=tl.bfloat16)
    r56_value = tl.dot(r56_left, r56_right)
    r56_value = r56_value * 2.0315
    r56_value = r56_value + tl.load($(vmm_13_ptr) + (((r56_row[:, None] % $(128)) * $(128)) + (r56_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r56_value = r56_value.to(tl.bfloat16)
    tl.store($(vaddmm_26_ptr) + ((r56_row[:, None] * $(128)) + r56_col[None, :]), r56_value)
    r57_row = tl.arange($(128))
    r57_col = tl.arange($(256))
    r57_lane = tl.arange($(128))
    r57_left = tl.load($(vaddmm_26_ptr) + (((r57_row[:, None] % $(128)) * $(128)) + (r57_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r57_right = tl.load($(vaddmm_25_ptr) + (((r57_lane[:, None] % $(128)) * $(256)) + (r57_col[None, :] % $(256))), dtype=tl.bfloat16)
    r57_value = tl.dot(r57_left, r57_right)
    r57_value = r57_value + tl.load($(vaddmm_25_ptr) + (((r57_row[:, None] % $(128)) * $(256)) + (r57_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r57_value = r57_value.to(tl.bfloat16)
    tl.store($(vaddmm_27_ptr) + ((r57_row[:, None] * $(256)) + r57_col[None, :]), r57_value)
    r58_row = tl.arange($(128))
    r58_col = tl.arange($(128))
    r58_lane = tl.arange($(256))
    r58_left = tl.load($(vaddmm_27_ptr) + (((r58_row[:, None] % $(128)) * $(256)) + (r58_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r58_right = tl.load($(vaddmm_27_ptr) + (((r58_col[None, :] % $(128)) * $(256)) + (r58_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r58_value = tl.dot(r58_left, r58_right)
    r58_value = r58_value.to(tl.bfloat16)
    tl.store($(vmm_14_ptr) + ((r58_row[:, None] * $(128)) + r58_col[None, :]), r58_value)
    r59_row = tl.arange($(128))
    r59_col = tl.arange($(128))
    r59_lane = tl.arange($(128))
    r59_left = tl.load($(vmm_14_ptr) + (((r59_row[:, None] % $(128)) * $(128)) + (r59_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r59_right = tl.load($(vmm_14_ptr) + (((r59_lane[:, None] % $(128)) * $(128)) + (r59_col[None, :] % $(128))), dtype=tl.bfloat16)
    r59_value = tl.dot(r59_left, r59_right)
    r59_value = r59_value * 2.0315
    r59_value = r59_value + tl.load($(vmm_14_ptr) + (((r59_row[:, None] % $(128)) * $(128)) + (r59_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r59_value = r59_value.to(tl.bfloat16)
    tl.store($(vaddmm_28_ptr) + ((r59_row[:, None] * $(128)) + r59_col[None, :]), r59_value)
    r60_row = tl.arange($(128))
    r60_col = tl.arange($(256))
    r60_lane = tl.arange($(128))
    r60_left = tl.load($(vaddmm_28_ptr) + (((r60_row[:, None] % $(128)) * $(128)) + (r60_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r60_right = tl.load($(vaddmm_27_ptr) + (((r60_lane[:, None] % $(128)) * $(256)) + (r60_col[None, :] % $(256))), dtype=tl.bfloat16)
    r60_value = tl.dot(r60_left, r60_right)
    r60_value = r60_value + tl.load($(vaddmm_27_ptr) + (((r60_row[:, None] % $(128)) * $(256)) + (r60_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r60_value = r60_value.to(tl.bfloat16)
    tl.store($(vaddmm_29_ptr) + ((r60_row[:, None] * $(256)) + r60_col[None, :]), r60_value)
    r61_row = tl.arange($(128))
    r61_col = tl.arange($(128))
    r61_lane = tl.arange($(256))
    r61_left = tl.load($(vaddmm_29_ptr) + (((r61_row[:, None] % $(128)) * $(256)) + (r61_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r61_right = tl.load($(vaddmm_29_ptr) + (((r61_col[None, :] % $(128)) * $(256)) + (r61_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r61_value = tl.dot(r61_left, r61_right)
    r61_value = r61_value.to(tl.bfloat16)
    tl.store($(vmm_15_ptr) + ((r61_row[:, None] * $(128)) + r61_col[None, :]), r61_value)
    r62_row = tl.arange($(128))
    r62_col = tl.arange($(128))
    r62_lane = tl.arange($(128))
    r62_left = tl.load($(vmm_15_ptr) + (((r62_row[:, None] % $(128)) * $(128)) + (r62_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r62_right = tl.load($(vmm_15_ptr) + (((r62_lane[:, None] % $(128)) * $(128)) + (r62_col[None, :] % $(128))), dtype=tl.bfloat16)
    r62_value = tl.dot(r62_left, r62_right)
    r62_value = r62_value * 2.0315
    r62_value = r62_value + tl.load($(vmm_15_ptr) + (((r62_row[:, None] % $(128)) * $(128)) + (r62_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r62_value = r62_value.to(tl.bfloat16)
    tl.store($(vaddmm_30_ptr) + ((r62_row[:, None] * $(128)) + r62_col[None, :]), r62_value)
    r63_row = tl.arange($(128))
    r63_col = tl.arange($(256))
    r63_lane = tl.arange($(128))
    r63_left = tl.load($(vaddmm_30_ptr) + (((r63_row[:, None] % $(128)) * $(128)) + (r63_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r63_right = tl.load($(vaddmm_29_ptr) + (((r63_lane[:, None] % $(128)) * $(256)) + (r63_col[None, :] % $(256))), dtype=tl.bfloat16)
    r63_value = tl.dot(r63_left, r63_right)
    r63_value = r63_value + tl.load($(vaddmm_29_ptr) + (((r63_row[:, None] % $(128)) * $(256)) + (r63_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r63_value = r63_value.to(tl.bfloat16)
    tl.store($(vaddmm_31_ptr) + ((r63_row[:, None] * $(256)) + r63_col[None, :]), r63_value)
    r64_row = tl.arange($(128))
    r64_col = tl.arange($(128))
    r64_lane = tl.arange($(256))
    r64_left = tl.load($(vaddmm_31_ptr) + (((r64_row[:, None] % $(128)) * $(256)) + (r64_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r64_right = tl.load($(vaddmm_31_ptr) + (((r64_col[None, :] % $(128)) * $(256)) + (r64_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r64_value = tl.dot(r64_left, r64_right)
    r64_value = r64_value.to(tl.bfloat16)
    tl.store($(vmm_16_ptr) + ((r64_row[:, None] * $(128)) + r64_col[None, :]), r64_value)
    r65_row = tl.arange($(128))
    r65_col = tl.arange($(128))
    r65_lane = tl.arange($(128))
    r65_left = tl.load($(vmm_16_ptr) + (((r65_row[:, None] % $(128)) * $(128)) + (r65_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r65_right = tl.load($(vmm_16_ptr) + (((r65_lane[:, None] % $(128)) * $(128)) + (r65_col[None, :] % $(128))), dtype=tl.bfloat16)
    r65_value = tl.dot(r65_left, r65_right)
    r65_value = r65_value * 2.0315
    r65_value = r65_value + tl.load($(vmm_16_ptr) + (((r65_row[:, None] % $(128)) * $(128)) + (r65_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r65_value = r65_value.to(tl.bfloat16)
    tl.store($(vaddmm_32_ptr) + ((r65_row[:, None] * $(128)) + r65_col[None, :]), r65_value)
    r66_row = tl.arange($(128))
    r66_col = tl.arange($(256))
    r66_lane = tl.arange($(128))
    r66_left = tl.load($(vaddmm_32_ptr) + (((r66_row[:, None] % $(128)) * $(128)) + (r66_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r66_right = tl.load($(vaddmm_31_ptr) + (((r66_lane[:, None] % $(128)) * $(256)) + (r66_col[None, :] % $(256))), dtype=tl.bfloat16)
    r66_value = tl.dot(r66_left, r66_right)
    r66_value = r66_value + tl.load($(vaddmm_31_ptr) + (((r66_row[:, None] % $(128)) * $(256)) + (r66_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r66_value = r66_value.to(tl.bfloat16)
    tl.store($(vaddmm_33_ptr) + ((r66_row[:, None] * $(256)) + r66_col[None, :]), r66_value)
    r67_row = tl.arange($(128))
    r67_col = tl.arange($(128))
    r67_lane = tl.arange($(256))
    r67_left = tl.load($(vaddmm_33_ptr) + (((r67_row[:, None] % $(128)) * $(256)) + (r67_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r67_right = tl.load($(vaddmm_33_ptr) + (((r67_col[None, :] % $(128)) * $(256)) + (r67_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r67_value = tl.dot(r67_left, r67_right)
    r67_value = r67_value.to(tl.bfloat16)
    tl.store($(vmm_17_ptr) + ((r67_row[:, None] * $(128)) + r67_col[None, :]), r67_value)
    r68_row = tl.arange($(128))
    r68_col = tl.arange($(128))
    r68_lane = tl.arange($(128))
    r68_left = tl.load($(vmm_17_ptr) + (((r68_row[:, None] % $(128)) * $(128)) + (r68_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r68_right = tl.load($(vmm_17_ptr) + (((r68_lane[:, None] % $(128)) * $(128)) + (r68_col[None, :] % $(128))), dtype=tl.bfloat16)
    r68_value = tl.dot(r68_left, r68_right)
    r68_value = r68_value * 2.0315
    r68_value = r68_value + tl.load($(vmm_17_ptr) + (((r68_row[:, None] % $(128)) * $(128)) + (r68_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r68_value = r68_value.to(tl.bfloat16)
    tl.store($(vaddmm_34_ptr) + ((r68_row[:, None] * $(128)) + r68_col[None, :]), r68_value)
    r69_row = tl.arange($(128))
    r69_col = tl.arange($(256))
    r69_lane = tl.arange($(128))
    r69_left = tl.load($(vaddmm_34_ptr) + (((r69_row[:, None] % $(128)) * $(128)) + (r69_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r69_right = tl.load($(vaddmm_33_ptr) + (((r69_lane[:, None] % $(128)) * $(256)) + (r69_col[None, :] % $(256))), dtype=tl.bfloat16)
    r69_value = tl.dot(r69_left, r69_right)
    r69_value = r69_value + tl.load($(vaddmm_33_ptr) + (((r69_row[:, None] % $(128)) * $(256)) + (r69_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r69_value = r69_value.to(tl.bfloat16)
    tl.store($(vaddmm_35_ptr) + ((r69_row[:, None] * $(256)) + r69_col[None, :]), r69_value)
    r70_row = tl.arange($(128))
    r70_col = tl.arange($(128))
    r70_lane = tl.arange($(256))
    r70_left = tl.load($(vaddmm_35_ptr) + (((r70_row[:, None] % $(128)) * $(256)) + (r70_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r70_right = tl.load($(vaddmm_35_ptr) + (((r70_col[None, :] % $(128)) * $(256)) + (r70_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r70_value = tl.dot(r70_left, r70_right)
    r70_value = r70_value.to(tl.bfloat16)
    tl.store($(vmm_18_ptr) + ((r70_row[:, None] * $(128)) + r70_col[None, :]), r70_value)
    r71_row = tl.arange($(128))
    r71_col = tl.arange($(128))
    r71_lane = tl.arange($(128))
    r71_left = tl.load($(vmm_18_ptr) + (((r71_row[:, None] % $(128)) * $(128)) + (r71_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r71_right = tl.load($(vmm_18_ptr) + (((r71_lane[:, None] % $(128)) * $(128)) + (r71_col[None, :] % $(128))), dtype=tl.bfloat16)
    r71_value = tl.dot(r71_left, r71_right)
    r71_value = r71_value * 0.5
    r71_value = r71_value + tl.load($(vmm_18_ptr) + (((r71_row[:, None] % $(128)) * $(128)) + (r71_col[None, :] % $(128))), dtype=tl.bfloat16) * -1.5
    r71_value = r71_value.to(tl.bfloat16)
    tl.store($(vaddmm_36_ptr) + ((r71_row[:, None] * $(128)) + r71_col[None, :]), r71_value)
    r72_row = tl.arange($(128))
    r72_col = tl.arange($(256))
    r72_lane = tl.arange($(128))
    r72_left = tl.load($(vaddmm_36_ptr) + (((r72_row[:, None] % $(128)) * $(128)) + (r72_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r72_right = tl.load($(vaddmm_35_ptr) + (((r72_lane[:, None] % $(128)) * $(256)) + (r72_col[None, :] % $(256))), dtype=tl.bfloat16)
    r72_value = tl.dot(r72_left, r72_right)
    r72_value = r72_value + tl.load($(vaddmm_35_ptr) + (((r72_row[:, None] % $(128)) * $(256)) + (r72_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
    r72_value = r72_value.to(tl.bfloat16)
    tl.store($(vaddmm_37_ptr) + ((r72_row[:, None] * $(256)) + r72_col[None, :]), r72_value)
    r73_row = tl.arange($(128))
    r73_col = tl.arange($(128))
    r73_lane = tl.arange($(256))
    r73_left = tl.load($(vaddmm_37_ptr) + (((r73_row[:, None] % $(128)) * $(256)) + (r73_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r73_right = tl.load($(vaddmm_37_ptr) + (((r73_col[None, :] % $(128)) * $(256)) + (r73_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r73_value = tl.dot(r73_left, r73_right)
    r73_value = r73_value.to(tl.bfloat16)
    tl.store($(vmm_19_ptr) + ((r73_row[:, None] * $(128)) + r73_col[None, :]), r73_value)
    r74_row = tl.arange($(128))
    r74_col = tl.arange($(128))
    r74_lane = tl.arange($(128))
    r74_left = tl.load($(vmm_19_ptr) + (((r74_row[:, None] % $(128)) * $(128)) + (r74_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r74_right = tl.load($(vmm_19_ptr) + (((r74_lane[:, None] % $(128)) * $(128)) + (r74_col[None, :] % $(128))), dtype=tl.bfloat16)
    r74_value = tl.dot(r74_left, r74_right)
    r74_value = r74_value * 0.5
    r74_value = r74_value + tl.load($(vmm_19_ptr) + (((r74_row[:, None] % $(128)) * $(128)) + (r74_col[None, :] % $(128))), dtype=tl.bfloat16) * -1.5
    r74_value = r74_value.to(tl.bfloat16)
    tl.store($(vaddmm_38_ptr) + ((r74_row[:, None] * $(128)) + r74_col[None, :]), r74_value)
    r75_row = tl.arange($(128))
    r75_col = tl.arange($(256))
    r75_lane = tl.arange($(128))
    r75_left = tl.load($(vaddmm_38_ptr) + (((r75_row[:, None] % $(128)) * $(128)) + (r75_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r75_right = tl.load($(vaddmm_37_ptr) + (((r75_lane[:, None] % $(128)) * $(256)) + (r75_col[None, :] % $(256))), dtype=tl.bfloat16)
    r75_value = tl.dot(r75_left, r75_right)
    r75_value = r75_value + tl.load($(vaddmm_37_ptr) + (((r75_row[:, None] % $(128)) * $(256)) + (r75_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
    r75_value = r75_value.to(tl.bfloat16)
    tl.store($(vaddmm_39_ptr) + ((r75_row[:, None] * $(256)) + r75_col[None, :]), r75_value)
    r76_z = tl.arange($(32768))
    r76_value = (tl.load($(vselect_scatter_2_ptr) + ((((r76_z / $(128)) % $(256)) * $(128)) + (r76_z % $(128)) + $(32768)), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r76_value = r76_value.to(tl.bfloat16)
    r76_value = tl.broadcast_to(r76_value, [$(32768)])
    tl.store($(vmul_1_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(131072))
    r77_value = tl.where((((r77_z / $(32768)) % $(4)) == $(1)), tl.load($(vmul_1_ptr) + ((((r77_z / $(128)) % $(256)) * $(128)) + (r77_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_2_ptr) + ((((r77_z / $(32768)) % $(4)) * $(32768)) + (((r77_z / $(128)) % $(256)) * $(128)) + (r77_z % $(128))), dtype=tl.bfloat16))
    r77_value = r77_value.to(tl.bfloat16)
    r77_value = tl.broadcast_to(r77_value, [$(131072)])
    tl.store($(vselect_scatter_4_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(32768))
    r78_value = (tl.load($(vselect_scatter_4_ptr) + ((((r78_z / $(128)) % $(256)) * $(128)) + (r78_z % $(128)) + $(32768)), dtype=tl.bfloat16) + (tl.load($(vaddmm_39_ptr) + (((r78_z % $(128)) * $(256)) + ((r78_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r78_value = r78_value.to(tl.bfloat16)
    r78_value = tl.broadcast_to(r78_value, [$(32768)])
    tl.store($(vadd_1_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(131072))
    r79_value = tl.where((((r79_z / $(32768)) % $(4)) == $(1)), tl.load($(vadd_1_ptr) + ((((r79_z / $(128)) % $(256)) * $(128)) + (r79_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_4_ptr) + ((((r79_z / $(32768)) % $(4)) * $(32768)) + (((r79_z / $(128)) % $(256)) * $(128)) + (r79_z % $(128))), dtype=tl.bfloat16))
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(131072)])
    tl.store($(vselect_scatter_5_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(32768))
    r80_value = (tl.load($(vselect_scatter_3_ptr) + ((((r80_z / $(128)) % $(256)) * $(128)) + (r80_z % $(128)) + $(65536)), dtype=tl.bfloat16) + (tl.load($(input0_ptr) + ((((r80_z / $(128)) % $(256)) * $(128)) + (r80_z % $(128)) + $(65536)), dtype=tl.bfloat16) - tl.load($(vselect_scatter_3_ptr) + ((((r80_z / $(128)) % $(256)) * $(128)) + (r80_z % $(128)) + $(65536)), dtype=tl.bfloat16)) * 0.050000000000000044)
    r80_value = r80_value.to(tl.bfloat16)
    r80_value = tl.broadcast_to(r80_value, [$(32768)])
    tl.store($(vlerp_4_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(131072))
    r81_value = tl.where((((r81_z / $(32768)) % $(4)) == $(2)), tl.load($(vlerp_4_ptr) + ((((r81_z / $(128)) % $(256)) * $(128)) + (r81_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_3_ptr) + ((((r81_z / $(32768)) % $(4)) * $(32768)) + (((r81_z / $(128)) % $(256)) * $(128)) + (r81_z % $(128))), dtype=tl.bfloat16))
    r81_value = r81_value.to(tl.bfloat16)
    r81_value = tl.broadcast_to(r81_value, [$(131072)])
    tl.store($(vselect_scatter_6_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(32768))
    r82_value = (tl.load($(input0_ptr) + ((((r82_z / $(128)) % $(256)) * $(128)) + (r82_z % $(128)) + $(65536)), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_6_ptr) + ((((r82_z / $(128)) % $(256)) * $(128)) + (r82_z % $(128)) + $(65536)), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r82_z / $(128)) % $(256)) * $(128)) + (r82_z % $(128)) + $(65536)), dtype=tl.bfloat16)) * 0.95)
    r82_value = r82_value.to(tl.bfloat16)
    r82_value = tl.broadcast_to(r82_value, [$(32768)])
    tl.store($(vlerp_5_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(1))
    r83_lane = tl.arange($(32768))
    r83_value = tl.sqrt(tl.sum(tl.load($(vlerp_5_ptr) + (((r83_lane[None, :] % $(256)) * $(128)) + ((r83_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16) * tl.load($(vlerp_5_ptr) + (((r83_lane[None, :] % $(256)) * $(128)) + ((r83_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16), axis=1))
    r83_value = r83_value.to(tl.bfloat16)
    r83_value = tl.broadcast_to(r83_value, [$(1)])
    tl.store($(vlinalg_vector_norm_2_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(1))
    r84_value = tl.maximum(tl.load($(vlinalg_vector_norm_2_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r84_value = r84_value.to(tl.bfloat16)
    r84_value = tl.broadcast_to(r84_value, [$(1)])
    tl.store($(vclamp_2_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(32768))
    r85_value = (tl.load($(vlerp_5_ptr) + (((r85_z % $(256)) * $(128)) + ((r85_z / $(256)) % $(128))), dtype=tl.bfloat16) / tl.load($(vclamp_2_ptr) + $(0), dtype=tl.bfloat16))
    r85_value = r85_value.to(tl.bfloat16)
    r85_value = tl.broadcast_to(r85_value, [$(32768)])
    tl.store($(vdiv_2_ptr) + r85_z, r85_value)
    r86_row = tl.arange($(128))
    r86_col = tl.arange($(128))
    r86_lane = tl.arange($(256))
    r86_left = tl.load($(vdiv_2_ptr) + (((r86_row[:, None] % $(128)) * $(256)) + (r86_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r86_right = tl.load($(vdiv_2_ptr) + (((r86_col[None, :] % $(128)) * $(256)) + (r86_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r86_value = tl.dot(r86_left, r86_right)
    r86_value = r86_value.to(tl.bfloat16)
    tl.store($(vmm_20_ptr) + ((r86_row[:, None] * $(128)) + r86_col[None, :]), r86_value)
    r87_row = tl.arange($(128))
    r87_col = tl.arange($(128))
    r87_lane = tl.arange($(128))
    r87_left = tl.load($(vmm_20_ptr) + (((r87_row[:, None] % $(128)) * $(128)) + (r87_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r87_right = tl.load($(vmm_20_ptr) + (((r87_lane[:, None] % $(128)) * $(128)) + (r87_col[None, :] % $(128))), dtype=tl.bfloat16)
    r87_value = tl.dot(r87_left, r87_right)
    r87_value = r87_value * 2.0315
    r87_value = r87_value + tl.load($(vmm_20_ptr) + (((r87_row[:, None] % $(128)) * $(128)) + (r87_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r87_value = r87_value.to(tl.bfloat16)
    tl.store($(vaddmm_40_ptr) + ((r87_row[:, None] * $(128)) + r87_col[None, :]), r87_value)
    r88_row = tl.arange($(128))
    r88_col = tl.arange($(256))
    r88_lane = tl.arange($(128))
    r88_left = tl.load($(vaddmm_40_ptr) + (((r88_row[:, None] % $(128)) * $(128)) + (r88_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r88_right = tl.load($(vdiv_2_ptr) + (((r88_lane[:, None] % $(128)) * $(256)) + (r88_col[None, :] % $(256))), dtype=tl.bfloat16)
    r88_value = tl.dot(r88_left, r88_right)
    r88_value = r88_value + tl.load($(vdiv_2_ptr) + (((r88_row[:, None] % $(128)) * $(256)) + (r88_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r88_value = r88_value.to(tl.bfloat16)
    tl.store($(vaddmm_41_ptr) + ((r88_row[:, None] * $(256)) + r88_col[None, :]), r88_value)
    r89_row = tl.arange($(128))
    r89_col = tl.arange($(128))
    r89_lane = tl.arange($(256))
    r89_left = tl.load($(vaddmm_41_ptr) + (((r89_row[:, None] % $(128)) * $(256)) + (r89_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r89_right = tl.load($(vaddmm_41_ptr) + (((r89_col[None, :] % $(128)) * $(256)) + (r89_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r89_value = tl.dot(r89_left, r89_right)
    r89_value = r89_value.to(tl.bfloat16)
    tl.store($(vmm_21_ptr) + ((r89_row[:, None] * $(128)) + r89_col[None, :]), r89_value)
    r90_row = tl.arange($(128))
    r90_col = tl.arange($(128))
    r90_lane = tl.arange($(128))
    r90_left = tl.load($(vmm_21_ptr) + (((r90_row[:, None] % $(128)) * $(128)) + (r90_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r90_right = tl.load($(vmm_21_ptr) + (((r90_lane[:, None] % $(128)) * $(128)) + (r90_col[None, :] % $(128))), dtype=tl.bfloat16)
    r90_value = tl.dot(r90_left, r90_right)
    r90_value = r90_value * 2.0315
    r90_value = r90_value + tl.load($(vmm_21_ptr) + (((r90_row[:, None] % $(128)) * $(128)) + (r90_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r90_value = r90_value.to(tl.bfloat16)
    tl.store($(vaddmm_42_ptr) + ((r90_row[:, None] * $(128)) + r90_col[None, :]), r90_value)
    r91_row = tl.arange($(128))
    r91_col = tl.arange($(256))
    r91_lane = tl.arange($(128))
    r91_left = tl.load($(vaddmm_42_ptr) + (((r91_row[:, None] % $(128)) * $(128)) + (r91_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r91_right = tl.load($(vaddmm_41_ptr) + (((r91_lane[:, None] % $(128)) * $(256)) + (r91_col[None, :] % $(256))), dtype=tl.bfloat16)
    r91_value = tl.dot(r91_left, r91_right)
    r91_value = r91_value + tl.load($(vaddmm_41_ptr) + (((r91_row[:, None] % $(128)) * $(256)) + (r91_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r91_value = r91_value.to(tl.bfloat16)
    tl.store($(vaddmm_43_ptr) + ((r91_row[:, None] * $(256)) + r91_col[None, :]), r91_value)
    r92_row = tl.arange($(128))
    r92_col = tl.arange($(128))
    r92_lane = tl.arange($(256))
    r92_left = tl.load($(vaddmm_43_ptr) + (((r92_row[:, None] % $(128)) * $(256)) + (r92_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r92_right = tl.load($(vaddmm_43_ptr) + (((r92_col[None, :] % $(128)) * $(256)) + (r92_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r92_value = tl.dot(r92_left, r92_right)
    r92_value = r92_value.to(tl.bfloat16)
    tl.store($(vmm_22_ptr) + ((r92_row[:, None] * $(128)) + r92_col[None, :]), r92_value)
    r93_row = tl.arange($(128))
    r93_col = tl.arange($(128))
    r93_lane = tl.arange($(128))
    r93_left = tl.load($(vmm_22_ptr) + (((r93_row[:, None] % $(128)) * $(128)) + (r93_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r93_right = tl.load($(vmm_22_ptr) + (((r93_lane[:, None] % $(128)) * $(128)) + (r93_col[None, :] % $(128))), dtype=tl.bfloat16)
    r93_value = tl.dot(r93_left, r93_right)
    r93_value = r93_value * 2.0315
    r93_value = r93_value + tl.load($(vmm_22_ptr) + (((r93_row[:, None] % $(128)) * $(128)) + (r93_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r93_value = r93_value.to(tl.bfloat16)
    tl.store($(vaddmm_44_ptr) + ((r93_row[:, None] * $(128)) + r93_col[None, :]), r93_value)
    r94_row = tl.arange($(128))
    r94_col = tl.arange($(256))
    r94_lane = tl.arange($(128))
    r94_left = tl.load($(vaddmm_44_ptr) + (((r94_row[:, None] % $(128)) * $(128)) + (r94_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r94_right = tl.load($(vaddmm_43_ptr) + (((r94_lane[:, None] % $(128)) * $(256)) + (r94_col[None, :] % $(256))), dtype=tl.bfloat16)
    r94_value = tl.dot(r94_left, r94_right)
    r94_value = r94_value + tl.load($(vaddmm_43_ptr) + (((r94_row[:, None] % $(128)) * $(256)) + (r94_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r94_value = r94_value.to(tl.bfloat16)
    tl.store($(vaddmm_45_ptr) + ((r94_row[:, None] * $(256)) + r94_col[None, :]), r94_value)
    r95_row = tl.arange($(128))
    r95_col = tl.arange($(128))
    r95_lane = tl.arange($(256))
    r95_left = tl.load($(vaddmm_45_ptr) + (((r95_row[:, None] % $(128)) * $(256)) + (r95_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r95_right = tl.load($(vaddmm_45_ptr) + (((r95_col[None, :] % $(128)) * $(256)) + (r95_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r95_value = tl.dot(r95_left, r95_right)
    r95_value = r95_value.to(tl.bfloat16)
    tl.store($(vmm_23_ptr) + ((r95_row[:, None] * $(128)) + r95_col[None, :]), r95_value)
    r96_row = tl.arange($(128))
    r96_col = tl.arange($(128))
    r96_lane = tl.arange($(128))
    r96_left = tl.load($(vmm_23_ptr) + (((r96_row[:, None] % $(128)) * $(128)) + (r96_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r96_right = tl.load($(vmm_23_ptr) + (((r96_lane[:, None] % $(128)) * $(128)) + (r96_col[None, :] % $(128))), dtype=tl.bfloat16)
    r96_value = tl.dot(r96_left, r96_right)
    r96_value = r96_value * 2.0315
    r96_value = r96_value + tl.load($(vmm_23_ptr) + (((r96_row[:, None] % $(128)) * $(128)) + (r96_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r96_value = r96_value.to(tl.bfloat16)
    tl.store($(vaddmm_46_ptr) + ((r96_row[:, None] * $(128)) + r96_col[None, :]), r96_value)
    r97_row = tl.arange($(128))
    r97_col = tl.arange($(256))
    r97_lane = tl.arange($(128))
    r97_left = tl.load($(vaddmm_46_ptr) + (((r97_row[:, None] % $(128)) * $(128)) + (r97_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r97_right = tl.load($(vaddmm_45_ptr) + (((r97_lane[:, None] % $(128)) * $(256)) + (r97_col[None, :] % $(256))), dtype=tl.bfloat16)
    r97_value = tl.dot(r97_left, r97_right)
    r97_value = r97_value + tl.load($(vaddmm_45_ptr) + (((r97_row[:, None] % $(128)) * $(256)) + (r97_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r97_value = r97_value.to(tl.bfloat16)
    tl.store($(vaddmm_47_ptr) + ((r97_row[:, None] * $(256)) + r97_col[None, :]), r97_value)
    r98_row = tl.arange($(128))
    r98_col = tl.arange($(128))
    r98_lane = tl.arange($(256))
    r98_left = tl.load($(vaddmm_47_ptr) + (((r98_row[:, None] % $(128)) * $(256)) + (r98_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r98_right = tl.load($(vaddmm_47_ptr) + (((r98_col[None, :] % $(128)) * $(256)) + (r98_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r98_value = tl.dot(r98_left, r98_right)
    r98_value = r98_value.to(tl.bfloat16)
    tl.store($(vmm_24_ptr) + ((r98_row[:, None] * $(128)) + r98_col[None, :]), r98_value)
    r99_row = tl.arange($(128))
    r99_col = tl.arange($(128))
    r99_lane = tl.arange($(128))
    r99_left = tl.load($(vmm_24_ptr) + (((r99_row[:, None] % $(128)) * $(128)) + (r99_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r99_right = tl.load($(vmm_24_ptr) + (((r99_lane[:, None] % $(128)) * $(128)) + (r99_col[None, :] % $(128))), dtype=tl.bfloat16)
    r99_value = tl.dot(r99_left, r99_right)
    r99_value = r99_value * 2.0315
    r99_value = r99_value + tl.load($(vmm_24_ptr) + (((r99_row[:, None] % $(128)) * $(128)) + (r99_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r99_value = r99_value.to(tl.bfloat16)
    tl.store($(vaddmm_48_ptr) + ((r99_row[:, None] * $(128)) + r99_col[None, :]), r99_value)
    r100_row = tl.arange($(128))
    r100_col = tl.arange($(256))
    r100_lane = tl.arange($(128))
    r100_left = tl.load($(vaddmm_48_ptr) + (((r100_row[:, None] % $(128)) * $(128)) + (r100_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r100_right = tl.load($(vaddmm_47_ptr) + (((r100_lane[:, None] % $(128)) * $(256)) + (r100_col[None, :] % $(256))), dtype=tl.bfloat16)
    r100_value = tl.dot(r100_left, r100_right)
    r100_value = r100_value + tl.load($(vaddmm_47_ptr) + (((r100_row[:, None] % $(128)) * $(256)) + (r100_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r100_value = r100_value.to(tl.bfloat16)
    tl.store($(vaddmm_49_ptr) + ((r100_row[:, None] * $(256)) + r100_col[None, :]), r100_value)
    r101_row = tl.arange($(128))
    r101_col = tl.arange($(128))
    r101_lane = tl.arange($(256))
    r101_left = tl.load($(vaddmm_49_ptr) + (((r101_row[:, None] % $(128)) * $(256)) + (r101_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r101_right = tl.load($(vaddmm_49_ptr) + (((r101_col[None, :] % $(128)) * $(256)) + (r101_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r101_value = tl.dot(r101_left, r101_right)
    r101_value = r101_value.to(tl.bfloat16)
    tl.store($(vmm_25_ptr) + ((r101_row[:, None] * $(128)) + r101_col[None, :]), r101_value)
    r102_row = tl.arange($(128))
    r102_col = tl.arange($(128))
    r102_lane = tl.arange($(128))
    r102_left = tl.load($(vmm_25_ptr) + (((r102_row[:, None] % $(128)) * $(128)) + (r102_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r102_right = tl.load($(vmm_25_ptr) + (((r102_lane[:, None] % $(128)) * $(128)) + (r102_col[None, :] % $(128))), dtype=tl.bfloat16)
    r102_value = tl.dot(r102_left, r102_right)
    r102_value = r102_value * 2.0315
    r102_value = r102_value + tl.load($(vmm_25_ptr) + (((r102_row[:, None] % $(128)) * $(128)) + (r102_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r102_value = r102_value.to(tl.bfloat16)
    tl.store($(vaddmm_50_ptr) + ((r102_row[:, None] * $(128)) + r102_col[None, :]), r102_value)
    r103_row = tl.arange($(128))
    r103_col = tl.arange($(256))
    r103_lane = tl.arange($(128))
    r103_left = tl.load($(vaddmm_50_ptr) + (((r103_row[:, None] % $(128)) * $(128)) + (r103_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r103_right = tl.load($(vaddmm_49_ptr) + (((r103_lane[:, None] % $(128)) * $(256)) + (r103_col[None, :] % $(256))), dtype=tl.bfloat16)
    r103_value = tl.dot(r103_left, r103_right)
    r103_value = r103_value + tl.load($(vaddmm_49_ptr) + (((r103_row[:, None] % $(128)) * $(256)) + (r103_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r103_value = r103_value.to(tl.bfloat16)
    tl.store($(vaddmm_51_ptr) + ((r103_row[:, None] * $(256)) + r103_col[None, :]), r103_value)
    r104_row = tl.arange($(128))
    r104_col = tl.arange($(128))
    r104_lane = tl.arange($(256))
    r104_left = tl.load($(vaddmm_51_ptr) + (((r104_row[:, None] % $(128)) * $(256)) + (r104_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r104_right = tl.load($(vaddmm_51_ptr) + (((r104_col[None, :] % $(128)) * $(256)) + (r104_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r104_value = tl.dot(r104_left, r104_right)
    r104_value = r104_value.to(tl.bfloat16)
    tl.store($(vmm_26_ptr) + ((r104_row[:, None] * $(128)) + r104_col[None, :]), r104_value)
    r105_row = tl.arange($(128))
    r105_col = tl.arange($(128))
    r105_lane = tl.arange($(128))
    r105_left = tl.load($(vmm_26_ptr) + (((r105_row[:, None] % $(128)) * $(128)) + (r105_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r105_right = tl.load($(vmm_26_ptr) + (((r105_lane[:, None] % $(128)) * $(128)) + (r105_col[None, :] % $(128))), dtype=tl.bfloat16)
    r105_value = tl.dot(r105_left, r105_right)
    r105_value = r105_value * 2.0315
    r105_value = r105_value + tl.load($(vmm_26_ptr) + (((r105_row[:, None] % $(128)) * $(128)) + (r105_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r105_value = r105_value.to(tl.bfloat16)
    tl.store($(vaddmm_52_ptr) + ((r105_row[:, None] * $(128)) + r105_col[None, :]), r105_value)
    r106_row = tl.arange($(128))
    r106_col = tl.arange($(256))
    r106_lane = tl.arange($(128))
    r106_left = tl.load($(vaddmm_52_ptr) + (((r106_row[:, None] % $(128)) * $(128)) + (r106_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r106_right = tl.load($(vaddmm_51_ptr) + (((r106_lane[:, None] % $(128)) * $(256)) + (r106_col[None, :] % $(256))), dtype=tl.bfloat16)
    r106_value = tl.dot(r106_left, r106_right)
    r106_value = r106_value + tl.load($(vaddmm_51_ptr) + (((r106_row[:, None] % $(128)) * $(256)) + (r106_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r106_value = r106_value.to(tl.bfloat16)
    tl.store($(vaddmm_53_ptr) + ((r106_row[:, None] * $(256)) + r106_col[None, :]), r106_value)
    r107_row = tl.arange($(128))
    r107_col = tl.arange($(128))
    r107_lane = tl.arange($(256))
    r107_left = tl.load($(vaddmm_53_ptr) + (((r107_row[:, None] % $(128)) * $(256)) + (r107_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r107_right = tl.load($(vaddmm_53_ptr) + (((r107_col[None, :] % $(128)) * $(256)) + (r107_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r107_value = tl.dot(r107_left, r107_right)
    r107_value = r107_value.to(tl.bfloat16)
    tl.store($(vmm_27_ptr) + ((r107_row[:, None] * $(128)) + r107_col[None, :]), r107_value)
    r108_row = tl.arange($(128))
    r108_col = tl.arange($(128))
    r108_lane = tl.arange($(128))
    r108_left = tl.load($(vmm_27_ptr) + (((r108_row[:, None] % $(128)) * $(128)) + (r108_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r108_right = tl.load($(vmm_27_ptr) + (((r108_lane[:, None] % $(128)) * $(128)) + (r108_col[None, :] % $(128))), dtype=tl.bfloat16)
    r108_value = tl.dot(r108_left, r108_right)
    r108_value = r108_value * 2.0315
    r108_value = r108_value + tl.load($(vmm_27_ptr) + (((r108_row[:, None] % $(128)) * $(128)) + (r108_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r108_value = r108_value.to(tl.bfloat16)
    tl.store($(vaddmm_54_ptr) + ((r108_row[:, None] * $(128)) + r108_col[None, :]), r108_value)
    r109_row = tl.arange($(128))
    r109_col = tl.arange($(256))
    r109_lane = tl.arange($(128))
    r109_left = tl.load($(vaddmm_54_ptr) + (((r109_row[:, None] % $(128)) * $(128)) + (r109_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r109_right = tl.load($(vaddmm_53_ptr) + (((r109_lane[:, None] % $(128)) * $(256)) + (r109_col[None, :] % $(256))), dtype=tl.bfloat16)
    r109_value = tl.dot(r109_left, r109_right)
    r109_value = r109_value + tl.load($(vaddmm_53_ptr) + (((r109_row[:, None] % $(128)) * $(256)) + (r109_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r109_value = r109_value.to(tl.bfloat16)
    tl.store($(vaddmm_55_ptr) + ((r109_row[:, None] * $(256)) + r109_col[None, :]), r109_value)
    r110_row = tl.arange($(128))
    r110_col = tl.arange($(128))
    r110_lane = tl.arange($(256))
    r110_left = tl.load($(vaddmm_55_ptr) + (((r110_row[:, None] % $(128)) * $(256)) + (r110_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r110_right = tl.load($(vaddmm_55_ptr) + (((r110_col[None, :] % $(128)) * $(256)) + (r110_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r110_value = tl.dot(r110_left, r110_right)
    r110_value = r110_value.to(tl.bfloat16)
    tl.store($(vmm_28_ptr) + ((r110_row[:, None] * $(128)) + r110_col[None, :]), r110_value)
    r111_row = tl.arange($(128))
    r111_col = tl.arange($(128))
    r111_lane = tl.arange($(128))
    r111_left = tl.load($(vmm_28_ptr) + (((r111_row[:, None] % $(128)) * $(128)) + (r111_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r111_right = tl.load($(vmm_28_ptr) + (((r111_lane[:, None] % $(128)) * $(128)) + (r111_col[None, :] % $(128))), dtype=tl.bfloat16)
    r111_value = tl.dot(r111_left, r111_right)
    r111_value = r111_value * 0.5
    r111_value = r111_value + tl.load($(vmm_28_ptr) + (((r111_row[:, None] % $(128)) * $(128)) + (r111_col[None, :] % $(128))), dtype=tl.bfloat16) * -1.5
    r111_value = r111_value.to(tl.bfloat16)
    tl.store($(vaddmm_56_ptr) + ((r111_row[:, None] * $(128)) + r111_col[None, :]), r111_value)
    r112_row = tl.arange($(128))
    r112_col = tl.arange($(256))
    r112_lane = tl.arange($(128))
    r112_left = tl.load($(vaddmm_56_ptr) + (((r112_row[:, None] % $(128)) * $(128)) + (r112_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r112_right = tl.load($(vaddmm_55_ptr) + (((r112_lane[:, None] % $(128)) * $(256)) + (r112_col[None, :] % $(256))), dtype=tl.bfloat16)
    r112_value = tl.dot(r112_left, r112_right)
    r112_value = r112_value + tl.load($(vaddmm_55_ptr) + (((r112_row[:, None] % $(128)) * $(256)) + (r112_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
    r112_value = r112_value.to(tl.bfloat16)
    tl.store($(vaddmm_57_ptr) + ((r112_row[:, None] * $(256)) + r112_col[None, :]), r112_value)
    r113_row = tl.arange($(128))
    r113_col = tl.arange($(128))
    r113_lane = tl.arange($(256))
    r113_left = tl.load($(vaddmm_57_ptr) + (((r113_row[:, None] % $(128)) * $(256)) + (r113_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r113_right = tl.load($(vaddmm_57_ptr) + (((r113_col[None, :] % $(128)) * $(256)) + (r113_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r113_value = tl.dot(r113_left, r113_right)
    r113_value = r113_value.to(tl.bfloat16)
    tl.store($(vmm_29_ptr) + ((r113_row[:, None] * $(128)) + r113_col[None, :]), r113_value)
    r114_row = tl.arange($(128))
    r114_col = tl.arange($(128))
    r114_lane = tl.arange($(128))
    r114_left = tl.load($(vmm_29_ptr) + (((r114_row[:, None] % $(128)) * $(128)) + (r114_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r114_right = tl.load($(vmm_29_ptr) + (((r114_lane[:, None] % $(128)) * $(128)) + (r114_col[None, :] % $(128))), dtype=tl.bfloat16)
    r114_value = tl.dot(r114_left, r114_right)
    r114_value = r114_value * 0.5
    r114_value = r114_value + tl.load($(vmm_29_ptr) + (((r114_row[:, None] % $(128)) * $(128)) + (r114_col[None, :] % $(128))), dtype=tl.bfloat16) * -1.5
    r114_value = r114_value.to(tl.bfloat16)
    tl.store($(vaddmm_58_ptr) + ((r114_row[:, None] * $(128)) + r114_col[None, :]), r114_value)
    r115_row = tl.arange($(128))
    r115_col = tl.arange($(256))
    r115_lane = tl.arange($(128))
    r115_left = tl.load($(vaddmm_58_ptr) + (((r115_row[:, None] % $(128)) * $(128)) + (r115_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r115_right = tl.load($(vaddmm_57_ptr) + (((r115_lane[:, None] % $(128)) * $(256)) + (r115_col[None, :] % $(256))), dtype=tl.bfloat16)
    r115_value = tl.dot(r115_left, r115_right)
    r115_value = r115_value + tl.load($(vaddmm_57_ptr) + (((r115_row[:, None] % $(128)) * $(256)) + (r115_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
    r115_value = r115_value.to(tl.bfloat16)
    tl.store($(vaddmm_59_ptr) + ((r115_row[:, None] * $(256)) + r115_col[None, :]), r115_value)
    r116_z = tl.arange($(32768))
    r116_value = (tl.load($(vselect_scatter_5_ptr) + ((((r116_z / $(128)) % $(256)) * $(128)) + (r116_z % $(128)) + $(65536)), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r116_value = r116_value.to(tl.bfloat16)
    r116_value = tl.broadcast_to(r116_value, [$(32768)])
    tl.store($(vmul_2_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(131072))
    r117_value = tl.where((((r117_z / $(32768)) % $(4)) == $(2)), tl.load($(vmul_2_ptr) + ((((r117_z / $(128)) % $(256)) * $(128)) + (r117_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_5_ptr) + ((((r117_z / $(32768)) % $(4)) * $(32768)) + (((r117_z / $(128)) % $(256)) * $(128)) + (r117_z % $(128))), dtype=tl.bfloat16))
    r117_value = r117_value.to(tl.bfloat16)
    r117_value = tl.broadcast_to(r117_value, [$(131072)])
    tl.store($(vselect_scatter_7_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(32768))
    r118_value = (tl.load($(vselect_scatter_7_ptr) + ((((r118_z / $(128)) % $(256)) * $(128)) + (r118_z % $(128)) + $(65536)), dtype=tl.bfloat16) + (tl.load($(vaddmm_59_ptr) + (((r118_z % $(128)) * $(256)) + ((r118_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r118_value = r118_value.to(tl.bfloat16)
    r118_value = tl.broadcast_to(r118_value, [$(32768)])
    tl.store($(vadd_2_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(131072))
    r119_value = tl.where((((r119_z / $(32768)) % $(4)) == $(2)), tl.load($(vadd_2_ptr) + ((((r119_z / $(128)) % $(256)) * $(128)) + (r119_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_7_ptr) + ((((r119_z / $(32768)) % $(4)) * $(32768)) + (((r119_z / $(128)) % $(256)) * $(128)) + (r119_z % $(128))), dtype=tl.bfloat16))
    r119_value = r119_value.to(tl.bfloat16)
    r119_value = tl.broadcast_to(r119_value, [$(131072)])
    tl.store($(vselect_scatter_8_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(32768))
    r120_value = (tl.load($(vselect_scatter_6_ptr) + ((((r120_z / $(128)) % $(256)) * $(128)) + (r120_z % $(128)) + $(98304)), dtype=tl.bfloat16) + (tl.load($(input0_ptr) + ((((r120_z / $(128)) % $(256)) * $(128)) + (r120_z % $(128)) + $(98304)), dtype=tl.bfloat16) - tl.load($(vselect_scatter_6_ptr) + ((((r120_z / $(128)) % $(256)) * $(128)) + (r120_z % $(128)) + $(98304)), dtype=tl.bfloat16)) * 0.050000000000000044)
    r120_value = r120_value.to(tl.bfloat16)
    r120_value = tl.broadcast_to(r120_value, [$(32768)])
    tl.store($(vlerp_6_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(131072))
    r121_value = tl.where((((r121_z / $(32768)) % $(4)) == $(3)), tl.load($(vlerp_6_ptr) + ((((r121_z / $(128)) % $(256)) * $(128)) + (r121_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_6_ptr) + ((((r121_z / $(32768)) % $(4)) * $(32768)) + (((r121_z / $(128)) % $(256)) * $(128)) + (r121_z % $(128))), dtype=tl.bfloat16))
    r121_value = r121_value.to(tl.bfloat16)
    r121_value = tl.broadcast_to(r121_value, [$(131072)])
    tl.store($(vselect_scatter_9_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(32768))
    r122_value = (tl.load($(input0_ptr) + ((((r122_z / $(128)) % $(256)) * $(128)) + (r122_z % $(128)) + $(98304)), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_9_ptr) + ((((r122_z / $(128)) % $(256)) * $(128)) + (r122_z % $(128)) + $(98304)), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r122_z / $(128)) % $(256)) * $(128)) + (r122_z % $(128)) + $(98304)), dtype=tl.bfloat16)) * 0.95)
    r122_value = r122_value.to(tl.bfloat16)
    r122_value = tl.broadcast_to(r122_value, [$(32768)])
    tl.store($(vlerp_7_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(1))
    r123_lane = tl.arange($(32768))
    r123_value = tl.sqrt(tl.sum(tl.load($(vlerp_7_ptr) + (((r123_lane[None, :] % $(256)) * $(128)) + ((r123_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16) * tl.load($(vlerp_7_ptr) + (((r123_lane[None, :] % $(256)) * $(128)) + ((r123_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16), axis=1))
    r123_value = r123_value.to(tl.bfloat16)
    r123_value = tl.broadcast_to(r123_value, [$(1)])
    tl.store($(vlinalg_vector_norm_3_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(1))
    r124_value = tl.maximum(tl.load($(vlinalg_vector_norm_3_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r124_value = r124_value.to(tl.bfloat16)
    r124_value = tl.broadcast_to(r124_value, [$(1)])
    tl.store($(vclamp_3_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(32768))
    r125_value = (tl.load($(vlerp_7_ptr) + (((r125_z % $(256)) * $(128)) + ((r125_z / $(256)) % $(128))), dtype=tl.bfloat16) / tl.load($(vclamp_3_ptr) + $(0), dtype=tl.bfloat16))
    r125_value = r125_value.to(tl.bfloat16)
    r125_value = tl.broadcast_to(r125_value, [$(32768)])
    tl.store($(vdiv_3_ptr) + r125_z, r125_value)
    r126_row = tl.arange($(128))
    r126_col = tl.arange($(128))
    r126_lane = tl.arange($(256))
    r126_left = tl.load($(vdiv_3_ptr) + (((r126_row[:, None] % $(128)) * $(256)) + (r126_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r126_right = tl.load($(vdiv_3_ptr) + (((r126_col[None, :] % $(128)) * $(256)) + (r126_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r126_value = tl.dot(r126_left, r126_right)
    r126_value = r126_value.to(tl.bfloat16)
    tl.store($(vmm_30_ptr) + ((r126_row[:, None] * $(128)) + r126_col[None, :]), r126_value)
    r127_row = tl.arange($(128))
    r127_col = tl.arange($(128))
    r127_lane = tl.arange($(128))
    r127_left = tl.load($(vmm_30_ptr) + (((r127_row[:, None] % $(128)) * $(128)) + (r127_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r127_right = tl.load($(vmm_30_ptr) + (((r127_lane[:, None] % $(128)) * $(128)) + (r127_col[None, :] % $(128))), dtype=tl.bfloat16)
    r127_value = tl.dot(r127_left, r127_right)
    r127_value = r127_value * 2.0315
    r127_value = r127_value + tl.load($(vmm_30_ptr) + (((r127_row[:, None] % $(128)) * $(128)) + (r127_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r127_value = r127_value.to(tl.bfloat16)
    tl.store($(vaddmm_60_ptr) + ((r127_row[:, None] * $(128)) + r127_col[None, :]), r127_value)
    r128_row = tl.arange($(128))
    r128_col = tl.arange($(256))
    r128_lane = tl.arange($(128))
    r128_left = tl.load($(vaddmm_60_ptr) + (((r128_row[:, None] % $(128)) * $(128)) + (r128_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r128_right = tl.load($(vdiv_3_ptr) + (((r128_lane[:, None] % $(128)) * $(256)) + (r128_col[None, :] % $(256))), dtype=tl.bfloat16)
    r128_value = tl.dot(r128_left, r128_right)
    r128_value = r128_value + tl.load($(vdiv_3_ptr) + (((r128_row[:, None] % $(128)) * $(256)) + (r128_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r128_value = r128_value.to(tl.bfloat16)
    tl.store($(vaddmm_61_ptr) + ((r128_row[:, None] * $(256)) + r128_col[None, :]), r128_value)
    r129_row = tl.arange($(128))
    r129_col = tl.arange($(128))
    r129_lane = tl.arange($(256))
    r129_left = tl.load($(vaddmm_61_ptr) + (((r129_row[:, None] % $(128)) * $(256)) + (r129_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r129_right = tl.load($(vaddmm_61_ptr) + (((r129_col[None, :] % $(128)) * $(256)) + (r129_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r129_value = tl.dot(r129_left, r129_right)
    r129_value = r129_value.to(tl.bfloat16)
    tl.store($(vmm_31_ptr) + ((r129_row[:, None] * $(128)) + r129_col[None, :]), r129_value)
    r130_row = tl.arange($(128))
    r130_col = tl.arange($(128))
    r130_lane = tl.arange($(128))
    r130_left = tl.load($(vmm_31_ptr) + (((r130_row[:, None] % $(128)) * $(128)) + (r130_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r130_right = tl.load($(vmm_31_ptr) + (((r130_lane[:, None] % $(128)) * $(128)) + (r130_col[None, :] % $(128))), dtype=tl.bfloat16)
    r130_value = tl.dot(r130_left, r130_right)
    r130_value = r130_value * 2.0315
    r130_value = r130_value + tl.load($(vmm_31_ptr) + (((r130_row[:, None] % $(128)) * $(128)) + (r130_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r130_value = r130_value.to(tl.bfloat16)
    tl.store($(vaddmm_62_ptr) + ((r130_row[:, None] * $(128)) + r130_col[None, :]), r130_value)
    r131_row = tl.arange($(128))
    r131_col = tl.arange($(256))
    r131_lane = tl.arange($(128))
    r131_left = tl.load($(vaddmm_62_ptr) + (((r131_row[:, None] % $(128)) * $(128)) + (r131_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r131_right = tl.load($(vaddmm_61_ptr) + (((r131_lane[:, None] % $(128)) * $(256)) + (r131_col[None, :] % $(256))), dtype=tl.bfloat16)
    r131_value = tl.dot(r131_left, r131_right)
    r131_value = r131_value + tl.load($(vaddmm_61_ptr) + (((r131_row[:, None] % $(128)) * $(256)) + (r131_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r131_value = r131_value.to(tl.bfloat16)
    tl.store($(vaddmm_63_ptr) + ((r131_row[:, None] * $(256)) + r131_col[None, :]), r131_value)
    r132_row = tl.arange($(128))
    r132_col = tl.arange($(128))
    r132_lane = tl.arange($(256))
    r132_left = tl.load($(vaddmm_63_ptr) + (((r132_row[:, None] % $(128)) * $(256)) + (r132_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r132_right = tl.load($(vaddmm_63_ptr) + (((r132_col[None, :] % $(128)) * $(256)) + (r132_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r132_value = tl.dot(r132_left, r132_right)
    r132_value = r132_value.to(tl.bfloat16)
    tl.store($(vmm_32_ptr) + ((r132_row[:, None] * $(128)) + r132_col[None, :]), r132_value)
    r133_row = tl.arange($(128))
    r133_col = tl.arange($(128))
    r133_lane = tl.arange($(128))
    r133_left = tl.load($(vmm_32_ptr) + (((r133_row[:, None] % $(128)) * $(128)) + (r133_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r133_right = tl.load($(vmm_32_ptr) + (((r133_lane[:, None] % $(128)) * $(128)) + (r133_col[None, :] % $(128))), dtype=tl.bfloat16)
    r133_value = tl.dot(r133_left, r133_right)
    r133_value = r133_value * 2.0315
    r133_value = r133_value + tl.load($(vmm_32_ptr) + (((r133_row[:, None] % $(128)) * $(128)) + (r133_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r133_value = r133_value.to(tl.bfloat16)
    tl.store($(vaddmm_64_ptr) + ((r133_row[:, None] * $(128)) + r133_col[None, :]), r133_value)
    r134_row = tl.arange($(128))
    r134_col = tl.arange($(256))
    r134_lane = tl.arange($(128))
    r134_left = tl.load($(vaddmm_64_ptr) + (((r134_row[:, None] % $(128)) * $(128)) + (r134_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r134_right = tl.load($(vaddmm_63_ptr) + (((r134_lane[:, None] % $(128)) * $(256)) + (r134_col[None, :] % $(256))), dtype=tl.bfloat16)
    r134_value = tl.dot(r134_left, r134_right)
    r134_value = r134_value + tl.load($(vaddmm_63_ptr) + (((r134_row[:, None] % $(128)) * $(256)) + (r134_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r134_value = r134_value.to(tl.bfloat16)
    tl.store($(vaddmm_65_ptr) + ((r134_row[:, None] * $(256)) + r134_col[None, :]), r134_value)
    r135_row = tl.arange($(128))
    r135_col = tl.arange($(128))
    r135_lane = tl.arange($(256))
    r135_left = tl.load($(vaddmm_65_ptr) + (((r135_row[:, None] % $(128)) * $(256)) + (r135_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r135_right = tl.load($(vaddmm_65_ptr) + (((r135_col[None, :] % $(128)) * $(256)) + (r135_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r135_value = tl.dot(r135_left, r135_right)
    r135_value = r135_value.to(tl.bfloat16)
    tl.store($(vmm_33_ptr) + ((r135_row[:, None] * $(128)) + r135_col[None, :]), r135_value)
    r136_row = tl.arange($(128))
    r136_col = tl.arange($(128))
    r136_lane = tl.arange($(128))
    r136_left = tl.load($(vmm_33_ptr) + (((r136_row[:, None] % $(128)) * $(128)) + (r136_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r136_right = tl.load($(vmm_33_ptr) + (((r136_lane[:, None] % $(128)) * $(128)) + (r136_col[None, :] % $(128))), dtype=tl.bfloat16)
    r136_value = tl.dot(r136_left, r136_right)
    r136_value = r136_value * 2.0315
    r136_value = r136_value + tl.load($(vmm_33_ptr) + (((r136_row[:, None] % $(128)) * $(128)) + (r136_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r136_value = r136_value.to(tl.bfloat16)
    tl.store($(vaddmm_66_ptr) + ((r136_row[:, None] * $(128)) + r136_col[None, :]), r136_value)
    r137_row = tl.arange($(128))
    r137_col = tl.arange($(256))
    r137_lane = tl.arange($(128))
    r137_left = tl.load($(vaddmm_66_ptr) + (((r137_row[:, None] % $(128)) * $(128)) + (r137_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r137_right = tl.load($(vaddmm_65_ptr) + (((r137_lane[:, None] % $(128)) * $(256)) + (r137_col[None, :] % $(256))), dtype=tl.bfloat16)
    r137_value = tl.dot(r137_left, r137_right)
    r137_value = r137_value + tl.load($(vaddmm_65_ptr) + (((r137_row[:, None] % $(128)) * $(256)) + (r137_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r137_value = r137_value.to(tl.bfloat16)
    tl.store($(vaddmm_67_ptr) + ((r137_row[:, None] * $(256)) + r137_col[None, :]), r137_value)
    r138_row = tl.arange($(128))
    r138_col = tl.arange($(128))
    r138_lane = tl.arange($(256))
    r138_left = tl.load($(vaddmm_67_ptr) + (((r138_row[:, None] % $(128)) * $(256)) + (r138_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r138_right = tl.load($(vaddmm_67_ptr) + (((r138_col[None, :] % $(128)) * $(256)) + (r138_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r138_value = tl.dot(r138_left, r138_right)
    r138_value = r138_value.to(tl.bfloat16)
    tl.store($(vmm_34_ptr) + ((r138_row[:, None] * $(128)) + r138_col[None, :]), r138_value)
    r139_row = tl.arange($(128))
    r139_col = tl.arange($(128))
    r139_lane = tl.arange($(128))
    r139_left = tl.load($(vmm_34_ptr) + (((r139_row[:, None] % $(128)) * $(128)) + (r139_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r139_right = tl.load($(vmm_34_ptr) + (((r139_lane[:, None] % $(128)) * $(128)) + (r139_col[None, :] % $(128))), dtype=tl.bfloat16)
    r139_value = tl.dot(r139_left, r139_right)
    r139_value = r139_value * 2.0315
    r139_value = r139_value + tl.load($(vmm_34_ptr) + (((r139_row[:, None] % $(128)) * $(128)) + (r139_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r139_value = r139_value.to(tl.bfloat16)
    tl.store($(vaddmm_68_ptr) + ((r139_row[:, None] * $(128)) + r139_col[None, :]), r139_value)
    r140_row = tl.arange($(128))
    r140_col = tl.arange($(256))
    r140_lane = tl.arange($(128))
    r140_left = tl.load($(vaddmm_68_ptr) + (((r140_row[:, None] % $(128)) * $(128)) + (r140_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r140_right = tl.load($(vaddmm_67_ptr) + (((r140_lane[:, None] % $(128)) * $(256)) + (r140_col[None, :] % $(256))), dtype=tl.bfloat16)
    r140_value = tl.dot(r140_left, r140_right)
    r140_value = r140_value + tl.load($(vaddmm_67_ptr) + (((r140_row[:, None] % $(128)) * $(256)) + (r140_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r140_value = r140_value.to(tl.bfloat16)
    tl.store($(vaddmm_69_ptr) + ((r140_row[:, None] * $(256)) + r140_col[None, :]), r140_value)
    r141_row = tl.arange($(128))
    r141_col = tl.arange($(128))
    r141_lane = tl.arange($(256))
    r141_left = tl.load($(vaddmm_69_ptr) + (((r141_row[:, None] % $(128)) * $(256)) + (r141_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r141_right = tl.load($(vaddmm_69_ptr) + (((r141_col[None, :] % $(128)) * $(256)) + (r141_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r141_value = tl.dot(r141_left, r141_right)
    r141_value = r141_value.to(tl.bfloat16)
    tl.store($(vmm_35_ptr) + ((r141_row[:, None] * $(128)) + r141_col[None, :]), r141_value)
    r142_row = tl.arange($(128))
    r142_col = tl.arange($(128))
    r142_lane = tl.arange($(128))
    r142_left = tl.load($(vmm_35_ptr) + (((r142_row[:, None] % $(128)) * $(128)) + (r142_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r142_right = tl.load($(vmm_35_ptr) + (((r142_lane[:, None] % $(128)) * $(128)) + (r142_col[None, :] % $(128))), dtype=tl.bfloat16)
    r142_value = tl.dot(r142_left, r142_right)
    r142_value = r142_value * 2.0315
    r142_value = r142_value + tl.load($(vmm_35_ptr) + (((r142_row[:, None] % $(128)) * $(128)) + (r142_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r142_value = r142_value.to(tl.bfloat16)
    tl.store($(vaddmm_70_ptr) + ((r142_row[:, None] * $(128)) + r142_col[None, :]), r142_value)
    r143_row = tl.arange($(128))
    r143_col = tl.arange($(256))
    r143_lane = tl.arange($(128))
    r143_left = tl.load($(vaddmm_70_ptr) + (((r143_row[:, None] % $(128)) * $(128)) + (r143_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r143_right = tl.load($(vaddmm_69_ptr) + (((r143_lane[:, None] % $(128)) * $(256)) + (r143_col[None, :] % $(256))), dtype=tl.bfloat16)
    r143_value = tl.dot(r143_left, r143_right)
    r143_value = r143_value + tl.load($(vaddmm_69_ptr) + (((r143_row[:, None] % $(128)) * $(256)) + (r143_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r143_value = r143_value.to(tl.bfloat16)
    tl.store($(vaddmm_71_ptr) + ((r143_row[:, None] * $(256)) + r143_col[None, :]), r143_value)
    r144_row = tl.arange($(128))
    r144_col = tl.arange($(128))
    r144_lane = tl.arange($(256))
    r144_left = tl.load($(vaddmm_71_ptr) + (((r144_row[:, None] % $(128)) * $(256)) + (r144_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r144_right = tl.load($(vaddmm_71_ptr) + (((r144_col[None, :] % $(128)) * $(256)) + (r144_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r144_value = tl.dot(r144_left, r144_right)
    r144_value = r144_value.to(tl.bfloat16)
    tl.store($(vmm_36_ptr) + ((r144_row[:, None] * $(128)) + r144_col[None, :]), r144_value)
    r145_row = tl.arange($(128))
    r145_col = tl.arange($(128))
    r145_lane = tl.arange($(128))
    r145_left = tl.load($(vmm_36_ptr) + (((r145_row[:, None] % $(128)) * $(128)) + (r145_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r145_right = tl.load($(vmm_36_ptr) + (((r145_lane[:, None] % $(128)) * $(128)) + (r145_col[None, :] % $(128))), dtype=tl.bfloat16)
    r145_value = tl.dot(r145_left, r145_right)
    r145_value = r145_value * 2.0315
    r145_value = r145_value + tl.load($(vmm_36_ptr) + (((r145_row[:, None] % $(128)) * $(128)) + (r145_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r145_value = r145_value.to(tl.bfloat16)
    tl.store($(vaddmm_72_ptr) + ((r145_row[:, None] * $(128)) + r145_col[None, :]), r145_value)
    r146_row = tl.arange($(128))
    r146_col = tl.arange($(256))
    r146_lane = tl.arange($(128))
    r146_left = tl.load($(vaddmm_72_ptr) + (((r146_row[:, None] % $(128)) * $(128)) + (r146_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r146_right = tl.load($(vaddmm_71_ptr) + (((r146_lane[:, None] % $(128)) * $(256)) + (r146_col[None, :] % $(256))), dtype=tl.bfloat16)
    r146_value = tl.dot(r146_left, r146_right)
    r146_value = r146_value + tl.load($(vaddmm_71_ptr) + (((r146_row[:, None] % $(128)) * $(256)) + (r146_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r146_value = r146_value.to(tl.bfloat16)
    tl.store($(vaddmm_73_ptr) + ((r146_row[:, None] * $(256)) + r146_col[None, :]), r146_value)
    r147_row = tl.arange($(128))
    r147_col = tl.arange($(128))
    r147_lane = tl.arange($(256))
    r147_left = tl.load($(vaddmm_73_ptr) + (((r147_row[:, None] % $(128)) * $(256)) + (r147_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r147_right = tl.load($(vaddmm_73_ptr) + (((r147_col[None, :] % $(128)) * $(256)) + (r147_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r147_value = tl.dot(r147_left, r147_right)
    r147_value = r147_value.to(tl.bfloat16)
    tl.store($(vmm_37_ptr) + ((r147_row[:, None] * $(128)) + r147_col[None, :]), r147_value)
    r148_row = tl.arange($(128))
    r148_col = tl.arange($(128))
    r148_lane = tl.arange($(128))
    r148_left = tl.load($(vmm_37_ptr) + (((r148_row[:, None] % $(128)) * $(128)) + (r148_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r148_right = tl.load($(vmm_37_ptr) + (((r148_lane[:, None] % $(128)) * $(128)) + (r148_col[None, :] % $(128))), dtype=tl.bfloat16)
    r148_value = tl.dot(r148_left, r148_right)
    r148_value = r148_value * 2.0315
    r148_value = r148_value + tl.load($(vmm_37_ptr) + (((r148_row[:, None] % $(128)) * $(128)) + (r148_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r148_value = r148_value.to(tl.bfloat16)
    tl.store($(vaddmm_74_ptr) + ((r148_row[:, None] * $(128)) + r148_col[None, :]), r148_value)
    r149_row = tl.arange($(128))
    r149_col = tl.arange($(256))
    r149_lane = tl.arange($(128))
    r149_left = tl.load($(vaddmm_74_ptr) + (((r149_row[:, None] % $(128)) * $(128)) + (r149_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r149_right = tl.load($(vaddmm_73_ptr) + (((r149_lane[:, None] % $(128)) * $(256)) + (r149_col[None, :] % $(256))), dtype=tl.bfloat16)
    r149_value = tl.dot(r149_left, r149_right)
    r149_value = r149_value + tl.load($(vaddmm_73_ptr) + (((r149_row[:, None] % $(128)) * $(256)) + (r149_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r149_value = r149_value.to(tl.bfloat16)
    tl.store($(vaddmm_75_ptr) + ((r149_row[:, None] * $(256)) + r149_col[None, :]), r149_value)
    r150_row = tl.arange($(128))
    r150_col = tl.arange($(128))
    r150_lane = tl.arange($(256))
    r150_left = tl.load($(vaddmm_75_ptr) + (((r150_row[:, None] % $(128)) * $(256)) + (r150_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r150_right = tl.load($(vaddmm_75_ptr) + (((r150_col[None, :] % $(128)) * $(256)) + (r150_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r150_value = tl.dot(r150_left, r150_right)
    r150_value = r150_value.to(tl.bfloat16)
    tl.store($(vmm_38_ptr) + ((r150_row[:, None] * $(128)) + r150_col[None, :]), r150_value)
    r151_row = tl.arange($(128))
    r151_col = tl.arange($(128))
    r151_lane = tl.arange($(128))
    r151_left = tl.load($(vmm_38_ptr) + (((r151_row[:, None] % $(128)) * $(128)) + (r151_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r151_right = tl.load($(vmm_38_ptr) + (((r151_lane[:, None] % $(128)) * $(128)) + (r151_col[None, :] % $(128))), dtype=tl.bfloat16)
    r151_value = tl.dot(r151_left, r151_right)
    r151_value = r151_value * 0.5
    r151_value = r151_value + tl.load($(vmm_38_ptr) + (((r151_row[:, None] % $(128)) * $(128)) + (r151_col[None, :] % $(128))), dtype=tl.bfloat16) * -1.5
    r151_value = r151_value.to(tl.bfloat16)
    tl.store($(vaddmm_76_ptr) + ((r151_row[:, None] * $(128)) + r151_col[None, :]), r151_value)
    r152_row = tl.arange($(128))
    r152_col = tl.arange($(256))
    r152_lane = tl.arange($(128))
    r152_left = tl.load($(vaddmm_76_ptr) + (((r152_row[:, None] % $(128)) * $(128)) + (r152_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r152_right = tl.load($(vaddmm_75_ptr) + (((r152_lane[:, None] % $(128)) * $(256)) + (r152_col[None, :] % $(256))), dtype=tl.bfloat16)
    r152_value = tl.dot(r152_left, r152_right)
    r152_value = r152_value + tl.load($(vaddmm_75_ptr) + (((r152_row[:, None] % $(128)) * $(256)) + (r152_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
    r152_value = r152_value.to(tl.bfloat16)
    tl.store($(vaddmm_77_ptr) + ((r152_row[:, None] * $(256)) + r152_col[None, :]), r152_value)
    r153_row = tl.arange($(128))
    r153_col = tl.arange($(128))
    r153_lane = tl.arange($(256))
    r153_left = tl.load($(vaddmm_77_ptr) + (((r153_row[:, None] % $(128)) * $(256)) + (r153_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r153_right = tl.load($(vaddmm_77_ptr) + (((r153_col[None, :] % $(128)) * $(256)) + (r153_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r153_value = tl.dot(r153_left, r153_right)
    r153_value = r153_value.to(tl.bfloat16)
    tl.store($(vmm_39_ptr) + ((r153_row[:, None] * $(128)) + r153_col[None, :]), r153_value)
    r154_row = tl.arange($(128))
    r154_col = tl.arange($(128))
    r154_lane = tl.arange($(128))
    r154_left = tl.load($(vmm_39_ptr) + (((r154_row[:, None] % $(128)) * $(128)) + (r154_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r154_right = tl.load($(vmm_39_ptr) + (((r154_lane[:, None] % $(128)) * $(128)) + (r154_col[None, :] % $(128))), dtype=tl.bfloat16)
    r154_value = tl.dot(r154_left, r154_right)
    r154_value = r154_value * 0.5
    r154_value = r154_value + tl.load($(vmm_39_ptr) + (((r154_row[:, None] % $(128)) * $(128)) + (r154_col[None, :] % $(128))), dtype=tl.bfloat16) * -1.5
    r154_value = r154_value.to(tl.bfloat16)
    tl.store($(vaddmm_78_ptr) + ((r154_row[:, None] * $(128)) + r154_col[None, :]), r154_value)
    r155_row = tl.arange($(128))
    r155_col = tl.arange($(256))
    r155_lane = tl.arange($(128))
    r155_left = tl.load($(vaddmm_78_ptr) + (((r155_row[:, None] % $(128)) * $(128)) + (r155_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r155_right = tl.load($(vaddmm_77_ptr) + (((r155_lane[:, None] % $(128)) * $(256)) + (r155_col[None, :] % $(256))), dtype=tl.bfloat16)
    r155_value = tl.dot(r155_left, r155_right)
    r155_value = r155_value + tl.load($(vaddmm_77_ptr) + (((r155_row[:, None] % $(128)) * $(256)) + (r155_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
    r155_value = r155_value.to(tl.bfloat16)
    tl.store($(vaddmm_79_ptr) + ((r155_row[:, None] * $(256)) + r155_col[None, :]), r155_value)
    r156_z = tl.arange($(32768))
    r156_value = (tl.load($(vselect_scatter_8_ptr) + ((((r156_z / $(128)) % $(256)) * $(128)) + (r156_z % $(128)) + $(98304)), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r156_value = r156_value.to(tl.bfloat16)
    r156_value = tl.broadcast_to(r156_value, [$(32768)])
    tl.store($(vmul_3_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(131072))
    r157_value = tl.where((((r157_z / $(32768)) % $(4)) == $(3)), tl.load($(vmul_3_ptr) + ((((r157_z / $(128)) % $(256)) * $(128)) + (r157_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_8_ptr) + ((((r157_z / $(32768)) % $(4)) * $(32768)) + (((r157_z / $(128)) % $(256)) * $(128)) + (r157_z % $(128))), dtype=tl.bfloat16))
    r157_value = r157_value.to(tl.bfloat16)
    r157_value = tl.broadcast_to(r157_value, [$(131072)])
    tl.store($(vselect_scatter_10_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(32768))
    r158_value = (tl.load($(vselect_scatter_10_ptr) + ((((r158_z / $(128)) % $(256)) * $(128)) + (r158_z % $(128)) + $(98304)), dtype=tl.bfloat16) + (tl.load($(vaddmm_79_ptr) + (((r158_z % $(128)) * $(256)) + ((r158_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r158_value = r158_value.to(tl.bfloat16)
    r158_value = tl.broadcast_to(r158_value, [$(32768)])
    tl.store($(vadd_3_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(131072))
    r159_value = tl.where((((r159_z / $(32768)) % $(4)) == $(3)), tl.load($(vadd_3_ptr) + ((((r159_z / $(128)) % $(256)) * $(128)) + (r159_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_10_ptr) + ((((r159_z / $(32768)) % $(4)) * $(32768)) + (((r159_z / $(128)) % $(256)) * $(128)) + (r159_z % $(128))), dtype=tl.bfloat16))
    r159_value = r159_value.to(tl.bfloat16)
    r159_value = tl.broadcast_to(r159_value, [$(131072)])
    tl.store($(vselect_scatter_11_ptr) + r159_z, r159_value)
    z = tl.arange($(131072))
    value = tl.load($(vselect_scatter_11_ptr) + ((((z / $(32768)) % $(4)) * $(32768)) + (((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vlerp_ptr : RegionName := "vlerp"
  let vlerp_1_ptr : RegionName := "vlerp_1"
  let vlinalg_vector_norm_ptr : RegionName := "vlinalg_vector_norm"
  let vclamp_ptr : RegionName := "vclamp"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let candidate_vlerp_ptr : RegionName := "candidate_vlerp"
  let candidate_vlerp_1_ptr : RegionName := "candidate_vlerp_1"
  let candidate_vlinalg_vector_norm_ptr : RegionName := "candidate_vlinalg_vector_norm"
  let candidate_vclamp_ptr : RegionName := "candidate_vclamp"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vbaddbmm_ptr : RegionName := "candidate_vbaddbmm"
  let candidate_vbaddbmm_1_ptr : RegionName := "candidate_vbaddbmm_1"
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vbaddbmm_2_ptr : RegionName := "candidate_vbaddbmm_2"
  let candidate_vbaddbmm_3_ptr : RegionName := "candidate_vbaddbmm_3"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vbaddbmm_4_ptr : RegionName := "candidate_vbaddbmm_4"
  let candidate_vbaddbmm_5_ptr : RegionName := "candidate_vbaddbmm_5"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vbaddbmm_6_ptr : RegionName := "candidate_vbaddbmm_6"
  let candidate_vbaddbmm_7_ptr : RegionName := "candidate_vbaddbmm_7"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vbaddbmm_8_ptr : RegionName := "candidate_vbaddbmm_8"
  let candidate_vbaddbmm_9_ptr : RegionName := "candidate_vbaddbmm_9"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vbaddbmm_10_ptr : RegionName := "candidate_vbaddbmm_10"
  let candidate_vbaddbmm_11_ptr : RegionName := "candidate_vbaddbmm_11"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vbaddbmm_12_ptr : RegionName := "candidate_vbaddbmm_12"
  let candidate_vbaddbmm_13_ptr : RegionName := "candidate_vbaddbmm_13"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vbaddbmm_14_ptr : RegionName := "candidate_vbaddbmm_14"
  let candidate_vbaddbmm_15_ptr : RegionName := "candidate_vbaddbmm_15"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vbaddbmm_16_ptr : RegionName := "candidate_vbaddbmm_16"
  let candidate_vbaddbmm_17_ptr : RegionName := "candidate_vbaddbmm_17"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vbaddbmm_18_ptr : RegionName := "candidate_vbaddbmm_18"
  let candidate_vbaddbmm_19_ptr : RegionName := "candidate_vbaddbmm_19"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  triton {
    r0_z = tl.arange($(131072))
    r0_value = (tl.cast(0.0, tl.bfloat16) + (tl.load($(input0_ptr) + ((((r0_z / $(32768)) % $(4)) * $(32768)) + (((r0_z / $(128)) % $(256)) * $(128)) + (r0_z % $(128))), dtype=tl.bfloat16) - tl.cast(0.0, tl.bfloat16)) * 0.050000000000000044)
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(131072)])
    tl.store($(candidate_vlerp_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(131072))
    r1_value = (tl.load($(input0_ptr) + ((((r1_z / $(32768)) % $(4)) * $(32768)) + (((r1_z / $(128)) % $(256)) * $(128)) + (r1_z % $(128))), dtype=tl.bfloat16) + (tl.load($(candidate_vlerp_ptr) + ((((r1_z / $(32768)) % $(4)) * $(32768)) + (((r1_z / $(128)) % $(256)) * $(128)) + (r1_z % $(128))), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r1_z / $(32768)) % $(4)) * $(32768)) + (((r1_z / $(128)) % $(256)) * $(128)) + (r1_z % $(128))), dtype=tl.bfloat16)) * 0.95)
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(131072)])
    tl.store($(candidate_vlerp_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(4))
    r2_lane = tl.arange($(32768))
    r2_value = tl.sqrt(tl.sum(tl.load($(candidate_vlerp_1_ptr) + ((r2_z[:, None] * $(32768)) + ((r2_lane[None, :] % $(256)) * $(128)) + ((r2_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16) * tl.load($(candidate_vlerp_1_ptr) + ((r2_z[:, None] * $(32768)) + ((r2_lane[None, :] % $(256)) * $(128)) + ((r2_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16), axis=1))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(4)])
    tl.store($(candidate_vlinalg_vector_norm_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(4))
    r3_value = tl.maximum(tl.load($(candidate_vlinalg_vector_norm_ptr) + r3_z, dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(4)])
    tl.store($(candidate_vclamp_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(131072))
    r4_value = (tl.load($(candidate_vlerp_1_ptr) + ((((r4_z / $(32768)) % $(4)) * $(32768)) + ((r4_z % $(256)) * $(128)) + ((r4_z / $(256)) % $(128))), dtype=tl.bfloat16) / tl.load($(candidate_vclamp_ptr) + ((r4_z / $(32768)) % $(4)), dtype=tl.bfloat16))
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(131072)])
    tl.store($(candidate_vdiv_ptr) + r4_z, r4_value)
    r5_row = tl.arange($(128))
    r5_col = tl.arange($(128))
    r5_lane = tl.arange($(256))
    for r5_group in range($(4)) {
      r5_left = tl.load($(candidate_vdiv_ptr) + (((((((r5_group % $(4)) * $(32768)) + (r5_row[:, None] * $(256)) + r5_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r5_group % $(4)) * $(32768)) + (r5_row[:, None] * $(256)) + r5_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r5_group % $(4)) * $(32768)) + (r5_row[:, None] * $(256)) + r5_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r5_right = tl.load($(candidate_vdiv_ptr) + (((((((r5_group % $(4)) * $(32768)) + (r5_lane[:, None] * $(128)) + r5_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r5_group % $(4)) * $(32768)) + (r5_lane[:, None] * $(128)) + r5_col[None, :]) % $(128)) * $(256)) + (((((r5_group % $(4)) * $(32768)) + (r5_lane[:, None] * $(128)) + r5_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r5_value = tl.dot(r5_left, r5_right)
      r5_value = r5_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r5_group * $(16384)) + (r5_row[:, None] * $(128)) + r5_col[None, :]), r5_value)
    }
    r6_row = tl.arange($(128))
    r6_col = tl.arange($(128))
    r6_lane = tl.arange($(128))
    for r6_group in range($(4)) {
      r6_left = tl.load($(candidate_vbmm_ptr) + (((((((r6_group % $(4)) * $(16384)) + (r6_row[:, None] * $(128)) + r6_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r6_group % $(4)) * $(16384)) + (r6_row[:, None] * $(128)) + r6_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r6_group % $(4)) * $(16384)) + (r6_row[:, None] * $(128)) + r6_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r6_right = tl.load($(candidate_vbmm_ptr) + (((((((r6_group % $(4)) * $(16384)) + (r6_lane[:, None] * $(128)) + r6_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r6_group % $(4)) * $(16384)) + (r6_lane[:, None] * $(128)) + r6_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r6_group % $(4)) * $(16384)) + (r6_lane[:, None] * $(128)) + r6_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r6_value = tl.dot(r6_left, r6_right)
      r6_value = r6_value * 2.0315
      r6_value = r6_value + tl.load($(candidate_vbmm_ptr) + (((((((r6_group % $(4)) * $(16384)) + (r6_row[:, None] * $(128)) + r6_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r6_group % $(4)) * $(16384)) + (r6_row[:, None] * $(128)) + r6_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r6_group % $(4)) * $(16384)) + (r6_row[:, None] * $(128)) + r6_col[None, :]) % $(128))), dtype=tl.bfloat16) * -4.775
      r6_value = r6_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_ptr) + ((r6_group * $(16384)) + (r6_row[:, None] * $(128)) + r6_col[None, :]), r6_value)
    }
    r7_row = tl.arange($(128))
    r7_col = tl.arange($(256))
    r7_lane = tl.arange($(128))
    for r7_group in range($(4)) {
      r7_left = tl.load($(candidate_vbaddbmm_ptr) + (((r7_group % $(4)) * $(16384)) + ((r7_row[:, None] % $(128)) * $(128)) + (r7_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r7_right = tl.load($(candidate_vdiv_ptr) + (((r7_group % $(4)) * $(32768)) + ((r7_lane[:, None] % $(128)) * $(256)) + (r7_col[None, :] % $(256))), dtype=tl.bfloat16)
      r7_value = tl.dot(r7_left, r7_right)
      r7_value = r7_value + tl.load($(candidate_vdiv_ptr) + (((r7_group % $(4)) * $(32768)) + ((r7_row[:, None] % $(128)) * $(256)) + (r7_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
      r7_value = r7_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_1_ptr) + ((r7_group * $(32768)) + (r7_row[:, None] * $(256)) + r7_col[None, :]), r7_value)
    }
    r8_row = tl.arange($(128))
    r8_col = tl.arange($(128))
    r8_lane = tl.arange($(256))
    for r8_group in range($(4)) {
      r8_left = tl.load($(candidate_vbaddbmm_1_ptr) + (((((((r8_group % $(4)) * $(32768)) + (r8_row[:, None] * $(256)) + r8_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r8_group % $(4)) * $(32768)) + (r8_row[:, None] * $(256)) + r8_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r8_group % $(4)) * $(32768)) + (r8_row[:, None] * $(256)) + r8_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r8_right = tl.load($(candidate_vbaddbmm_1_ptr) + (((((((r8_group % $(4)) * $(32768)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r8_group % $(4)) * $(32768)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) % $(128)) * $(256)) + (((((r8_group % $(4)) * $(32768)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r8_value = tl.dot(r8_left, r8_right)
      r8_value = r8_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_1_ptr) + ((r8_group * $(16384)) + (r8_row[:, None] * $(128)) + r8_col[None, :]), r8_value)
    }
    r9_row = tl.arange($(128))
    r9_col = tl.arange($(128))
    r9_lane = tl.arange($(128))
    for r9_group in range($(4)) {
      r9_left = tl.load($(candidate_vbmm_1_ptr) + (((((((r9_group % $(4)) * $(16384)) + (r9_row[:, None] * $(128)) + r9_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r9_group % $(4)) * $(16384)) + (r9_row[:, None] * $(128)) + r9_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r9_group % $(4)) * $(16384)) + (r9_row[:, None] * $(128)) + r9_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r9_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r9_group % $(4)) * $(16384)) + (r9_lane[:, None] * $(128)) + r9_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r9_group % $(4)) * $(16384)) + (r9_lane[:, None] * $(128)) + r9_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r9_group % $(4)) * $(16384)) + (r9_lane[:, None] * $(128)) + r9_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r9_value = tl.dot(r9_left, r9_right)
      r9_value = r9_value * 2.0315
      r9_value = r9_value + tl.load($(candidate_vbmm_1_ptr) + (((((((r9_group % $(4)) * $(16384)) + (r9_row[:, None] * $(128)) + r9_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r9_group % $(4)) * $(16384)) + (r9_row[:, None] * $(128)) + r9_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r9_group % $(4)) * $(16384)) + (r9_row[:, None] * $(128)) + r9_col[None, :]) % $(128))), dtype=tl.bfloat16) * -4.775
      r9_value = r9_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_2_ptr) + ((r9_group * $(16384)) + (r9_row[:, None] * $(128)) + r9_col[None, :]), r9_value)
    }
    r10_row = tl.arange($(128))
    r10_col = tl.arange($(256))
    r10_lane = tl.arange($(128))
    for r10_group in range($(4)) {
      r10_left = tl.load($(candidate_vbaddbmm_2_ptr) + (((r10_group % $(4)) * $(16384)) + ((r10_row[:, None] % $(128)) * $(128)) + (r10_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r10_right = tl.load($(candidate_vbaddbmm_1_ptr) + (((r10_group % $(4)) * $(32768)) + ((r10_lane[:, None] % $(128)) * $(256)) + (r10_col[None, :] % $(256))), dtype=tl.bfloat16)
      r10_value = tl.dot(r10_left, r10_right)
      r10_value = r10_value + tl.load($(candidate_vbaddbmm_1_ptr) + (((r10_group % $(4)) * $(32768)) + ((r10_row[:, None] % $(128)) * $(256)) + (r10_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
      r10_value = r10_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_3_ptr) + ((r10_group * $(32768)) + (r10_row[:, None] * $(256)) + r10_col[None, :]), r10_value)
    }
    r11_row = tl.arange($(128))
    r11_col = tl.arange($(128))
    r11_lane = tl.arange($(256))
    for r11_group in range($(4)) {
      r11_left = tl.load($(candidate_vbaddbmm_3_ptr) + (((((((r11_group % $(4)) * $(32768)) + (r11_row[:, None] * $(256)) + r11_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r11_group % $(4)) * $(32768)) + (r11_row[:, None] * $(256)) + r11_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r11_group % $(4)) * $(32768)) + (r11_row[:, None] * $(256)) + r11_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r11_right = tl.load($(candidate_vbaddbmm_3_ptr) + (((((((r11_group % $(4)) * $(32768)) + (r11_lane[:, None] * $(128)) + r11_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r11_group % $(4)) * $(32768)) + (r11_lane[:, None] * $(128)) + r11_col[None, :]) % $(128)) * $(256)) + (((((r11_group % $(4)) * $(32768)) + (r11_lane[:, None] * $(128)) + r11_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r11_value = tl.dot(r11_left, r11_right)
      r11_value = r11_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r11_group * $(16384)) + (r11_row[:, None] * $(128)) + r11_col[None, :]), r11_value)
    }
    r12_row = tl.arange($(128))
    r12_col = tl.arange($(128))
    r12_lane = tl.arange($(128))
    for r12_group in range($(4)) {
      r12_left = tl.load($(candidate_vbmm_2_ptr) + (((((((r12_group % $(4)) * $(16384)) + (r12_row[:, None] * $(128)) + r12_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r12_group % $(4)) * $(16384)) + (r12_row[:, None] * $(128)) + r12_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r12_group % $(4)) * $(16384)) + (r12_row[:, None] * $(128)) + r12_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r12_right = tl.load($(candidate_vbmm_2_ptr) + (((((((r12_group % $(4)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r12_group % $(4)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r12_group % $(4)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r12_value = tl.dot(r12_left, r12_right)
      r12_value = r12_value * 2.0315
      r12_value = r12_value + tl.load($(candidate_vbmm_2_ptr) + (((((((r12_group % $(4)) * $(16384)) + (r12_row[:, None] * $(128)) + r12_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r12_group % $(4)) * $(16384)) + (r12_row[:, None] * $(128)) + r12_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r12_group % $(4)) * $(16384)) + (r12_row[:, None] * $(128)) + r12_col[None, :]) % $(128))), dtype=tl.bfloat16) * -4.775
      r12_value = r12_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_4_ptr) + ((r12_group * $(16384)) + (r12_row[:, None] * $(128)) + r12_col[None, :]), r12_value)
    }
    r13_row = tl.arange($(128))
    r13_col = tl.arange($(256))
    r13_lane = tl.arange($(128))
    for r13_group in range($(4)) {
      r13_left = tl.load($(candidate_vbaddbmm_4_ptr) + (((r13_group % $(4)) * $(16384)) + ((r13_row[:, None] % $(128)) * $(128)) + (r13_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r13_right = tl.load($(candidate_vbaddbmm_3_ptr) + (((r13_group % $(4)) * $(32768)) + ((r13_lane[:, None] % $(128)) * $(256)) + (r13_col[None, :] % $(256))), dtype=tl.bfloat16)
      r13_value = tl.dot(r13_left, r13_right)
      r13_value = r13_value + tl.load($(candidate_vbaddbmm_3_ptr) + (((r13_group % $(4)) * $(32768)) + ((r13_row[:, None] % $(128)) * $(256)) + (r13_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
      r13_value = r13_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_5_ptr) + ((r13_group * $(32768)) + (r13_row[:, None] * $(256)) + r13_col[None, :]), r13_value)
    }
    r14_row = tl.arange($(128))
    r14_col = tl.arange($(128))
    r14_lane = tl.arange($(256))
    for r14_group in range($(4)) {
      r14_left = tl.load($(candidate_vbaddbmm_5_ptr) + (((((((r14_group % $(4)) * $(32768)) + (r14_row[:, None] * $(256)) + r14_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r14_group % $(4)) * $(32768)) + (r14_row[:, None] * $(256)) + r14_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r14_group % $(4)) * $(32768)) + (r14_row[:, None] * $(256)) + r14_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r14_right = tl.load($(candidate_vbaddbmm_5_ptr) + (((((((r14_group % $(4)) * $(32768)) + (r14_lane[:, None] * $(128)) + r14_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r14_group % $(4)) * $(32768)) + (r14_lane[:, None] * $(128)) + r14_col[None, :]) % $(128)) * $(256)) + (((((r14_group % $(4)) * $(32768)) + (r14_lane[:, None] * $(128)) + r14_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r14_value = tl.dot(r14_left, r14_right)
      r14_value = r14_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_3_ptr) + ((r14_group * $(16384)) + (r14_row[:, None] * $(128)) + r14_col[None, :]), r14_value)
    }
    r15_row = tl.arange($(128))
    r15_col = tl.arange($(128))
    r15_lane = tl.arange($(128))
    for r15_group in range($(4)) {
      r15_left = tl.load($(candidate_vbmm_3_ptr) + (((((((r15_group % $(4)) * $(16384)) + (r15_row[:, None] * $(128)) + r15_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r15_group % $(4)) * $(16384)) + (r15_row[:, None] * $(128)) + r15_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r15_group % $(4)) * $(16384)) + (r15_row[:, None] * $(128)) + r15_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r15_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r15_group % $(4)) * $(16384)) + (r15_lane[:, None] * $(128)) + r15_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r15_group % $(4)) * $(16384)) + (r15_lane[:, None] * $(128)) + r15_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r15_group % $(4)) * $(16384)) + (r15_lane[:, None] * $(128)) + r15_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r15_value = tl.dot(r15_left, r15_right)
      r15_value = r15_value * 2.0315
      r15_value = r15_value + tl.load($(candidate_vbmm_3_ptr) + (((((((r15_group % $(4)) * $(16384)) + (r15_row[:, None] * $(128)) + r15_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r15_group % $(4)) * $(16384)) + (r15_row[:, None] * $(128)) + r15_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r15_group % $(4)) * $(16384)) + (r15_row[:, None] * $(128)) + r15_col[None, :]) % $(128))), dtype=tl.bfloat16) * -4.775
      r15_value = r15_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_6_ptr) + ((r15_group * $(16384)) + (r15_row[:, None] * $(128)) + r15_col[None, :]), r15_value)
    }
    r16_row = tl.arange($(128))
    r16_col = tl.arange($(256))
    r16_lane = tl.arange($(128))
    for r16_group in range($(4)) {
      r16_left = tl.load($(candidate_vbaddbmm_6_ptr) + (((r16_group % $(4)) * $(16384)) + ((r16_row[:, None] % $(128)) * $(128)) + (r16_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r16_right = tl.load($(candidate_vbaddbmm_5_ptr) + (((r16_group % $(4)) * $(32768)) + ((r16_lane[:, None] % $(128)) * $(256)) + (r16_col[None, :] % $(256))), dtype=tl.bfloat16)
      r16_value = tl.dot(r16_left, r16_right)
      r16_value = r16_value + tl.load($(candidate_vbaddbmm_5_ptr) + (((r16_group % $(4)) * $(32768)) + ((r16_row[:, None] % $(128)) * $(256)) + (r16_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
      r16_value = r16_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_7_ptr) + ((r16_group * $(32768)) + (r16_row[:, None] * $(256)) + r16_col[None, :]), r16_value)
    }
    r17_row = tl.arange($(128))
    r17_col = tl.arange($(128))
    r17_lane = tl.arange($(256))
    for r17_group in range($(4)) {
      r17_left = tl.load($(candidate_vbaddbmm_7_ptr) + (((((((r17_group % $(4)) * $(32768)) + (r17_row[:, None] * $(256)) + r17_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r17_group % $(4)) * $(32768)) + (r17_row[:, None] * $(256)) + r17_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r17_group % $(4)) * $(32768)) + (r17_row[:, None] * $(256)) + r17_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r17_right = tl.load($(candidate_vbaddbmm_7_ptr) + (((((((r17_group % $(4)) * $(32768)) + (r17_lane[:, None] * $(128)) + r17_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r17_group % $(4)) * $(32768)) + (r17_lane[:, None] * $(128)) + r17_col[None, :]) % $(128)) * $(256)) + (((((r17_group % $(4)) * $(32768)) + (r17_lane[:, None] * $(128)) + r17_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r17_value = tl.dot(r17_left, r17_right)
      r17_value = r17_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_4_ptr) + ((r17_group * $(16384)) + (r17_row[:, None] * $(128)) + r17_col[None, :]), r17_value)
    }
    r18_row = tl.arange($(128))
    r18_col = tl.arange($(128))
    r18_lane = tl.arange($(128))
    for r18_group in range($(4)) {
      r18_left = tl.load($(candidate_vbmm_4_ptr) + (((((((r18_group % $(4)) * $(16384)) + (r18_row[:, None] * $(128)) + r18_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r18_group % $(4)) * $(16384)) + (r18_row[:, None] * $(128)) + r18_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r18_group % $(4)) * $(16384)) + (r18_row[:, None] * $(128)) + r18_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r18_right = tl.load($(candidate_vbmm_4_ptr) + (((((((r18_group % $(4)) * $(16384)) + (r18_lane[:, None] * $(128)) + r18_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r18_group % $(4)) * $(16384)) + (r18_lane[:, None] * $(128)) + r18_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r18_group % $(4)) * $(16384)) + (r18_lane[:, None] * $(128)) + r18_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r18_value = tl.dot(r18_left, r18_right)
      r18_value = r18_value * 2.0315
      r18_value = r18_value + tl.load($(candidate_vbmm_4_ptr) + (((((((r18_group % $(4)) * $(16384)) + (r18_row[:, None] * $(128)) + r18_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r18_group % $(4)) * $(16384)) + (r18_row[:, None] * $(128)) + r18_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r18_group % $(4)) * $(16384)) + (r18_row[:, None] * $(128)) + r18_col[None, :]) % $(128))), dtype=tl.bfloat16) * -4.775
      r18_value = r18_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_8_ptr) + ((r18_group * $(16384)) + (r18_row[:, None] * $(128)) + r18_col[None, :]), r18_value)
    }
    r19_row = tl.arange($(128))
    r19_col = tl.arange($(256))
    r19_lane = tl.arange($(128))
    for r19_group in range($(4)) {
      r19_left = tl.load($(candidate_vbaddbmm_8_ptr) + (((r19_group % $(4)) * $(16384)) + ((r19_row[:, None] % $(128)) * $(128)) + (r19_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r19_right = tl.load($(candidate_vbaddbmm_7_ptr) + (((r19_group % $(4)) * $(32768)) + ((r19_lane[:, None] % $(128)) * $(256)) + (r19_col[None, :] % $(256))), dtype=tl.bfloat16)
      r19_value = tl.dot(r19_left, r19_right)
      r19_value = r19_value + tl.load($(candidate_vbaddbmm_7_ptr) + (((r19_group % $(4)) * $(32768)) + ((r19_row[:, None] % $(128)) * $(256)) + (r19_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
      r19_value = r19_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_9_ptr) + ((r19_group * $(32768)) + (r19_row[:, None] * $(256)) + r19_col[None, :]), r19_value)
    }
    r20_row = tl.arange($(128))
    r20_col = tl.arange($(128))
    r20_lane = tl.arange($(256))
    for r20_group in range($(4)) {
      r20_left = tl.load($(candidate_vbaddbmm_9_ptr) + (((((((r20_group % $(4)) * $(32768)) + (r20_row[:, None] * $(256)) + r20_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r20_group % $(4)) * $(32768)) + (r20_row[:, None] * $(256)) + r20_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r20_group % $(4)) * $(32768)) + (r20_row[:, None] * $(256)) + r20_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r20_right = tl.load($(candidate_vbaddbmm_9_ptr) + (((((((r20_group % $(4)) * $(32768)) + (r20_lane[:, None] * $(128)) + r20_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r20_group % $(4)) * $(32768)) + (r20_lane[:, None] * $(128)) + r20_col[None, :]) % $(128)) * $(256)) + (((((r20_group % $(4)) * $(32768)) + (r20_lane[:, None] * $(128)) + r20_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r20_value = tl.dot(r20_left, r20_right)
      r20_value = r20_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r20_group * $(16384)) + (r20_row[:, None] * $(128)) + r20_col[None, :]), r20_value)
    }
    r21_row = tl.arange($(128))
    r21_col = tl.arange($(128))
    r21_lane = tl.arange($(128))
    for r21_group in range($(4)) {
      r21_left = tl.load($(candidate_vbmm_5_ptr) + (((((((r21_group % $(4)) * $(16384)) + (r21_row[:, None] * $(128)) + r21_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r21_group % $(4)) * $(16384)) + (r21_row[:, None] * $(128)) + r21_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r21_group % $(4)) * $(16384)) + (r21_row[:, None] * $(128)) + r21_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r21_right = tl.load($(candidate_vbmm_5_ptr) + (((((((r21_group % $(4)) * $(16384)) + (r21_lane[:, None] * $(128)) + r21_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r21_group % $(4)) * $(16384)) + (r21_lane[:, None] * $(128)) + r21_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r21_group % $(4)) * $(16384)) + (r21_lane[:, None] * $(128)) + r21_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r21_value = tl.dot(r21_left, r21_right)
      r21_value = r21_value * 2.0315
      r21_value = r21_value + tl.load($(candidate_vbmm_5_ptr) + (((((((r21_group % $(4)) * $(16384)) + (r21_row[:, None] * $(128)) + r21_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r21_group % $(4)) * $(16384)) + (r21_row[:, None] * $(128)) + r21_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r21_group % $(4)) * $(16384)) + (r21_row[:, None] * $(128)) + r21_col[None, :]) % $(128))), dtype=tl.bfloat16) * -4.775
      r21_value = r21_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_10_ptr) + ((r21_group * $(16384)) + (r21_row[:, None] * $(128)) + r21_col[None, :]), r21_value)
    }
    r22_row = tl.arange($(128))
    r22_col = tl.arange($(256))
    r22_lane = tl.arange($(128))
    for r22_group in range($(4)) {
      r22_left = tl.load($(candidate_vbaddbmm_10_ptr) + (((r22_group % $(4)) * $(16384)) + ((r22_row[:, None] % $(128)) * $(128)) + (r22_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r22_right = tl.load($(candidate_vbaddbmm_9_ptr) + (((r22_group % $(4)) * $(32768)) + ((r22_lane[:, None] % $(128)) * $(256)) + (r22_col[None, :] % $(256))), dtype=tl.bfloat16)
      r22_value = tl.dot(r22_left, r22_right)
      r22_value = r22_value + tl.load($(candidate_vbaddbmm_9_ptr) + (((r22_group % $(4)) * $(32768)) + ((r22_row[:, None] % $(128)) * $(256)) + (r22_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
      r22_value = r22_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_11_ptr) + ((r22_group * $(32768)) + (r22_row[:, None] * $(256)) + r22_col[None, :]), r22_value)
    }
    r23_row = tl.arange($(128))
    r23_col = tl.arange($(128))
    r23_lane = tl.arange($(256))
    for r23_group in range($(4)) {
      r23_left = tl.load($(candidate_vbaddbmm_11_ptr) + (((((((r23_group % $(4)) * $(32768)) + (r23_row[:, None] * $(256)) + r23_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r23_group % $(4)) * $(32768)) + (r23_row[:, None] * $(256)) + r23_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r23_group % $(4)) * $(32768)) + (r23_row[:, None] * $(256)) + r23_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r23_right = tl.load($(candidate_vbaddbmm_11_ptr) + (((((((r23_group % $(4)) * $(32768)) + (r23_lane[:, None] * $(128)) + r23_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r23_group % $(4)) * $(32768)) + (r23_lane[:, None] * $(128)) + r23_col[None, :]) % $(128)) * $(256)) + (((((r23_group % $(4)) * $(32768)) + (r23_lane[:, None] * $(128)) + r23_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r23_value = tl.dot(r23_left, r23_right)
      r23_value = r23_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r23_group * $(16384)) + (r23_row[:, None] * $(128)) + r23_col[None, :]), r23_value)
    }
    r24_row = tl.arange($(128))
    r24_col = tl.arange($(128))
    r24_lane = tl.arange($(128))
    for r24_group in range($(4)) {
      r24_left = tl.load($(candidate_vbmm_6_ptr) + (((((((r24_group % $(4)) * $(16384)) + (r24_row[:, None] * $(128)) + r24_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r24_group % $(4)) * $(16384)) + (r24_row[:, None] * $(128)) + r24_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r24_group % $(4)) * $(16384)) + (r24_row[:, None] * $(128)) + r24_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r24_right = tl.load($(candidate_vbmm_6_ptr) + (((((((r24_group % $(4)) * $(16384)) + (r24_lane[:, None] * $(128)) + r24_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r24_group % $(4)) * $(16384)) + (r24_lane[:, None] * $(128)) + r24_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r24_group % $(4)) * $(16384)) + (r24_lane[:, None] * $(128)) + r24_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r24_value = tl.dot(r24_left, r24_right)
      r24_value = r24_value * 2.0315
      r24_value = r24_value + tl.load($(candidate_vbmm_6_ptr) + (((((((r24_group % $(4)) * $(16384)) + (r24_row[:, None] * $(128)) + r24_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r24_group % $(4)) * $(16384)) + (r24_row[:, None] * $(128)) + r24_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r24_group % $(4)) * $(16384)) + (r24_row[:, None] * $(128)) + r24_col[None, :]) % $(128))), dtype=tl.bfloat16) * -4.775
      r24_value = r24_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_12_ptr) + ((r24_group * $(16384)) + (r24_row[:, None] * $(128)) + r24_col[None, :]), r24_value)
    }
    r25_row = tl.arange($(128))
    r25_col = tl.arange($(256))
    r25_lane = tl.arange($(128))
    for r25_group in range($(4)) {
      r25_left = tl.load($(candidate_vbaddbmm_12_ptr) + (((r25_group % $(4)) * $(16384)) + ((r25_row[:, None] % $(128)) * $(128)) + (r25_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r25_right = tl.load($(candidate_vbaddbmm_11_ptr) + (((r25_group % $(4)) * $(32768)) + ((r25_lane[:, None] % $(128)) * $(256)) + (r25_col[None, :] % $(256))), dtype=tl.bfloat16)
      r25_value = tl.dot(r25_left, r25_right)
      r25_value = r25_value + tl.load($(candidate_vbaddbmm_11_ptr) + (((r25_group % $(4)) * $(32768)) + ((r25_row[:, None] % $(128)) * $(256)) + (r25_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
      r25_value = r25_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_13_ptr) + ((r25_group * $(32768)) + (r25_row[:, None] * $(256)) + r25_col[None, :]), r25_value)
    }
    r26_row = tl.arange($(128))
    r26_col = tl.arange($(128))
    r26_lane = tl.arange($(256))
    for r26_group in range($(4)) {
      r26_left = tl.load($(candidate_vbaddbmm_13_ptr) + (((((((r26_group % $(4)) * $(32768)) + (r26_row[:, None] * $(256)) + r26_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r26_group % $(4)) * $(32768)) + (r26_row[:, None] * $(256)) + r26_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r26_group % $(4)) * $(32768)) + (r26_row[:, None] * $(256)) + r26_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r26_right = tl.load($(candidate_vbaddbmm_13_ptr) + (((((((r26_group % $(4)) * $(32768)) + (r26_lane[:, None] * $(128)) + r26_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r26_group % $(4)) * $(32768)) + (r26_lane[:, None] * $(128)) + r26_col[None, :]) % $(128)) * $(256)) + (((((r26_group % $(4)) * $(32768)) + (r26_lane[:, None] * $(128)) + r26_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r26_value = tl.dot(r26_left, r26_right)
      r26_value = r26_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_7_ptr) + ((r26_group * $(16384)) + (r26_row[:, None] * $(128)) + r26_col[None, :]), r26_value)
    }
    r27_row = tl.arange($(128))
    r27_col = tl.arange($(128))
    r27_lane = tl.arange($(128))
    for r27_group in range($(4)) {
      r27_left = tl.load($(candidate_vbmm_7_ptr) + (((((((r27_group % $(4)) * $(16384)) + (r27_row[:, None] * $(128)) + r27_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r27_group % $(4)) * $(16384)) + (r27_row[:, None] * $(128)) + r27_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r27_group % $(4)) * $(16384)) + (r27_row[:, None] * $(128)) + r27_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r27_right = tl.load($(candidate_vbmm_7_ptr) + (((((((r27_group % $(4)) * $(16384)) + (r27_lane[:, None] * $(128)) + r27_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r27_group % $(4)) * $(16384)) + (r27_lane[:, None] * $(128)) + r27_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r27_group % $(4)) * $(16384)) + (r27_lane[:, None] * $(128)) + r27_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r27_value = tl.dot(r27_left, r27_right)
      r27_value = r27_value * 2.0315
      r27_value = r27_value + tl.load($(candidate_vbmm_7_ptr) + (((((((r27_group % $(4)) * $(16384)) + (r27_row[:, None] * $(128)) + r27_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r27_group % $(4)) * $(16384)) + (r27_row[:, None] * $(128)) + r27_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r27_group % $(4)) * $(16384)) + (r27_row[:, None] * $(128)) + r27_col[None, :]) % $(128))), dtype=tl.bfloat16) * -4.775
      r27_value = r27_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_14_ptr) + ((r27_group * $(16384)) + (r27_row[:, None] * $(128)) + r27_col[None, :]), r27_value)
    }
    r28_row = tl.arange($(128))
    r28_col = tl.arange($(256))
    r28_lane = tl.arange($(128))
    for r28_group in range($(4)) {
      r28_left = tl.load($(candidate_vbaddbmm_14_ptr) + (((r28_group % $(4)) * $(16384)) + ((r28_row[:, None] % $(128)) * $(128)) + (r28_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r28_right = tl.load($(candidate_vbaddbmm_13_ptr) + (((r28_group % $(4)) * $(32768)) + ((r28_lane[:, None] % $(128)) * $(256)) + (r28_col[None, :] % $(256))), dtype=tl.bfloat16)
      r28_value = tl.dot(r28_left, r28_right)
      r28_value = r28_value + tl.load($(candidate_vbaddbmm_13_ptr) + (((r28_group % $(4)) * $(32768)) + ((r28_row[:, None] % $(128)) * $(256)) + (r28_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
      r28_value = r28_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_15_ptr) + ((r28_group * $(32768)) + (r28_row[:, None] * $(256)) + r28_col[None, :]), r28_value)
    }
    r29_row = tl.arange($(128))
    r29_col = tl.arange($(128))
    r29_lane = tl.arange($(256))
    for r29_group in range($(4)) {
      r29_left = tl.load($(candidate_vbaddbmm_15_ptr) + (((((((r29_group % $(4)) * $(32768)) + (r29_row[:, None] * $(256)) + r29_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r29_group % $(4)) * $(32768)) + (r29_row[:, None] * $(256)) + r29_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r29_group % $(4)) * $(32768)) + (r29_row[:, None] * $(256)) + r29_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r29_right = tl.load($(candidate_vbaddbmm_15_ptr) + (((((((r29_group % $(4)) * $(32768)) + (r29_lane[:, None] * $(128)) + r29_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r29_group % $(4)) * $(32768)) + (r29_lane[:, None] * $(128)) + r29_col[None, :]) % $(128)) * $(256)) + (((((r29_group % $(4)) * $(32768)) + (r29_lane[:, None] * $(128)) + r29_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r29_value = tl.dot(r29_left, r29_right)
      r29_value = r29_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_8_ptr) + ((r29_group * $(16384)) + (r29_row[:, None] * $(128)) + r29_col[None, :]), r29_value)
    }
    r30_row = tl.arange($(128))
    r30_col = tl.arange($(128))
    r30_lane = tl.arange($(128))
    for r30_group in range($(4)) {
      r30_left = tl.load($(candidate_vbmm_8_ptr) + (((((((r30_group % $(4)) * $(16384)) + (r30_row[:, None] * $(128)) + r30_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r30_group % $(4)) * $(16384)) + (r30_row[:, None] * $(128)) + r30_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r30_group % $(4)) * $(16384)) + (r30_row[:, None] * $(128)) + r30_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r30_right = tl.load($(candidate_vbmm_8_ptr) + (((((((r30_group % $(4)) * $(16384)) + (r30_lane[:, None] * $(128)) + r30_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r30_group % $(4)) * $(16384)) + (r30_lane[:, None] * $(128)) + r30_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r30_group % $(4)) * $(16384)) + (r30_lane[:, None] * $(128)) + r30_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r30_value = tl.dot(r30_left, r30_right)
      r30_value = r30_value * 0.5
      r30_value = r30_value + tl.load($(candidate_vbmm_8_ptr) + (((((((r30_group % $(4)) * $(16384)) + (r30_row[:, None] * $(128)) + r30_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r30_group % $(4)) * $(16384)) + (r30_row[:, None] * $(128)) + r30_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r30_group % $(4)) * $(16384)) + (r30_row[:, None] * $(128)) + r30_col[None, :]) % $(128))), dtype=tl.bfloat16) * -1.5
      r30_value = r30_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_16_ptr) + ((r30_group * $(16384)) + (r30_row[:, None] * $(128)) + r30_col[None, :]), r30_value)
    }
    r31_row = tl.arange($(128))
    r31_col = tl.arange($(256))
    r31_lane = tl.arange($(128))
    for r31_group in range($(4)) {
      r31_left = tl.load($(candidate_vbaddbmm_16_ptr) + (((r31_group % $(4)) * $(16384)) + ((r31_row[:, None] % $(128)) * $(128)) + (r31_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r31_right = tl.load($(candidate_vbaddbmm_15_ptr) + (((r31_group % $(4)) * $(32768)) + ((r31_lane[:, None] % $(128)) * $(256)) + (r31_col[None, :] % $(256))), dtype=tl.bfloat16)
      r31_value = tl.dot(r31_left, r31_right)
      r31_value = r31_value + tl.load($(candidate_vbaddbmm_15_ptr) + (((r31_group % $(4)) * $(32768)) + ((r31_row[:, None] % $(128)) * $(256)) + (r31_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
      r31_value = r31_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_17_ptr) + ((r31_group * $(32768)) + (r31_row[:, None] * $(256)) + r31_col[None, :]), r31_value)
    }
    r32_row = tl.arange($(128))
    r32_col = tl.arange($(128))
    r32_lane = tl.arange($(256))
    for r32_group in range($(4)) {
      r32_left = tl.load($(candidate_vbaddbmm_17_ptr) + (((((((r32_group % $(4)) * $(32768)) + (r32_row[:, None] * $(256)) + r32_lane[None, :]) / $(32768)) % $(4)) * $(32768)) + ((((((r32_group % $(4)) * $(32768)) + (r32_row[:, None] * $(256)) + r32_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r32_group % $(4)) * $(32768)) + (r32_row[:, None] * $(256)) + r32_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r32_right = tl.load($(candidate_vbaddbmm_17_ptr) + (((((((r32_group % $(4)) * $(32768)) + (r32_lane[:, None] * $(128)) + r32_col[None, :]) / $(32768)) % $(4)) * $(32768)) + (((((r32_group % $(4)) * $(32768)) + (r32_lane[:, None] * $(128)) + r32_col[None, :]) % $(128)) * $(256)) + (((((r32_group % $(4)) * $(32768)) + (r32_lane[:, None] * $(128)) + r32_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r32_value = tl.dot(r32_left, r32_right)
      r32_value = r32_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_9_ptr) + ((r32_group * $(16384)) + (r32_row[:, None] * $(128)) + r32_col[None, :]), r32_value)
    }
    r33_row = tl.arange($(128))
    r33_col = tl.arange($(128))
    r33_lane = tl.arange($(128))
    for r33_group in range($(4)) {
      r33_left = tl.load($(candidate_vbmm_9_ptr) + (((((((r33_group % $(4)) * $(16384)) + (r33_row[:, None] * $(128)) + r33_lane[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r33_group % $(4)) * $(16384)) + (r33_row[:, None] * $(128)) + r33_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r33_group % $(4)) * $(16384)) + (r33_row[:, None] * $(128)) + r33_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r33_right = tl.load($(candidate_vbmm_9_ptr) + (((((((r33_group % $(4)) * $(16384)) + (r33_lane[:, None] * $(128)) + r33_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r33_group % $(4)) * $(16384)) + (r33_lane[:, None] * $(128)) + r33_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r33_group % $(4)) * $(16384)) + (r33_lane[:, None] * $(128)) + r33_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r33_value = tl.dot(r33_left, r33_right)
      r33_value = r33_value * 0.5
      r33_value = r33_value + tl.load($(candidate_vbmm_9_ptr) + (((((((r33_group % $(4)) * $(16384)) + (r33_row[:, None] * $(128)) + r33_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r33_group % $(4)) * $(16384)) + (r33_row[:, None] * $(128)) + r33_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r33_group % $(4)) * $(16384)) + (r33_row[:, None] * $(128)) + r33_col[None, :]) % $(128))), dtype=tl.bfloat16) * -1.5
      r33_value = r33_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_18_ptr) + ((r33_group * $(16384)) + (r33_row[:, None] * $(128)) + r33_col[None, :]), r33_value)
    }
    r34_row = tl.arange($(128))
    r34_col = tl.arange($(256))
    r34_lane = tl.arange($(128))
    for r34_group in range($(4)) {
      r34_left = tl.load($(candidate_vbaddbmm_18_ptr) + (((r34_group % $(4)) * $(16384)) + ((r34_row[:, None] % $(128)) * $(128)) + (r34_lane[None, :] % $(128))), dtype=tl.bfloat16)
      r34_right = tl.load($(candidate_vbaddbmm_17_ptr) + (((r34_group % $(4)) * $(32768)) + ((r34_lane[:, None] % $(128)) * $(256)) + (r34_col[None, :] % $(256))), dtype=tl.bfloat16)
      r34_value = tl.dot(r34_left, r34_right)
      r34_value = r34_value + tl.load($(candidate_vbaddbmm_17_ptr) + (((r34_group % $(4)) * $(32768)) + ((r34_row[:, None] % $(128)) * $(256)) + (r34_col[None, :] % $(256))), dtype=tl.bfloat16) * 2.0
      r34_value = r34_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_19_ptr) + ((r34_group * $(32768)) + (r34_row[:, None] * $(256)) + r34_col[None, :]), r34_value)
    }
    r35_z = tl.arange($(131072))
    r35_value = (tl.load($(parameter0_ptr) + ((((r35_z / $(32768)) % $(4)) * $(32768)) + (((r35_z / $(128)) % $(256)) * $(128)) + (r35_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r35_value = r35_value.to(tl.bfloat16)
    r35_value = tl.broadcast_to(r35_value, [$(131072)])
    tl.store($(candidate_vmul_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(131072))
    r36_value = (tl.load($(candidate_vmul_ptr) + ((((r36_z / $(32768)) % $(4)) * $(32768)) + (((r36_z / $(128)) % $(256)) * $(128)) + (r36_z % $(128))), dtype=tl.bfloat16) + (tl.load($(candidate_vbaddbmm_19_ptr) + ((((r36_z / $(32768)) % $(4)) * $(32768)) + ((r36_z % $(128)) * $(256)) + ((r36_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(131072)])
    tl.store($(candidate_vadd_ptr) + r36_z, r36_value)
    z = tl.arange($(131072))
    value = tl.load($(candidate_vadd_ptr) + ((((z / $(32768)) % $(4)) * $(32768)) + (((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_muon_ns_schedule_bmm_vs_loop
