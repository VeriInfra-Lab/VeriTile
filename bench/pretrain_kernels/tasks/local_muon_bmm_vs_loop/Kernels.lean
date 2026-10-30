import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_muon_bmm_vs_loop
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
    r21_z = tl.arange($(32768))
    r21_value = (tl.load($(parameter0_ptr) + ((((r21_z / $(128)) % $(256)) * $(128)) + (r21_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r21_value = r21_value.to(tl.bfloat16)
    r21_value = tl.broadcast_to(r21_value, [$(32768)])
    tl.store($(vmul_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(131072))
    r22_value = tl.where((((r22_z / $(32768)) % $(4)) == $(0)), tl.load($(vmul_ptr) + ((((r22_z / $(128)) % $(256)) * $(128)) + (r22_z % $(128))), dtype=tl.bfloat16), tl.load($(parameter0_ptr) + ((((r22_z / $(32768)) % $(4)) * $(32768)) + (((r22_z / $(128)) % $(256)) * $(128)) + (r22_z % $(128))), dtype=tl.bfloat16))
    r22_value = r22_value.to(tl.bfloat16)
    r22_value = tl.broadcast_to(r22_value, [$(131072)])
    tl.store($(vselect_scatter_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(32768))
    r23_value = (tl.load($(vselect_scatter_1_ptr) + ((((r23_z / $(128)) % $(256)) * $(128)) + (r23_z % $(128))), dtype=tl.bfloat16) + (tl.load($(vaddmm_9_ptr) + (((r23_z % $(128)) * $(256)) + ((r23_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r23_value = r23_value.to(tl.bfloat16)
    r23_value = tl.broadcast_to(r23_value, [$(32768)])
    tl.store($(vadd_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(131072))
    r24_value = tl.where((((r24_z / $(32768)) % $(4)) == $(0)), tl.load($(vadd_ptr) + ((((r24_z / $(128)) % $(256)) * $(128)) + (r24_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_1_ptr) + ((((r24_z / $(32768)) % $(4)) * $(32768)) + (((r24_z / $(128)) % $(256)) * $(128)) + (r24_z % $(128))), dtype=tl.bfloat16))
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(131072)])
    tl.store($(vselect_scatter_2_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(32768))
    r25_value = (tl.load($(vselect_scatter_ptr) + ((((r25_z / $(128)) % $(256)) * $(128)) + (r25_z % $(128)) + $(32768)), dtype=tl.bfloat16) + (tl.load($(input0_ptr) + ((((r25_z / $(128)) % $(256)) * $(128)) + (r25_z % $(128)) + $(32768)), dtype=tl.bfloat16) - tl.load($(vselect_scatter_ptr) + ((((r25_z / $(128)) % $(256)) * $(128)) + (r25_z % $(128)) + $(32768)), dtype=tl.bfloat16)) * 0.050000000000000044)
    r25_value = r25_value.to(tl.bfloat16)
    r25_value = tl.broadcast_to(r25_value, [$(32768)])
    tl.store($(vlerp_2_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(131072))
    r26_value = tl.where((((r26_z / $(32768)) % $(4)) == $(1)), tl.load($(vlerp_2_ptr) + ((((r26_z / $(128)) % $(256)) * $(128)) + (r26_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_ptr) + ((((r26_z / $(32768)) % $(4)) * $(32768)) + (((r26_z / $(128)) % $(256)) * $(128)) + (r26_z % $(128))), dtype=tl.bfloat16))
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(131072)])
    tl.store($(vselect_scatter_3_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(32768))
    r27_value = (tl.load($(input0_ptr) + ((((r27_z / $(128)) % $(256)) * $(128)) + (r27_z % $(128)) + $(32768)), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_3_ptr) + ((((r27_z / $(128)) % $(256)) * $(128)) + (r27_z % $(128)) + $(32768)), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r27_z / $(128)) % $(256)) * $(128)) + (r27_z % $(128)) + $(32768)), dtype=tl.bfloat16)) * 0.95)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(32768)])
    tl.store($(vlerp_3_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(1))
    r28_lane = tl.arange($(32768))
    r28_value = tl.sqrt(tl.sum(tl.load($(vlerp_3_ptr) + (((r28_lane[None, :] % $(256)) * $(128)) + ((r28_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16) * tl.load($(vlerp_3_ptr) + (((r28_lane[None, :] % $(256)) * $(128)) + ((r28_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16), axis=1))
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(1)])
    tl.store($(vlinalg_vector_norm_1_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(1))
    r29_value = tl.maximum(tl.load($(vlinalg_vector_norm_1_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r29_value = r29_value.to(tl.bfloat16)
    r29_value = tl.broadcast_to(r29_value, [$(1)])
    tl.store($(vclamp_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(32768))
    r30_value = (tl.load($(vlerp_3_ptr) + (((r30_z % $(256)) * $(128)) + ((r30_z / $(256)) % $(128))), dtype=tl.bfloat16) / tl.load($(vclamp_1_ptr) + $(0), dtype=tl.bfloat16))
    r30_value = r30_value.to(tl.bfloat16)
    r30_value = tl.broadcast_to(r30_value, [$(32768)])
    tl.store($(vdiv_1_ptr) + r30_z, r30_value)
    r31_row = tl.arange($(128))
    r31_col = tl.arange($(128))
    r31_lane = tl.arange($(256))
    r31_left = tl.load($(vdiv_1_ptr) + (((r31_row[:, None] % $(128)) * $(256)) + (r31_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r31_right = tl.load($(vdiv_1_ptr) + (((r31_col[None, :] % $(128)) * $(256)) + (r31_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r31_value = tl.dot(r31_left, r31_right)
    r31_value = r31_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r31_row[:, None] * $(128)) + r31_col[None, :]), r31_value)
    r32_row = tl.arange($(128))
    r32_col = tl.arange($(128))
    r32_lane = tl.arange($(128))
    r32_left = tl.load($(vmm_5_ptr) + (((r32_row[:, None] % $(128)) * $(128)) + (r32_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r32_right = tl.load($(vmm_5_ptr) + (((r32_lane[:, None] % $(128)) * $(128)) + (r32_col[None, :] % $(128))), dtype=tl.bfloat16)
    r32_value = tl.dot(r32_left, r32_right)
    r32_value = r32_value * 2.0315
    r32_value = r32_value + tl.load($(vmm_5_ptr) + (((r32_row[:, None] % $(128)) * $(128)) + (r32_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r32_value = r32_value.to(tl.bfloat16)
    tl.store($(vaddmm_10_ptr) + ((r32_row[:, None] * $(128)) + r32_col[None, :]), r32_value)
    r33_row = tl.arange($(128))
    r33_col = tl.arange($(256))
    r33_lane = tl.arange($(128))
    r33_left = tl.load($(vaddmm_10_ptr) + (((r33_row[:, None] % $(128)) * $(128)) + (r33_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r33_right = tl.load($(vdiv_1_ptr) + (((r33_lane[:, None] % $(128)) * $(256)) + (r33_col[None, :] % $(256))), dtype=tl.bfloat16)
    r33_value = tl.dot(r33_left, r33_right)
    r33_value = r33_value + tl.load($(vdiv_1_ptr) + (((r33_row[:, None] % $(128)) * $(256)) + (r33_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r33_value = r33_value.to(tl.bfloat16)
    tl.store($(vaddmm_11_ptr) + ((r33_row[:, None] * $(256)) + r33_col[None, :]), r33_value)
    r34_row = tl.arange($(128))
    r34_col = tl.arange($(128))
    r34_lane = tl.arange($(256))
    r34_left = tl.load($(vaddmm_11_ptr) + (((r34_row[:, None] % $(128)) * $(256)) + (r34_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r34_right = tl.load($(vaddmm_11_ptr) + (((r34_col[None, :] % $(128)) * $(256)) + (r34_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r34_value = tl.dot(r34_left, r34_right)
    r34_value = r34_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r34_row[:, None] * $(128)) + r34_col[None, :]), r34_value)
    r35_row = tl.arange($(128))
    r35_col = tl.arange($(128))
    r35_lane = tl.arange($(128))
    r35_left = tl.load($(vmm_6_ptr) + (((r35_row[:, None] % $(128)) * $(128)) + (r35_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r35_right = tl.load($(vmm_6_ptr) + (((r35_lane[:, None] % $(128)) * $(128)) + (r35_col[None, :] % $(128))), dtype=tl.bfloat16)
    r35_value = tl.dot(r35_left, r35_right)
    r35_value = r35_value * 2.0315
    r35_value = r35_value + tl.load($(vmm_6_ptr) + (((r35_row[:, None] % $(128)) * $(128)) + (r35_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r35_value = r35_value.to(tl.bfloat16)
    tl.store($(vaddmm_12_ptr) + ((r35_row[:, None] * $(128)) + r35_col[None, :]), r35_value)
    r36_row = tl.arange($(128))
    r36_col = tl.arange($(256))
    r36_lane = tl.arange($(128))
    r36_left = tl.load($(vaddmm_12_ptr) + (((r36_row[:, None] % $(128)) * $(128)) + (r36_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r36_right = tl.load($(vaddmm_11_ptr) + (((r36_lane[:, None] % $(128)) * $(256)) + (r36_col[None, :] % $(256))), dtype=tl.bfloat16)
    r36_value = tl.dot(r36_left, r36_right)
    r36_value = r36_value + tl.load($(vaddmm_11_ptr) + (((r36_row[:, None] % $(128)) * $(256)) + (r36_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r36_value = r36_value.to(tl.bfloat16)
    tl.store($(vaddmm_13_ptr) + ((r36_row[:, None] * $(256)) + r36_col[None, :]), r36_value)
    r37_row = tl.arange($(128))
    r37_col = tl.arange($(128))
    r37_lane = tl.arange($(256))
    r37_left = tl.load($(vaddmm_13_ptr) + (((r37_row[:, None] % $(128)) * $(256)) + (r37_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r37_right = tl.load($(vaddmm_13_ptr) + (((r37_col[None, :] % $(128)) * $(256)) + (r37_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r37_value = tl.dot(r37_left, r37_right)
    r37_value = r37_value.to(tl.bfloat16)
    tl.store($(vmm_7_ptr) + ((r37_row[:, None] * $(128)) + r37_col[None, :]), r37_value)
    r38_row = tl.arange($(128))
    r38_col = tl.arange($(128))
    r38_lane = tl.arange($(128))
    r38_left = tl.load($(vmm_7_ptr) + (((r38_row[:, None] % $(128)) * $(128)) + (r38_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r38_right = tl.load($(vmm_7_ptr) + (((r38_lane[:, None] % $(128)) * $(128)) + (r38_col[None, :] % $(128))), dtype=tl.bfloat16)
    r38_value = tl.dot(r38_left, r38_right)
    r38_value = r38_value * 2.0315
    r38_value = r38_value + tl.load($(vmm_7_ptr) + (((r38_row[:, None] % $(128)) * $(128)) + (r38_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r38_value = r38_value.to(tl.bfloat16)
    tl.store($(vaddmm_14_ptr) + ((r38_row[:, None] * $(128)) + r38_col[None, :]), r38_value)
    r39_row = tl.arange($(128))
    r39_col = tl.arange($(256))
    r39_lane = tl.arange($(128))
    r39_left = tl.load($(vaddmm_14_ptr) + (((r39_row[:, None] % $(128)) * $(128)) + (r39_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r39_right = tl.load($(vaddmm_13_ptr) + (((r39_lane[:, None] % $(128)) * $(256)) + (r39_col[None, :] % $(256))), dtype=tl.bfloat16)
    r39_value = tl.dot(r39_left, r39_right)
    r39_value = r39_value + tl.load($(vaddmm_13_ptr) + (((r39_row[:, None] % $(128)) * $(256)) + (r39_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r39_value = r39_value.to(tl.bfloat16)
    tl.store($(vaddmm_15_ptr) + ((r39_row[:, None] * $(256)) + r39_col[None, :]), r39_value)
    r40_row = tl.arange($(128))
    r40_col = tl.arange($(128))
    r40_lane = tl.arange($(256))
    r40_left = tl.load($(vaddmm_15_ptr) + (((r40_row[:, None] % $(128)) * $(256)) + (r40_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r40_right = tl.load($(vaddmm_15_ptr) + (((r40_col[None, :] % $(128)) * $(256)) + (r40_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r40_value = tl.dot(r40_left, r40_right)
    r40_value = r40_value.to(tl.bfloat16)
    tl.store($(vmm_8_ptr) + ((r40_row[:, None] * $(128)) + r40_col[None, :]), r40_value)
    r41_row = tl.arange($(128))
    r41_col = tl.arange($(128))
    r41_lane = tl.arange($(128))
    r41_left = tl.load($(vmm_8_ptr) + (((r41_row[:, None] % $(128)) * $(128)) + (r41_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r41_right = tl.load($(vmm_8_ptr) + (((r41_lane[:, None] % $(128)) * $(128)) + (r41_col[None, :] % $(128))), dtype=tl.bfloat16)
    r41_value = tl.dot(r41_left, r41_right)
    r41_value = r41_value * 2.0315
    r41_value = r41_value + tl.load($(vmm_8_ptr) + (((r41_row[:, None] % $(128)) * $(128)) + (r41_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r41_value = r41_value.to(tl.bfloat16)
    tl.store($(vaddmm_16_ptr) + ((r41_row[:, None] * $(128)) + r41_col[None, :]), r41_value)
    r42_row = tl.arange($(128))
    r42_col = tl.arange($(256))
    r42_lane = tl.arange($(128))
    r42_left = tl.load($(vaddmm_16_ptr) + (((r42_row[:, None] % $(128)) * $(128)) + (r42_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r42_right = tl.load($(vaddmm_15_ptr) + (((r42_lane[:, None] % $(128)) * $(256)) + (r42_col[None, :] % $(256))), dtype=tl.bfloat16)
    r42_value = tl.dot(r42_left, r42_right)
    r42_value = r42_value + tl.load($(vaddmm_15_ptr) + (((r42_row[:, None] % $(128)) * $(256)) + (r42_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r42_value = r42_value.to(tl.bfloat16)
    tl.store($(vaddmm_17_ptr) + ((r42_row[:, None] * $(256)) + r42_col[None, :]), r42_value)
    r43_row = tl.arange($(128))
    r43_col = tl.arange($(128))
    r43_lane = tl.arange($(256))
    r43_left = tl.load($(vaddmm_17_ptr) + (((r43_row[:, None] % $(128)) * $(256)) + (r43_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r43_right = tl.load($(vaddmm_17_ptr) + (((r43_col[None, :] % $(128)) * $(256)) + (r43_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r43_value = tl.dot(r43_left, r43_right)
    r43_value = r43_value.to(tl.bfloat16)
    tl.store($(vmm_9_ptr) + ((r43_row[:, None] * $(128)) + r43_col[None, :]), r43_value)
    r44_row = tl.arange($(128))
    r44_col = tl.arange($(128))
    r44_lane = tl.arange($(128))
    r44_left = tl.load($(vmm_9_ptr) + (((r44_row[:, None] % $(128)) * $(128)) + (r44_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r44_right = tl.load($(vmm_9_ptr) + (((r44_lane[:, None] % $(128)) * $(128)) + (r44_col[None, :] % $(128))), dtype=tl.bfloat16)
    r44_value = tl.dot(r44_left, r44_right)
    r44_value = r44_value * 2.0315
    r44_value = r44_value + tl.load($(vmm_9_ptr) + (((r44_row[:, None] % $(128)) * $(128)) + (r44_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r44_value = r44_value.to(tl.bfloat16)
    tl.store($(vaddmm_18_ptr) + ((r44_row[:, None] * $(128)) + r44_col[None, :]), r44_value)
    r45_row = tl.arange($(128))
    r45_col = tl.arange($(256))
    r45_lane = tl.arange($(128))
    r45_left = tl.load($(vaddmm_18_ptr) + (((r45_row[:, None] % $(128)) * $(128)) + (r45_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r45_right = tl.load($(vaddmm_17_ptr) + (((r45_lane[:, None] % $(128)) * $(256)) + (r45_col[None, :] % $(256))), dtype=tl.bfloat16)
    r45_value = tl.dot(r45_left, r45_right)
    r45_value = r45_value + tl.load($(vaddmm_17_ptr) + (((r45_row[:, None] % $(128)) * $(256)) + (r45_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r45_value = r45_value.to(tl.bfloat16)
    tl.store($(vaddmm_19_ptr) + ((r45_row[:, None] * $(256)) + r45_col[None, :]), r45_value)
    r46_z = tl.arange($(32768))
    r46_value = (tl.load($(vselect_scatter_2_ptr) + ((((r46_z / $(128)) % $(256)) * $(128)) + (r46_z % $(128)) + $(32768)), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(32768)])
    tl.store($(vmul_1_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(131072))
    r47_value = tl.where((((r47_z / $(32768)) % $(4)) == $(1)), tl.load($(vmul_1_ptr) + ((((r47_z / $(128)) % $(256)) * $(128)) + (r47_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_2_ptr) + ((((r47_z / $(32768)) % $(4)) * $(32768)) + (((r47_z / $(128)) % $(256)) * $(128)) + (r47_z % $(128))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(131072)])
    tl.store($(vselect_scatter_4_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(32768))
    r48_value = (tl.load($(vselect_scatter_4_ptr) + ((((r48_z / $(128)) % $(256)) * $(128)) + (r48_z % $(128)) + $(32768)), dtype=tl.bfloat16) + (tl.load($(vaddmm_19_ptr) + (((r48_z % $(128)) * $(256)) + ((r48_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(32768)])
    tl.store($(vadd_1_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(131072))
    r49_value = tl.where((((r49_z / $(32768)) % $(4)) == $(1)), tl.load($(vadd_1_ptr) + ((((r49_z / $(128)) % $(256)) * $(128)) + (r49_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_4_ptr) + ((((r49_z / $(32768)) % $(4)) * $(32768)) + (((r49_z / $(128)) % $(256)) * $(128)) + (r49_z % $(128))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(131072)])
    tl.store($(vselect_scatter_5_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(32768))
    r50_value = (tl.load($(vselect_scatter_3_ptr) + ((((r50_z / $(128)) % $(256)) * $(128)) + (r50_z % $(128)) + $(65536)), dtype=tl.bfloat16) + (tl.load($(input0_ptr) + ((((r50_z / $(128)) % $(256)) * $(128)) + (r50_z % $(128)) + $(65536)), dtype=tl.bfloat16) - tl.load($(vselect_scatter_3_ptr) + ((((r50_z / $(128)) % $(256)) * $(128)) + (r50_z % $(128)) + $(65536)), dtype=tl.bfloat16)) * 0.050000000000000044)
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(32768)])
    tl.store($(vlerp_4_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(131072))
    r51_value = tl.where((((r51_z / $(32768)) % $(4)) == $(2)), tl.load($(vlerp_4_ptr) + ((((r51_z / $(128)) % $(256)) * $(128)) + (r51_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_3_ptr) + ((((r51_z / $(32768)) % $(4)) * $(32768)) + (((r51_z / $(128)) % $(256)) * $(128)) + (r51_z % $(128))), dtype=tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(131072)])
    tl.store($(vselect_scatter_6_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(32768))
    r52_value = (tl.load($(input0_ptr) + ((((r52_z / $(128)) % $(256)) * $(128)) + (r52_z % $(128)) + $(65536)), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_6_ptr) + ((((r52_z / $(128)) % $(256)) * $(128)) + (r52_z % $(128)) + $(65536)), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r52_z / $(128)) % $(256)) * $(128)) + (r52_z % $(128)) + $(65536)), dtype=tl.bfloat16)) * 0.95)
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(32768)])
    tl.store($(vlerp_5_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(1))
    r53_lane = tl.arange($(32768))
    r53_value = tl.sqrt(tl.sum(tl.load($(vlerp_5_ptr) + (((r53_lane[None, :] % $(256)) * $(128)) + ((r53_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16) * tl.load($(vlerp_5_ptr) + (((r53_lane[None, :] % $(256)) * $(128)) + ((r53_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16), axis=1))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(1)])
    tl.store($(vlinalg_vector_norm_2_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(1))
    r54_value = tl.maximum(tl.load($(vlinalg_vector_norm_2_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(1)])
    tl.store($(vclamp_2_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(32768))
    r55_value = (tl.load($(vlerp_5_ptr) + (((r55_z % $(256)) * $(128)) + ((r55_z / $(256)) % $(128))), dtype=tl.bfloat16) / tl.load($(vclamp_2_ptr) + $(0), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(32768)])
    tl.store($(vdiv_2_ptr) + r55_z, r55_value)
    r56_row = tl.arange($(128))
    r56_col = tl.arange($(128))
    r56_lane = tl.arange($(256))
    r56_left = tl.load($(vdiv_2_ptr) + (((r56_row[:, None] % $(128)) * $(256)) + (r56_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r56_right = tl.load($(vdiv_2_ptr) + (((r56_col[None, :] % $(128)) * $(256)) + (r56_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r56_value = tl.dot(r56_left, r56_right)
    r56_value = r56_value.to(tl.bfloat16)
    tl.store($(vmm_10_ptr) + ((r56_row[:, None] * $(128)) + r56_col[None, :]), r56_value)
    r57_row = tl.arange($(128))
    r57_col = tl.arange($(128))
    r57_lane = tl.arange($(128))
    r57_left = tl.load($(vmm_10_ptr) + (((r57_row[:, None] % $(128)) * $(128)) + (r57_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r57_right = tl.load($(vmm_10_ptr) + (((r57_lane[:, None] % $(128)) * $(128)) + (r57_col[None, :] % $(128))), dtype=tl.bfloat16)
    r57_value = tl.dot(r57_left, r57_right)
    r57_value = r57_value * 2.0315
    r57_value = r57_value + tl.load($(vmm_10_ptr) + (((r57_row[:, None] % $(128)) * $(128)) + (r57_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r57_value = r57_value.to(tl.bfloat16)
    tl.store($(vaddmm_20_ptr) + ((r57_row[:, None] * $(128)) + r57_col[None, :]), r57_value)
    r58_row = tl.arange($(128))
    r58_col = tl.arange($(256))
    r58_lane = tl.arange($(128))
    r58_left = tl.load($(vaddmm_20_ptr) + (((r58_row[:, None] % $(128)) * $(128)) + (r58_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r58_right = tl.load($(vdiv_2_ptr) + (((r58_lane[:, None] % $(128)) * $(256)) + (r58_col[None, :] % $(256))), dtype=tl.bfloat16)
    r58_value = tl.dot(r58_left, r58_right)
    r58_value = r58_value + tl.load($(vdiv_2_ptr) + (((r58_row[:, None] % $(128)) * $(256)) + (r58_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r58_value = r58_value.to(tl.bfloat16)
    tl.store($(vaddmm_21_ptr) + ((r58_row[:, None] * $(256)) + r58_col[None, :]), r58_value)
    r59_row = tl.arange($(128))
    r59_col = tl.arange($(128))
    r59_lane = tl.arange($(256))
    r59_left = tl.load($(vaddmm_21_ptr) + (((r59_row[:, None] % $(128)) * $(256)) + (r59_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r59_right = tl.load($(vaddmm_21_ptr) + (((r59_col[None, :] % $(128)) * $(256)) + (r59_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r59_value = tl.dot(r59_left, r59_right)
    r59_value = r59_value.to(tl.bfloat16)
    tl.store($(vmm_11_ptr) + ((r59_row[:, None] * $(128)) + r59_col[None, :]), r59_value)
    r60_row = tl.arange($(128))
    r60_col = tl.arange($(128))
    r60_lane = tl.arange($(128))
    r60_left = tl.load($(vmm_11_ptr) + (((r60_row[:, None] % $(128)) * $(128)) + (r60_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r60_right = tl.load($(vmm_11_ptr) + (((r60_lane[:, None] % $(128)) * $(128)) + (r60_col[None, :] % $(128))), dtype=tl.bfloat16)
    r60_value = tl.dot(r60_left, r60_right)
    r60_value = r60_value * 2.0315
    r60_value = r60_value + tl.load($(vmm_11_ptr) + (((r60_row[:, None] % $(128)) * $(128)) + (r60_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r60_value = r60_value.to(tl.bfloat16)
    tl.store($(vaddmm_22_ptr) + ((r60_row[:, None] * $(128)) + r60_col[None, :]), r60_value)
    r61_row = tl.arange($(128))
    r61_col = tl.arange($(256))
    r61_lane = tl.arange($(128))
    r61_left = tl.load($(vaddmm_22_ptr) + (((r61_row[:, None] % $(128)) * $(128)) + (r61_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r61_right = tl.load($(vaddmm_21_ptr) + (((r61_lane[:, None] % $(128)) * $(256)) + (r61_col[None, :] % $(256))), dtype=tl.bfloat16)
    r61_value = tl.dot(r61_left, r61_right)
    r61_value = r61_value + tl.load($(vaddmm_21_ptr) + (((r61_row[:, None] % $(128)) * $(256)) + (r61_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r61_value = r61_value.to(tl.bfloat16)
    tl.store($(vaddmm_23_ptr) + ((r61_row[:, None] * $(256)) + r61_col[None, :]), r61_value)
    r62_row = tl.arange($(128))
    r62_col = tl.arange($(128))
    r62_lane = tl.arange($(256))
    r62_left = tl.load($(vaddmm_23_ptr) + (((r62_row[:, None] % $(128)) * $(256)) + (r62_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r62_right = tl.load($(vaddmm_23_ptr) + (((r62_col[None, :] % $(128)) * $(256)) + (r62_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r62_value = tl.dot(r62_left, r62_right)
    r62_value = r62_value.to(tl.bfloat16)
    tl.store($(vmm_12_ptr) + ((r62_row[:, None] * $(128)) + r62_col[None, :]), r62_value)
    r63_row = tl.arange($(128))
    r63_col = tl.arange($(128))
    r63_lane = tl.arange($(128))
    r63_left = tl.load($(vmm_12_ptr) + (((r63_row[:, None] % $(128)) * $(128)) + (r63_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r63_right = tl.load($(vmm_12_ptr) + (((r63_lane[:, None] % $(128)) * $(128)) + (r63_col[None, :] % $(128))), dtype=tl.bfloat16)
    r63_value = tl.dot(r63_left, r63_right)
    r63_value = r63_value * 2.0315
    r63_value = r63_value + tl.load($(vmm_12_ptr) + (((r63_row[:, None] % $(128)) * $(128)) + (r63_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r63_value = r63_value.to(tl.bfloat16)
    tl.store($(vaddmm_24_ptr) + ((r63_row[:, None] * $(128)) + r63_col[None, :]), r63_value)
    r64_row = tl.arange($(128))
    r64_col = tl.arange($(256))
    r64_lane = tl.arange($(128))
    r64_left = tl.load($(vaddmm_24_ptr) + (((r64_row[:, None] % $(128)) * $(128)) + (r64_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r64_right = tl.load($(vaddmm_23_ptr) + (((r64_lane[:, None] % $(128)) * $(256)) + (r64_col[None, :] % $(256))), dtype=tl.bfloat16)
    r64_value = tl.dot(r64_left, r64_right)
    r64_value = r64_value + tl.load($(vaddmm_23_ptr) + (((r64_row[:, None] % $(128)) * $(256)) + (r64_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r64_value = r64_value.to(tl.bfloat16)
    tl.store($(vaddmm_25_ptr) + ((r64_row[:, None] * $(256)) + r64_col[None, :]), r64_value)
    r65_row = tl.arange($(128))
    r65_col = tl.arange($(128))
    r65_lane = tl.arange($(256))
    r65_left = tl.load($(vaddmm_25_ptr) + (((r65_row[:, None] % $(128)) * $(256)) + (r65_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r65_right = tl.load($(vaddmm_25_ptr) + (((r65_col[None, :] % $(128)) * $(256)) + (r65_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r65_value = tl.dot(r65_left, r65_right)
    r65_value = r65_value.to(tl.bfloat16)
    tl.store($(vmm_13_ptr) + ((r65_row[:, None] * $(128)) + r65_col[None, :]), r65_value)
    r66_row = tl.arange($(128))
    r66_col = tl.arange($(128))
    r66_lane = tl.arange($(128))
    r66_left = tl.load($(vmm_13_ptr) + (((r66_row[:, None] % $(128)) * $(128)) + (r66_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r66_right = tl.load($(vmm_13_ptr) + (((r66_lane[:, None] % $(128)) * $(128)) + (r66_col[None, :] % $(128))), dtype=tl.bfloat16)
    r66_value = tl.dot(r66_left, r66_right)
    r66_value = r66_value * 2.0315
    r66_value = r66_value + tl.load($(vmm_13_ptr) + (((r66_row[:, None] % $(128)) * $(128)) + (r66_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r66_value = r66_value.to(tl.bfloat16)
    tl.store($(vaddmm_26_ptr) + ((r66_row[:, None] * $(128)) + r66_col[None, :]), r66_value)
    r67_row = tl.arange($(128))
    r67_col = tl.arange($(256))
    r67_lane = tl.arange($(128))
    r67_left = tl.load($(vaddmm_26_ptr) + (((r67_row[:, None] % $(128)) * $(128)) + (r67_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r67_right = tl.load($(vaddmm_25_ptr) + (((r67_lane[:, None] % $(128)) * $(256)) + (r67_col[None, :] % $(256))), dtype=tl.bfloat16)
    r67_value = tl.dot(r67_left, r67_right)
    r67_value = r67_value + tl.load($(vaddmm_25_ptr) + (((r67_row[:, None] % $(128)) * $(256)) + (r67_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r67_value = r67_value.to(tl.bfloat16)
    tl.store($(vaddmm_27_ptr) + ((r67_row[:, None] * $(256)) + r67_col[None, :]), r67_value)
    r68_row = tl.arange($(128))
    r68_col = tl.arange($(128))
    r68_lane = tl.arange($(256))
    r68_left = tl.load($(vaddmm_27_ptr) + (((r68_row[:, None] % $(128)) * $(256)) + (r68_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r68_right = tl.load($(vaddmm_27_ptr) + (((r68_col[None, :] % $(128)) * $(256)) + (r68_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r68_value = tl.dot(r68_left, r68_right)
    r68_value = r68_value.to(tl.bfloat16)
    tl.store($(vmm_14_ptr) + ((r68_row[:, None] * $(128)) + r68_col[None, :]), r68_value)
    r69_row = tl.arange($(128))
    r69_col = tl.arange($(128))
    r69_lane = tl.arange($(128))
    r69_left = tl.load($(vmm_14_ptr) + (((r69_row[:, None] % $(128)) * $(128)) + (r69_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r69_right = tl.load($(vmm_14_ptr) + (((r69_lane[:, None] % $(128)) * $(128)) + (r69_col[None, :] % $(128))), dtype=tl.bfloat16)
    r69_value = tl.dot(r69_left, r69_right)
    r69_value = r69_value * 2.0315
    r69_value = r69_value + tl.load($(vmm_14_ptr) + (((r69_row[:, None] % $(128)) * $(128)) + (r69_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r69_value = r69_value.to(tl.bfloat16)
    tl.store($(vaddmm_28_ptr) + ((r69_row[:, None] * $(128)) + r69_col[None, :]), r69_value)
    r70_row = tl.arange($(128))
    r70_col = tl.arange($(256))
    r70_lane = tl.arange($(128))
    r70_left = tl.load($(vaddmm_28_ptr) + (((r70_row[:, None] % $(128)) * $(128)) + (r70_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r70_right = tl.load($(vaddmm_27_ptr) + (((r70_lane[:, None] % $(128)) * $(256)) + (r70_col[None, :] % $(256))), dtype=tl.bfloat16)
    r70_value = tl.dot(r70_left, r70_right)
    r70_value = r70_value + tl.load($(vaddmm_27_ptr) + (((r70_row[:, None] % $(128)) * $(256)) + (r70_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r70_value = r70_value.to(tl.bfloat16)
    tl.store($(vaddmm_29_ptr) + ((r70_row[:, None] * $(256)) + r70_col[None, :]), r70_value)
    r71_z = tl.arange($(32768))
    r71_value = (tl.load($(vselect_scatter_5_ptr) + ((((r71_z / $(128)) % $(256)) * $(128)) + (r71_z % $(128)) + $(65536)), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r71_value = r71_value.to(tl.bfloat16)
    r71_value = tl.broadcast_to(r71_value, [$(32768)])
    tl.store($(vmul_2_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(131072))
    r72_value = tl.where((((r72_z / $(32768)) % $(4)) == $(2)), tl.load($(vmul_2_ptr) + ((((r72_z / $(128)) % $(256)) * $(128)) + (r72_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_5_ptr) + ((((r72_z / $(32768)) % $(4)) * $(32768)) + (((r72_z / $(128)) % $(256)) * $(128)) + (r72_z % $(128))), dtype=tl.bfloat16))
    r72_value = r72_value.to(tl.bfloat16)
    r72_value = tl.broadcast_to(r72_value, [$(131072)])
    tl.store($(vselect_scatter_7_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(32768))
    r73_value = (tl.load($(vselect_scatter_7_ptr) + ((((r73_z / $(128)) % $(256)) * $(128)) + (r73_z % $(128)) + $(65536)), dtype=tl.bfloat16) + (tl.load($(vaddmm_29_ptr) + (((r73_z % $(128)) * $(256)) + ((r73_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(32768)])
    tl.store($(vadd_2_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(131072))
    r74_value = tl.where((((r74_z / $(32768)) % $(4)) == $(2)), tl.load($(vadd_2_ptr) + ((((r74_z / $(128)) % $(256)) * $(128)) + (r74_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_7_ptr) + ((((r74_z / $(32768)) % $(4)) * $(32768)) + (((r74_z / $(128)) % $(256)) * $(128)) + (r74_z % $(128))), dtype=tl.bfloat16))
    r74_value = r74_value.to(tl.bfloat16)
    r74_value = tl.broadcast_to(r74_value, [$(131072)])
    tl.store($(vselect_scatter_8_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(32768))
    r75_value = (tl.load($(vselect_scatter_6_ptr) + ((((r75_z / $(128)) % $(256)) * $(128)) + (r75_z % $(128)) + $(98304)), dtype=tl.bfloat16) + (tl.load($(input0_ptr) + ((((r75_z / $(128)) % $(256)) * $(128)) + (r75_z % $(128)) + $(98304)), dtype=tl.bfloat16) - tl.load($(vselect_scatter_6_ptr) + ((((r75_z / $(128)) % $(256)) * $(128)) + (r75_z % $(128)) + $(98304)), dtype=tl.bfloat16)) * 0.050000000000000044)
    r75_value = r75_value.to(tl.bfloat16)
    r75_value = tl.broadcast_to(r75_value, [$(32768)])
    tl.store($(vlerp_6_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(131072))
    r76_value = tl.where((((r76_z / $(32768)) % $(4)) == $(3)), tl.load($(vlerp_6_ptr) + ((((r76_z / $(128)) % $(256)) * $(128)) + (r76_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_6_ptr) + ((((r76_z / $(32768)) % $(4)) * $(32768)) + (((r76_z / $(128)) % $(256)) * $(128)) + (r76_z % $(128))), dtype=tl.bfloat16))
    r76_value = r76_value.to(tl.bfloat16)
    r76_value = tl.broadcast_to(r76_value, [$(131072)])
    tl.store($(vselect_scatter_9_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(32768))
    r77_value = (tl.load($(input0_ptr) + ((((r77_z / $(128)) % $(256)) * $(128)) + (r77_z % $(128)) + $(98304)), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_9_ptr) + ((((r77_z / $(128)) % $(256)) * $(128)) + (r77_z % $(128)) + $(98304)), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r77_z / $(128)) % $(256)) * $(128)) + (r77_z % $(128)) + $(98304)), dtype=tl.bfloat16)) * 0.95)
    r77_value = r77_value.to(tl.bfloat16)
    r77_value = tl.broadcast_to(r77_value, [$(32768)])
    tl.store($(vlerp_7_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(1))
    r78_lane = tl.arange($(32768))
    r78_value = tl.sqrt(tl.sum(tl.load($(vlerp_7_ptr) + (((r78_lane[None, :] % $(256)) * $(128)) + ((r78_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16) * tl.load($(vlerp_7_ptr) + (((r78_lane[None, :] % $(256)) * $(128)) + ((r78_lane[None, :] / $(256)) % $(128))), dtype=tl.bfloat16), axis=1))
    r78_value = r78_value.to(tl.bfloat16)
    r78_value = tl.broadcast_to(r78_value, [$(1)])
    tl.store($(vlinalg_vector_norm_3_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(1))
    r79_value = tl.maximum(tl.load($(vlinalg_vector_norm_3_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(1)])
    tl.store($(vclamp_3_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(32768))
    r80_value = (tl.load($(vlerp_7_ptr) + (((r80_z % $(256)) * $(128)) + ((r80_z / $(256)) % $(128))), dtype=tl.bfloat16) / tl.load($(vclamp_3_ptr) + $(0), dtype=tl.bfloat16))
    r80_value = r80_value.to(tl.bfloat16)
    r80_value = tl.broadcast_to(r80_value, [$(32768)])
    tl.store($(vdiv_3_ptr) + r80_z, r80_value)
    r81_row = tl.arange($(128))
    r81_col = tl.arange($(128))
    r81_lane = tl.arange($(256))
    r81_left = tl.load($(vdiv_3_ptr) + (((r81_row[:, None] % $(128)) * $(256)) + (r81_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r81_right = tl.load($(vdiv_3_ptr) + (((r81_col[None, :] % $(128)) * $(256)) + (r81_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r81_value = tl.dot(r81_left, r81_right)
    r81_value = r81_value.to(tl.bfloat16)
    tl.store($(vmm_15_ptr) + ((r81_row[:, None] * $(128)) + r81_col[None, :]), r81_value)
    r82_row = tl.arange($(128))
    r82_col = tl.arange($(128))
    r82_lane = tl.arange($(128))
    r82_left = tl.load($(vmm_15_ptr) + (((r82_row[:, None] % $(128)) * $(128)) + (r82_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r82_right = tl.load($(vmm_15_ptr) + (((r82_lane[:, None] % $(128)) * $(128)) + (r82_col[None, :] % $(128))), dtype=tl.bfloat16)
    r82_value = tl.dot(r82_left, r82_right)
    r82_value = r82_value * 2.0315
    r82_value = r82_value + tl.load($(vmm_15_ptr) + (((r82_row[:, None] % $(128)) * $(128)) + (r82_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r82_value = r82_value.to(tl.bfloat16)
    tl.store($(vaddmm_30_ptr) + ((r82_row[:, None] * $(128)) + r82_col[None, :]), r82_value)
    r83_row = tl.arange($(128))
    r83_col = tl.arange($(256))
    r83_lane = tl.arange($(128))
    r83_left = tl.load($(vaddmm_30_ptr) + (((r83_row[:, None] % $(128)) * $(128)) + (r83_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r83_right = tl.load($(vdiv_3_ptr) + (((r83_lane[:, None] % $(128)) * $(256)) + (r83_col[None, :] % $(256))), dtype=tl.bfloat16)
    r83_value = tl.dot(r83_left, r83_right)
    r83_value = r83_value + tl.load($(vdiv_3_ptr) + (((r83_row[:, None] % $(128)) * $(256)) + (r83_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r83_value = r83_value.to(tl.bfloat16)
    tl.store($(vaddmm_31_ptr) + ((r83_row[:, None] * $(256)) + r83_col[None, :]), r83_value)
    r84_row = tl.arange($(128))
    r84_col = tl.arange($(128))
    r84_lane = tl.arange($(256))
    r84_left = tl.load($(vaddmm_31_ptr) + (((r84_row[:, None] % $(128)) * $(256)) + (r84_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r84_right = tl.load($(vaddmm_31_ptr) + (((r84_col[None, :] % $(128)) * $(256)) + (r84_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r84_value = tl.dot(r84_left, r84_right)
    r84_value = r84_value.to(tl.bfloat16)
    tl.store($(vmm_16_ptr) + ((r84_row[:, None] * $(128)) + r84_col[None, :]), r84_value)
    r85_row = tl.arange($(128))
    r85_col = tl.arange($(128))
    r85_lane = tl.arange($(128))
    r85_left = tl.load($(vmm_16_ptr) + (((r85_row[:, None] % $(128)) * $(128)) + (r85_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r85_right = tl.load($(vmm_16_ptr) + (((r85_lane[:, None] % $(128)) * $(128)) + (r85_col[None, :] % $(128))), dtype=tl.bfloat16)
    r85_value = tl.dot(r85_left, r85_right)
    r85_value = r85_value * 2.0315
    r85_value = r85_value + tl.load($(vmm_16_ptr) + (((r85_row[:, None] % $(128)) * $(128)) + (r85_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r85_value = r85_value.to(tl.bfloat16)
    tl.store($(vaddmm_32_ptr) + ((r85_row[:, None] * $(128)) + r85_col[None, :]), r85_value)
    r86_row = tl.arange($(128))
    r86_col = tl.arange($(256))
    r86_lane = tl.arange($(128))
    r86_left = tl.load($(vaddmm_32_ptr) + (((r86_row[:, None] % $(128)) * $(128)) + (r86_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r86_right = tl.load($(vaddmm_31_ptr) + (((r86_lane[:, None] % $(128)) * $(256)) + (r86_col[None, :] % $(256))), dtype=tl.bfloat16)
    r86_value = tl.dot(r86_left, r86_right)
    r86_value = r86_value + tl.load($(vaddmm_31_ptr) + (((r86_row[:, None] % $(128)) * $(256)) + (r86_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r86_value = r86_value.to(tl.bfloat16)
    tl.store($(vaddmm_33_ptr) + ((r86_row[:, None] * $(256)) + r86_col[None, :]), r86_value)
    r87_row = tl.arange($(128))
    r87_col = tl.arange($(128))
    r87_lane = tl.arange($(256))
    r87_left = tl.load($(vaddmm_33_ptr) + (((r87_row[:, None] % $(128)) * $(256)) + (r87_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r87_right = tl.load($(vaddmm_33_ptr) + (((r87_col[None, :] % $(128)) * $(256)) + (r87_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r87_value = tl.dot(r87_left, r87_right)
    r87_value = r87_value.to(tl.bfloat16)
    tl.store($(vmm_17_ptr) + ((r87_row[:, None] * $(128)) + r87_col[None, :]), r87_value)
    r88_row = tl.arange($(128))
    r88_col = tl.arange($(128))
    r88_lane = tl.arange($(128))
    r88_left = tl.load($(vmm_17_ptr) + (((r88_row[:, None] % $(128)) * $(128)) + (r88_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r88_right = tl.load($(vmm_17_ptr) + (((r88_lane[:, None] % $(128)) * $(128)) + (r88_col[None, :] % $(128))), dtype=tl.bfloat16)
    r88_value = tl.dot(r88_left, r88_right)
    r88_value = r88_value * 2.0315
    r88_value = r88_value + tl.load($(vmm_17_ptr) + (((r88_row[:, None] % $(128)) * $(128)) + (r88_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r88_value = r88_value.to(tl.bfloat16)
    tl.store($(vaddmm_34_ptr) + ((r88_row[:, None] * $(128)) + r88_col[None, :]), r88_value)
    r89_row = tl.arange($(128))
    r89_col = tl.arange($(256))
    r89_lane = tl.arange($(128))
    r89_left = tl.load($(vaddmm_34_ptr) + (((r89_row[:, None] % $(128)) * $(128)) + (r89_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r89_right = tl.load($(vaddmm_33_ptr) + (((r89_lane[:, None] % $(128)) * $(256)) + (r89_col[None, :] % $(256))), dtype=tl.bfloat16)
    r89_value = tl.dot(r89_left, r89_right)
    r89_value = r89_value + tl.load($(vaddmm_33_ptr) + (((r89_row[:, None] % $(128)) * $(256)) + (r89_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r89_value = r89_value.to(tl.bfloat16)
    tl.store($(vaddmm_35_ptr) + ((r89_row[:, None] * $(256)) + r89_col[None, :]), r89_value)
    r90_row = tl.arange($(128))
    r90_col = tl.arange($(128))
    r90_lane = tl.arange($(256))
    r90_left = tl.load($(vaddmm_35_ptr) + (((r90_row[:, None] % $(128)) * $(256)) + (r90_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r90_right = tl.load($(vaddmm_35_ptr) + (((r90_col[None, :] % $(128)) * $(256)) + (r90_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r90_value = tl.dot(r90_left, r90_right)
    r90_value = r90_value.to(tl.bfloat16)
    tl.store($(vmm_18_ptr) + ((r90_row[:, None] * $(128)) + r90_col[None, :]), r90_value)
    r91_row = tl.arange($(128))
    r91_col = tl.arange($(128))
    r91_lane = tl.arange($(128))
    r91_left = tl.load($(vmm_18_ptr) + (((r91_row[:, None] % $(128)) * $(128)) + (r91_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r91_right = tl.load($(vmm_18_ptr) + (((r91_lane[:, None] % $(128)) * $(128)) + (r91_col[None, :] % $(128))), dtype=tl.bfloat16)
    r91_value = tl.dot(r91_left, r91_right)
    r91_value = r91_value * 2.0315
    r91_value = r91_value + tl.load($(vmm_18_ptr) + (((r91_row[:, None] % $(128)) * $(128)) + (r91_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r91_value = r91_value.to(tl.bfloat16)
    tl.store($(vaddmm_36_ptr) + ((r91_row[:, None] * $(128)) + r91_col[None, :]), r91_value)
    r92_row = tl.arange($(128))
    r92_col = tl.arange($(256))
    r92_lane = tl.arange($(128))
    r92_left = tl.load($(vaddmm_36_ptr) + (((r92_row[:, None] % $(128)) * $(128)) + (r92_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r92_right = tl.load($(vaddmm_35_ptr) + (((r92_lane[:, None] % $(128)) * $(256)) + (r92_col[None, :] % $(256))), dtype=tl.bfloat16)
    r92_value = tl.dot(r92_left, r92_right)
    r92_value = r92_value + tl.load($(vaddmm_35_ptr) + (((r92_row[:, None] % $(128)) * $(256)) + (r92_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r92_value = r92_value.to(tl.bfloat16)
    tl.store($(vaddmm_37_ptr) + ((r92_row[:, None] * $(256)) + r92_col[None, :]), r92_value)
    r93_row = tl.arange($(128))
    r93_col = tl.arange($(128))
    r93_lane = tl.arange($(256))
    r93_left = tl.load($(vaddmm_37_ptr) + (((r93_row[:, None] % $(128)) * $(256)) + (r93_lane[None, :] % $(256))), dtype=tl.bfloat16)
    r93_right = tl.load($(vaddmm_37_ptr) + (((r93_col[None, :] % $(128)) * $(256)) + (r93_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r93_value = tl.dot(r93_left, r93_right)
    r93_value = r93_value.to(tl.bfloat16)
    tl.store($(vmm_19_ptr) + ((r93_row[:, None] * $(128)) + r93_col[None, :]), r93_value)
    r94_row = tl.arange($(128))
    r94_col = tl.arange($(128))
    r94_lane = tl.arange($(128))
    r94_left = tl.load($(vmm_19_ptr) + (((r94_row[:, None] % $(128)) * $(128)) + (r94_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r94_right = tl.load($(vmm_19_ptr) + (((r94_lane[:, None] % $(128)) * $(128)) + (r94_col[None, :] % $(128))), dtype=tl.bfloat16)
    r94_value = tl.dot(r94_left, r94_right)
    r94_value = r94_value * 2.0315
    r94_value = r94_value + tl.load($(vmm_19_ptr) + (((r94_row[:, None] % $(128)) * $(128)) + (r94_col[None, :] % $(128))), dtype=tl.bfloat16) * -4.775
    r94_value = r94_value.to(tl.bfloat16)
    tl.store($(vaddmm_38_ptr) + ((r94_row[:, None] * $(128)) + r94_col[None, :]), r94_value)
    r95_row = tl.arange($(128))
    r95_col = tl.arange($(256))
    r95_lane = tl.arange($(128))
    r95_left = tl.load($(vaddmm_38_ptr) + (((r95_row[:, None] % $(128)) * $(128)) + (r95_lane[None, :] % $(128))), dtype=tl.bfloat16)
    r95_right = tl.load($(vaddmm_37_ptr) + (((r95_lane[:, None] % $(128)) * $(256)) + (r95_col[None, :] % $(256))), dtype=tl.bfloat16)
    r95_value = tl.dot(r95_left, r95_right)
    r95_value = r95_value + tl.load($(vaddmm_37_ptr) + (((r95_row[:, None] % $(128)) * $(256)) + (r95_col[None, :] % $(256))), dtype=tl.bfloat16) * 3.4445
    r95_value = r95_value.to(tl.bfloat16)
    tl.store($(vaddmm_39_ptr) + ((r95_row[:, None] * $(256)) + r95_col[None, :]), r95_value)
    r96_z = tl.arange($(32768))
    r96_value = (tl.load($(vselect_scatter_8_ptr) + ((((r96_z / $(128)) % $(256)) * $(128)) + (r96_z % $(128)) + $(98304)), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r96_value = r96_value.to(tl.bfloat16)
    r96_value = tl.broadcast_to(r96_value, [$(32768)])
    tl.store($(vmul_3_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(131072))
    r97_value = tl.where((((r97_z / $(32768)) % $(4)) == $(3)), tl.load($(vmul_3_ptr) + ((((r97_z / $(128)) % $(256)) * $(128)) + (r97_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_8_ptr) + ((((r97_z / $(32768)) % $(4)) * $(32768)) + (((r97_z / $(128)) % $(256)) * $(128)) + (r97_z % $(128))), dtype=tl.bfloat16))
    r97_value = r97_value.to(tl.bfloat16)
    r97_value = tl.broadcast_to(r97_value, [$(131072)])
    tl.store($(vselect_scatter_10_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(32768))
    r98_value = (tl.load($(vselect_scatter_10_ptr) + ((((r98_z / $(128)) % $(256)) * $(128)) + (r98_z % $(128)) + $(98304)), dtype=tl.bfloat16) + (tl.load($(vaddmm_39_ptr) + (((r98_z % $(128)) * $(256)) + ((r98_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r98_value = r98_value.to(tl.bfloat16)
    r98_value = tl.broadcast_to(r98_value, [$(32768)])
    tl.store($(vadd_3_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(131072))
    r99_value = tl.where((((r99_z / $(32768)) % $(4)) == $(3)), tl.load($(vadd_3_ptr) + ((((r99_z / $(128)) % $(256)) * $(128)) + (r99_z % $(128))), dtype=tl.bfloat16), tl.load($(vselect_scatter_10_ptr) + ((((r99_z / $(32768)) % $(4)) * $(32768)) + (((r99_z / $(128)) % $(256)) * $(128)) + (r99_z % $(128))), dtype=tl.bfloat16))
    r99_value = r99_value.to(tl.bfloat16)
    r99_value = tl.broadcast_to(r99_value, [$(131072)])
    tl.store($(vselect_scatter_11_ptr) + r99_z, r99_value)
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
    r20_z = tl.arange($(131072))
    r20_value = (tl.load($(parameter0_ptr) + ((((r20_z / $(32768)) % $(4)) * $(32768)) + (((r20_z / $(128)) % $(256)) * $(128)) + (r20_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r20_value = r20_value.to(tl.bfloat16)
    r20_value = tl.broadcast_to(r20_value, [$(131072)])
    tl.store($(candidate_vmul_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(131072))
    r21_value = (tl.load($(candidate_vmul_ptr) + ((((r21_z / $(32768)) % $(4)) * $(32768)) + (((r21_z / $(128)) % $(256)) * $(128)) + (r21_z % $(128))), dtype=tl.bfloat16) + (tl.load($(candidate_vbaddbmm_9_ptr) + ((((r21_z / $(32768)) % $(4)) * $(32768)) + ((r21_z % $(128)) * $(256)) + ((r21_z / $(128)) % $(256))), dtype=tl.bfloat16) * -0.032))
    r21_value = r21_value.to(tl.bfloat16)
    r21_value = tl.broadcast_to(r21_value, [$(131072)])
    tl.store($(candidate_vadd_ptr) + r21_z, r21_value)
    z = tl.arange($(131072))
    value = tl.load($(candidate_vadd_ptr) + ((((z / $(32768)) % $(4)) * $(32768)) + (((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_muon_bmm_vs_loop
