import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_muon_fused_vs_loop
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
  triton {
    r0_z = tl.arange($(35904))
    r0_value = (tl.cast(0.0, tl.bfloat16) + (tl.load($(input0_ptr) + ((((r0_z / $(136)) % $(264)) * $(136)) + (r0_z % $(136))), dtype=tl.bfloat16) - tl.cast(0.0, tl.bfloat16)) * 0.050000000000000044)
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(35904)])
    tl.store($(vlerp_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(107712))
    r1_value = tl.where((((r1_z / $(35904)) % $(3)) == $(0)), tl.load($(vlerp_ptr) + ((((r1_z / $(136)) % $(264)) * $(136)) + (r1_z % $(136))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(107712)])
    tl.store($(vselect_scatter_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(35904))
    r2_value = (tl.load($(input0_ptr) + ((((r2_z / $(136)) % $(264)) * $(136)) + (r2_z % $(136))), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_ptr) + ((((r2_z / $(136)) % $(264)) * $(136)) + (r2_z % $(136))), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r2_z / $(136)) % $(264)) * $(136)) + (r2_z % $(136))), dtype=tl.bfloat16)) * 0.95)
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(35904)])
    tl.store($(vlerp_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1))
    r3_lane = tl.arange($(35904))
    r3_value = tl.sqrt(tl.sum(tl.load($(vlerp_1_ptr) + (((r3_lane[None, :] % $(264)) * $(136)) + ((r3_lane[None, :] / $(264)) % $(136))), dtype=tl.bfloat16) * tl.load($(vlerp_1_ptr) + (((r3_lane[None, :] % $(264)) * $(136)) + ((r3_lane[None, :] / $(264)) % $(136))), dtype=tl.bfloat16), axis=1))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(1)])
    tl.store($(vlinalg_vector_norm_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(1))
    r4_value = tl.maximum(tl.load($(vlinalg_vector_norm_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(1)])
    tl.store($(vclamp_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(35904))
    r5_value = (tl.load($(vlerp_1_ptr) + (((r5_z % $(264)) * $(136)) + ((r5_z / $(264)) % $(136))), dtype=tl.bfloat16) / tl.load($(vclamp_ptr) + $(0), dtype=tl.bfloat16))
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(35904)])
    tl.store($(vdiv_ptr) + r5_z, r5_value)
    r6_row = tl.arange($(136))
    r6_col = tl.arange($(136))
    r6_lane = tl.arange($(264))
    r6_left = tl.load($(vdiv_ptr) + (((r6_row[:, None] % $(136)) * $(264)) + (r6_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r6_right = tl.load($(vdiv_ptr) + (((r6_col[None, :] % $(136)) * $(264)) + (r6_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r6_value = tl.dot(r6_left, r6_right)
    r6_value = r6_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r6_row[:, None] * $(136)) + r6_col[None, :]), r6_value)
    r7_row = tl.arange($(136))
    r7_col = tl.arange($(136))
    r7_lane = tl.arange($(136))
    r7_left = tl.load($(vmm_ptr) + (((r7_row[:, None] % $(136)) * $(136)) + (r7_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r7_right = tl.load($(vmm_ptr) + (((r7_lane[:, None] % $(136)) * $(136)) + (r7_col[None, :] % $(136))), dtype=tl.bfloat16)
    r7_value = tl.dot(r7_left, r7_right)
    r7_value = r7_value * 2.0315
    r7_value = r7_value + tl.load($(vmm_ptr) + (((r7_row[:, None] % $(136)) * $(136)) + (r7_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r7_value = r7_value.to(tl.bfloat16)
    tl.store($(vaddmm_ptr) + ((r7_row[:, None] * $(136)) + r7_col[None, :]), r7_value)
    r8_row = tl.arange($(136))
    r8_col = tl.arange($(264))
    r8_lane = tl.arange($(136))
    r8_left = tl.load($(vaddmm_ptr) + (((r8_row[:, None] % $(136)) * $(136)) + (r8_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r8_right = tl.load($(vdiv_ptr) + (((r8_lane[:, None] % $(136)) * $(264)) + (r8_col[None, :] % $(264))), dtype=tl.bfloat16)
    r8_value = tl.dot(r8_left, r8_right)
    r8_value = r8_value + tl.load($(vdiv_ptr) + (((r8_row[:, None] % $(136)) * $(264)) + (r8_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r8_value = r8_value.to(tl.bfloat16)
    tl.store($(vaddmm_1_ptr) + ((r8_row[:, None] * $(264)) + r8_col[None, :]), r8_value)
    r9_row = tl.arange($(136))
    r9_col = tl.arange($(136))
    r9_lane = tl.arange($(264))
    r9_left = tl.load($(vaddmm_1_ptr) + (((r9_row[:, None] % $(136)) * $(264)) + (r9_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r9_right = tl.load($(vaddmm_1_ptr) + (((r9_col[None, :] % $(136)) * $(264)) + (r9_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r9_value = tl.dot(r9_left, r9_right)
    r9_value = r9_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r9_row[:, None] * $(136)) + r9_col[None, :]), r9_value)
    r10_row = tl.arange($(136))
    r10_col = tl.arange($(136))
    r10_lane = tl.arange($(136))
    r10_left = tl.load($(vmm_1_ptr) + (((r10_row[:, None] % $(136)) * $(136)) + (r10_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r10_right = tl.load($(vmm_1_ptr) + (((r10_lane[:, None] % $(136)) * $(136)) + (r10_col[None, :] % $(136))), dtype=tl.bfloat16)
    r10_value = tl.dot(r10_left, r10_right)
    r10_value = r10_value * 2.0315
    r10_value = r10_value + tl.load($(vmm_1_ptr) + (((r10_row[:, None] % $(136)) * $(136)) + (r10_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r10_value = r10_value.to(tl.bfloat16)
    tl.store($(vaddmm_2_ptr) + ((r10_row[:, None] * $(136)) + r10_col[None, :]), r10_value)
    r11_row = tl.arange($(136))
    r11_col = tl.arange($(264))
    r11_lane = tl.arange($(136))
    r11_left = tl.load($(vaddmm_2_ptr) + (((r11_row[:, None] % $(136)) * $(136)) + (r11_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r11_right = tl.load($(vaddmm_1_ptr) + (((r11_lane[:, None] % $(136)) * $(264)) + (r11_col[None, :] % $(264))), dtype=tl.bfloat16)
    r11_value = tl.dot(r11_left, r11_right)
    r11_value = r11_value + tl.load($(vaddmm_1_ptr) + (((r11_row[:, None] % $(136)) * $(264)) + (r11_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r11_value = r11_value.to(tl.bfloat16)
    tl.store($(vaddmm_3_ptr) + ((r11_row[:, None] * $(264)) + r11_col[None, :]), r11_value)
    r12_row = tl.arange($(136))
    r12_col = tl.arange($(136))
    r12_lane = tl.arange($(264))
    r12_left = tl.load($(vaddmm_3_ptr) + (((r12_row[:, None] % $(136)) * $(264)) + (r12_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r12_right = tl.load($(vaddmm_3_ptr) + (((r12_col[None, :] % $(136)) * $(264)) + (r12_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r12_value = tl.dot(r12_left, r12_right)
    r12_value = r12_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r12_row[:, None] * $(136)) + r12_col[None, :]), r12_value)
    r13_row = tl.arange($(136))
    r13_col = tl.arange($(136))
    r13_lane = tl.arange($(136))
    r13_left = tl.load($(vmm_2_ptr) + (((r13_row[:, None] % $(136)) * $(136)) + (r13_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r13_right = tl.load($(vmm_2_ptr) + (((r13_lane[:, None] % $(136)) * $(136)) + (r13_col[None, :] % $(136))), dtype=tl.bfloat16)
    r13_value = tl.dot(r13_left, r13_right)
    r13_value = r13_value * 2.0315
    r13_value = r13_value + tl.load($(vmm_2_ptr) + (((r13_row[:, None] % $(136)) * $(136)) + (r13_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r13_value = r13_value.to(tl.bfloat16)
    tl.store($(vaddmm_4_ptr) + ((r13_row[:, None] * $(136)) + r13_col[None, :]), r13_value)
    r14_row = tl.arange($(136))
    r14_col = tl.arange($(264))
    r14_lane = tl.arange($(136))
    r14_left = tl.load($(vaddmm_4_ptr) + (((r14_row[:, None] % $(136)) * $(136)) + (r14_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r14_right = tl.load($(vaddmm_3_ptr) + (((r14_lane[:, None] % $(136)) * $(264)) + (r14_col[None, :] % $(264))), dtype=tl.bfloat16)
    r14_value = tl.dot(r14_left, r14_right)
    r14_value = r14_value + tl.load($(vaddmm_3_ptr) + (((r14_row[:, None] % $(136)) * $(264)) + (r14_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r14_value = r14_value.to(tl.bfloat16)
    tl.store($(vaddmm_5_ptr) + ((r14_row[:, None] * $(264)) + r14_col[None, :]), r14_value)
    r15_row = tl.arange($(136))
    r15_col = tl.arange($(136))
    r15_lane = tl.arange($(264))
    r15_left = tl.load($(vaddmm_5_ptr) + (((r15_row[:, None] % $(136)) * $(264)) + (r15_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r15_right = tl.load($(vaddmm_5_ptr) + (((r15_col[None, :] % $(136)) * $(264)) + (r15_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r15_value = tl.dot(r15_left, r15_right)
    r15_value = r15_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r15_row[:, None] * $(136)) + r15_col[None, :]), r15_value)
    r16_row = tl.arange($(136))
    r16_col = tl.arange($(136))
    r16_lane = tl.arange($(136))
    r16_left = tl.load($(vmm_3_ptr) + (((r16_row[:, None] % $(136)) * $(136)) + (r16_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r16_right = tl.load($(vmm_3_ptr) + (((r16_lane[:, None] % $(136)) * $(136)) + (r16_col[None, :] % $(136))), dtype=tl.bfloat16)
    r16_value = tl.dot(r16_left, r16_right)
    r16_value = r16_value * 2.0315
    r16_value = r16_value + tl.load($(vmm_3_ptr) + (((r16_row[:, None] % $(136)) * $(136)) + (r16_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r16_value = r16_value.to(tl.bfloat16)
    tl.store($(vaddmm_6_ptr) + ((r16_row[:, None] * $(136)) + r16_col[None, :]), r16_value)
    r17_row = tl.arange($(136))
    r17_col = tl.arange($(264))
    r17_lane = tl.arange($(136))
    r17_left = tl.load($(vaddmm_6_ptr) + (((r17_row[:, None] % $(136)) * $(136)) + (r17_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r17_right = tl.load($(vaddmm_5_ptr) + (((r17_lane[:, None] % $(136)) * $(264)) + (r17_col[None, :] % $(264))), dtype=tl.bfloat16)
    r17_value = tl.dot(r17_left, r17_right)
    r17_value = r17_value + tl.load($(vaddmm_5_ptr) + (((r17_row[:, None] % $(136)) * $(264)) + (r17_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r17_value = r17_value.to(tl.bfloat16)
    tl.store($(vaddmm_7_ptr) + ((r17_row[:, None] * $(264)) + r17_col[None, :]), r17_value)
    r18_row = tl.arange($(136))
    r18_col = tl.arange($(136))
    r18_lane = tl.arange($(264))
    r18_left = tl.load($(vaddmm_7_ptr) + (((r18_row[:, None] % $(136)) * $(264)) + (r18_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r18_right = tl.load($(vaddmm_7_ptr) + (((r18_col[None, :] % $(136)) * $(264)) + (r18_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r18_value = tl.dot(r18_left, r18_right)
    r18_value = r18_value.to(tl.bfloat16)
    tl.store($(vmm_4_ptr) + ((r18_row[:, None] * $(136)) + r18_col[None, :]), r18_value)
    r19_row = tl.arange($(136))
    r19_col = tl.arange($(136))
    r19_lane = tl.arange($(136))
    r19_left = tl.load($(vmm_4_ptr) + (((r19_row[:, None] % $(136)) * $(136)) + (r19_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r19_right = tl.load($(vmm_4_ptr) + (((r19_lane[:, None] % $(136)) * $(136)) + (r19_col[None, :] % $(136))), dtype=tl.bfloat16)
    r19_value = tl.dot(r19_left, r19_right)
    r19_value = r19_value * 2.0315
    r19_value = r19_value + tl.load($(vmm_4_ptr) + (((r19_row[:, None] % $(136)) * $(136)) + (r19_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r19_value = r19_value.to(tl.bfloat16)
    tl.store($(vaddmm_8_ptr) + ((r19_row[:, None] * $(136)) + r19_col[None, :]), r19_value)
    r20_row = tl.arange($(136))
    r20_col = tl.arange($(264))
    r20_lane = tl.arange($(136))
    r20_left = tl.load($(vaddmm_8_ptr) + (((r20_row[:, None] % $(136)) * $(136)) + (r20_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r20_right = tl.load($(vaddmm_7_ptr) + (((r20_lane[:, None] % $(136)) * $(264)) + (r20_col[None, :] % $(264))), dtype=tl.bfloat16)
    r20_value = tl.dot(r20_left, r20_right)
    r20_value = r20_value + tl.load($(vaddmm_7_ptr) + (((r20_row[:, None] % $(136)) * $(264)) + (r20_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r20_value = r20_value.to(tl.bfloat16)
    tl.store($(vaddmm_9_ptr) + ((r20_row[:, None] * $(264)) + r20_col[None, :]), r20_value)
    r21_z = tl.arange($(35904))
    r21_value = (tl.load($(parameter0_ptr) + ((((r21_z / $(136)) % $(264)) * $(136)) + (r21_z % $(136))), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r21_value = r21_value.to(tl.bfloat16)
    r21_value = tl.broadcast_to(r21_value, [$(35904)])
    tl.store($(vmul_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(107712))
    r22_value = tl.where((((r22_z / $(35904)) % $(3)) == $(0)), tl.load($(vmul_ptr) + ((((r22_z / $(136)) % $(264)) * $(136)) + (r22_z % $(136))), dtype=tl.bfloat16), tl.load($(parameter0_ptr) + ((((r22_z / $(35904)) % $(3)) * $(35904)) + (((r22_z / $(136)) % $(264)) * $(136)) + (r22_z % $(136))), dtype=tl.bfloat16))
    r22_value = r22_value.to(tl.bfloat16)
    r22_value = tl.broadcast_to(r22_value, [$(107712)])
    tl.store($(vselect_scatter_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(35904))
    r23_value = (tl.load($(vselect_scatter_1_ptr) + ((((r23_z / $(136)) % $(264)) * $(136)) + (r23_z % $(136))), dtype=tl.bfloat16) + (tl.load($(vaddmm_9_ptr) + (((r23_z % $(136)) * $(264)) + ((r23_z / $(136)) % $(264))), dtype=tl.bfloat16) * -0.03249615361854385))
    r23_value = r23_value.to(tl.bfloat16)
    r23_value = tl.broadcast_to(r23_value, [$(35904)])
    tl.store($(vadd_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(107712))
    r24_value = tl.where((((r24_z / $(35904)) % $(3)) == $(0)), tl.load($(vadd_ptr) + ((((r24_z / $(136)) % $(264)) * $(136)) + (r24_z % $(136))), dtype=tl.bfloat16), tl.load($(vselect_scatter_1_ptr) + ((((r24_z / $(35904)) % $(3)) * $(35904)) + (((r24_z / $(136)) % $(264)) * $(136)) + (r24_z % $(136))), dtype=tl.bfloat16))
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(107712)])
    tl.store($(vselect_scatter_2_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(35904))
    r25_value = (tl.load($(vselect_scatter_ptr) + ((((r25_z / $(136)) % $(264)) * $(136)) + (r25_z % $(136)) + $(35904)), dtype=tl.bfloat16) + (tl.load($(input0_ptr) + ((((r25_z / $(136)) % $(264)) * $(136)) + (r25_z % $(136)) + $(35904)), dtype=tl.bfloat16) - tl.load($(vselect_scatter_ptr) + ((((r25_z / $(136)) % $(264)) * $(136)) + (r25_z % $(136)) + $(35904)), dtype=tl.bfloat16)) * 0.050000000000000044)
    r25_value = r25_value.to(tl.bfloat16)
    r25_value = tl.broadcast_to(r25_value, [$(35904)])
    tl.store($(vlerp_2_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(107712))
    r26_value = tl.where((((r26_z / $(35904)) % $(3)) == $(1)), tl.load($(vlerp_2_ptr) + ((((r26_z / $(136)) % $(264)) * $(136)) + (r26_z % $(136))), dtype=tl.bfloat16), tl.load($(vselect_scatter_ptr) + ((((r26_z / $(35904)) % $(3)) * $(35904)) + (((r26_z / $(136)) % $(264)) * $(136)) + (r26_z % $(136))), dtype=tl.bfloat16))
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(107712)])
    tl.store($(vselect_scatter_3_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(35904))
    r27_value = (tl.load($(input0_ptr) + ((((r27_z / $(136)) % $(264)) * $(136)) + (r27_z % $(136)) + $(35904)), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_3_ptr) + ((((r27_z / $(136)) % $(264)) * $(136)) + (r27_z % $(136)) + $(35904)), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r27_z / $(136)) % $(264)) * $(136)) + (r27_z % $(136)) + $(35904)), dtype=tl.bfloat16)) * 0.95)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(35904)])
    tl.store($(vlerp_3_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(1))
    r28_lane = tl.arange($(35904))
    r28_value = tl.sqrt(tl.sum(tl.load($(vlerp_3_ptr) + (((r28_lane[None, :] % $(264)) * $(136)) + ((r28_lane[None, :] / $(264)) % $(136))), dtype=tl.bfloat16) * tl.load($(vlerp_3_ptr) + (((r28_lane[None, :] % $(264)) * $(136)) + ((r28_lane[None, :] / $(264)) % $(136))), dtype=tl.bfloat16), axis=1))
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(1)])
    tl.store($(vlinalg_vector_norm_1_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(1))
    r29_value = tl.maximum(tl.load($(vlinalg_vector_norm_1_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r29_value = r29_value.to(tl.bfloat16)
    r29_value = tl.broadcast_to(r29_value, [$(1)])
    tl.store($(vclamp_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(35904))
    r30_value = (tl.load($(vlerp_3_ptr) + (((r30_z % $(264)) * $(136)) + ((r30_z / $(264)) % $(136))), dtype=tl.bfloat16) / tl.load($(vclamp_1_ptr) + $(0), dtype=tl.bfloat16))
    r30_value = r30_value.to(tl.bfloat16)
    r30_value = tl.broadcast_to(r30_value, [$(35904)])
    tl.store($(vdiv_1_ptr) + r30_z, r30_value)
    r31_row = tl.arange($(136))
    r31_col = tl.arange($(136))
    r31_lane = tl.arange($(264))
    r31_left = tl.load($(vdiv_1_ptr) + (((r31_row[:, None] % $(136)) * $(264)) + (r31_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r31_right = tl.load($(vdiv_1_ptr) + (((r31_col[None, :] % $(136)) * $(264)) + (r31_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r31_value = tl.dot(r31_left, r31_right)
    r31_value = r31_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r31_row[:, None] * $(136)) + r31_col[None, :]), r31_value)
    r32_row = tl.arange($(136))
    r32_col = tl.arange($(136))
    r32_lane = tl.arange($(136))
    r32_left = tl.load($(vmm_5_ptr) + (((r32_row[:, None] % $(136)) * $(136)) + (r32_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r32_right = tl.load($(vmm_5_ptr) + (((r32_lane[:, None] % $(136)) * $(136)) + (r32_col[None, :] % $(136))), dtype=tl.bfloat16)
    r32_value = tl.dot(r32_left, r32_right)
    r32_value = r32_value * 2.0315
    r32_value = r32_value + tl.load($(vmm_5_ptr) + (((r32_row[:, None] % $(136)) * $(136)) + (r32_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r32_value = r32_value.to(tl.bfloat16)
    tl.store($(vaddmm_10_ptr) + ((r32_row[:, None] * $(136)) + r32_col[None, :]), r32_value)
    r33_row = tl.arange($(136))
    r33_col = tl.arange($(264))
    r33_lane = tl.arange($(136))
    r33_left = tl.load($(vaddmm_10_ptr) + (((r33_row[:, None] % $(136)) * $(136)) + (r33_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r33_right = tl.load($(vdiv_1_ptr) + (((r33_lane[:, None] % $(136)) * $(264)) + (r33_col[None, :] % $(264))), dtype=tl.bfloat16)
    r33_value = tl.dot(r33_left, r33_right)
    r33_value = r33_value + tl.load($(vdiv_1_ptr) + (((r33_row[:, None] % $(136)) * $(264)) + (r33_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r33_value = r33_value.to(tl.bfloat16)
    tl.store($(vaddmm_11_ptr) + ((r33_row[:, None] * $(264)) + r33_col[None, :]), r33_value)
    r34_row = tl.arange($(136))
    r34_col = tl.arange($(136))
    r34_lane = tl.arange($(264))
    r34_left = tl.load($(vaddmm_11_ptr) + (((r34_row[:, None] % $(136)) * $(264)) + (r34_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r34_right = tl.load($(vaddmm_11_ptr) + (((r34_col[None, :] % $(136)) * $(264)) + (r34_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r34_value = tl.dot(r34_left, r34_right)
    r34_value = r34_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r34_row[:, None] * $(136)) + r34_col[None, :]), r34_value)
    r35_row = tl.arange($(136))
    r35_col = tl.arange($(136))
    r35_lane = tl.arange($(136))
    r35_left = tl.load($(vmm_6_ptr) + (((r35_row[:, None] % $(136)) * $(136)) + (r35_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r35_right = tl.load($(vmm_6_ptr) + (((r35_lane[:, None] % $(136)) * $(136)) + (r35_col[None, :] % $(136))), dtype=tl.bfloat16)
    r35_value = tl.dot(r35_left, r35_right)
    r35_value = r35_value * 2.0315
    r35_value = r35_value + tl.load($(vmm_6_ptr) + (((r35_row[:, None] % $(136)) * $(136)) + (r35_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r35_value = r35_value.to(tl.bfloat16)
    tl.store($(vaddmm_12_ptr) + ((r35_row[:, None] * $(136)) + r35_col[None, :]), r35_value)
    r36_row = tl.arange($(136))
    r36_col = tl.arange($(264))
    r36_lane = tl.arange($(136))
    r36_left = tl.load($(vaddmm_12_ptr) + (((r36_row[:, None] % $(136)) * $(136)) + (r36_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r36_right = tl.load($(vaddmm_11_ptr) + (((r36_lane[:, None] % $(136)) * $(264)) + (r36_col[None, :] % $(264))), dtype=tl.bfloat16)
    r36_value = tl.dot(r36_left, r36_right)
    r36_value = r36_value + tl.load($(vaddmm_11_ptr) + (((r36_row[:, None] % $(136)) * $(264)) + (r36_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r36_value = r36_value.to(tl.bfloat16)
    tl.store($(vaddmm_13_ptr) + ((r36_row[:, None] * $(264)) + r36_col[None, :]), r36_value)
    r37_row = tl.arange($(136))
    r37_col = tl.arange($(136))
    r37_lane = tl.arange($(264))
    r37_left = tl.load($(vaddmm_13_ptr) + (((r37_row[:, None] % $(136)) * $(264)) + (r37_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r37_right = tl.load($(vaddmm_13_ptr) + (((r37_col[None, :] % $(136)) * $(264)) + (r37_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r37_value = tl.dot(r37_left, r37_right)
    r37_value = r37_value.to(tl.bfloat16)
    tl.store($(vmm_7_ptr) + ((r37_row[:, None] * $(136)) + r37_col[None, :]), r37_value)
    r38_row = tl.arange($(136))
    r38_col = tl.arange($(136))
    r38_lane = tl.arange($(136))
    r38_left = tl.load($(vmm_7_ptr) + (((r38_row[:, None] % $(136)) * $(136)) + (r38_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r38_right = tl.load($(vmm_7_ptr) + (((r38_lane[:, None] % $(136)) * $(136)) + (r38_col[None, :] % $(136))), dtype=tl.bfloat16)
    r38_value = tl.dot(r38_left, r38_right)
    r38_value = r38_value * 2.0315
    r38_value = r38_value + tl.load($(vmm_7_ptr) + (((r38_row[:, None] % $(136)) * $(136)) + (r38_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r38_value = r38_value.to(tl.bfloat16)
    tl.store($(vaddmm_14_ptr) + ((r38_row[:, None] * $(136)) + r38_col[None, :]), r38_value)
    r39_row = tl.arange($(136))
    r39_col = tl.arange($(264))
    r39_lane = tl.arange($(136))
    r39_left = tl.load($(vaddmm_14_ptr) + (((r39_row[:, None] % $(136)) * $(136)) + (r39_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r39_right = tl.load($(vaddmm_13_ptr) + (((r39_lane[:, None] % $(136)) * $(264)) + (r39_col[None, :] % $(264))), dtype=tl.bfloat16)
    r39_value = tl.dot(r39_left, r39_right)
    r39_value = r39_value + tl.load($(vaddmm_13_ptr) + (((r39_row[:, None] % $(136)) * $(264)) + (r39_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r39_value = r39_value.to(tl.bfloat16)
    tl.store($(vaddmm_15_ptr) + ((r39_row[:, None] * $(264)) + r39_col[None, :]), r39_value)
    r40_row = tl.arange($(136))
    r40_col = tl.arange($(136))
    r40_lane = tl.arange($(264))
    r40_left = tl.load($(vaddmm_15_ptr) + (((r40_row[:, None] % $(136)) * $(264)) + (r40_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r40_right = tl.load($(vaddmm_15_ptr) + (((r40_col[None, :] % $(136)) * $(264)) + (r40_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r40_value = tl.dot(r40_left, r40_right)
    r40_value = r40_value.to(tl.bfloat16)
    tl.store($(vmm_8_ptr) + ((r40_row[:, None] * $(136)) + r40_col[None, :]), r40_value)
    r41_row = tl.arange($(136))
    r41_col = tl.arange($(136))
    r41_lane = tl.arange($(136))
    r41_left = tl.load($(vmm_8_ptr) + (((r41_row[:, None] % $(136)) * $(136)) + (r41_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r41_right = tl.load($(vmm_8_ptr) + (((r41_lane[:, None] % $(136)) * $(136)) + (r41_col[None, :] % $(136))), dtype=tl.bfloat16)
    r41_value = tl.dot(r41_left, r41_right)
    r41_value = r41_value * 2.0315
    r41_value = r41_value + tl.load($(vmm_8_ptr) + (((r41_row[:, None] % $(136)) * $(136)) + (r41_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r41_value = r41_value.to(tl.bfloat16)
    tl.store($(vaddmm_16_ptr) + ((r41_row[:, None] * $(136)) + r41_col[None, :]), r41_value)
    r42_row = tl.arange($(136))
    r42_col = tl.arange($(264))
    r42_lane = tl.arange($(136))
    r42_left = tl.load($(vaddmm_16_ptr) + (((r42_row[:, None] % $(136)) * $(136)) + (r42_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r42_right = tl.load($(vaddmm_15_ptr) + (((r42_lane[:, None] % $(136)) * $(264)) + (r42_col[None, :] % $(264))), dtype=tl.bfloat16)
    r42_value = tl.dot(r42_left, r42_right)
    r42_value = r42_value + tl.load($(vaddmm_15_ptr) + (((r42_row[:, None] % $(136)) * $(264)) + (r42_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r42_value = r42_value.to(tl.bfloat16)
    tl.store($(vaddmm_17_ptr) + ((r42_row[:, None] * $(264)) + r42_col[None, :]), r42_value)
    r43_row = tl.arange($(136))
    r43_col = tl.arange($(136))
    r43_lane = tl.arange($(264))
    r43_left = tl.load($(vaddmm_17_ptr) + (((r43_row[:, None] % $(136)) * $(264)) + (r43_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r43_right = tl.load($(vaddmm_17_ptr) + (((r43_col[None, :] % $(136)) * $(264)) + (r43_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r43_value = tl.dot(r43_left, r43_right)
    r43_value = r43_value.to(tl.bfloat16)
    tl.store($(vmm_9_ptr) + ((r43_row[:, None] * $(136)) + r43_col[None, :]), r43_value)
    r44_row = tl.arange($(136))
    r44_col = tl.arange($(136))
    r44_lane = tl.arange($(136))
    r44_left = tl.load($(vmm_9_ptr) + (((r44_row[:, None] % $(136)) * $(136)) + (r44_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r44_right = tl.load($(vmm_9_ptr) + (((r44_lane[:, None] % $(136)) * $(136)) + (r44_col[None, :] % $(136))), dtype=tl.bfloat16)
    r44_value = tl.dot(r44_left, r44_right)
    r44_value = r44_value * 2.0315
    r44_value = r44_value + tl.load($(vmm_9_ptr) + (((r44_row[:, None] % $(136)) * $(136)) + (r44_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r44_value = r44_value.to(tl.bfloat16)
    tl.store($(vaddmm_18_ptr) + ((r44_row[:, None] * $(136)) + r44_col[None, :]), r44_value)
    r45_row = tl.arange($(136))
    r45_col = tl.arange($(264))
    r45_lane = tl.arange($(136))
    r45_left = tl.load($(vaddmm_18_ptr) + (((r45_row[:, None] % $(136)) * $(136)) + (r45_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r45_right = tl.load($(vaddmm_17_ptr) + (((r45_lane[:, None] % $(136)) * $(264)) + (r45_col[None, :] % $(264))), dtype=tl.bfloat16)
    r45_value = tl.dot(r45_left, r45_right)
    r45_value = r45_value + tl.load($(vaddmm_17_ptr) + (((r45_row[:, None] % $(136)) * $(264)) + (r45_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r45_value = r45_value.to(tl.bfloat16)
    tl.store($(vaddmm_19_ptr) + ((r45_row[:, None] * $(264)) + r45_col[None, :]), r45_value)
    r46_z = tl.arange($(35904))
    r46_value = (tl.load($(vselect_scatter_2_ptr) + ((((r46_z / $(136)) % $(264)) * $(136)) + (r46_z % $(136)) + $(35904)), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(35904)])
    tl.store($(vmul_1_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(107712))
    r47_value = tl.where((((r47_z / $(35904)) % $(3)) == $(1)), tl.load($(vmul_1_ptr) + ((((r47_z / $(136)) % $(264)) * $(136)) + (r47_z % $(136))), dtype=tl.bfloat16), tl.load($(vselect_scatter_2_ptr) + ((((r47_z / $(35904)) % $(3)) * $(35904)) + (((r47_z / $(136)) % $(264)) * $(136)) + (r47_z % $(136))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(107712)])
    tl.store($(vselect_scatter_4_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(35904))
    r48_value = (tl.load($(vselect_scatter_4_ptr) + ((((r48_z / $(136)) % $(264)) * $(136)) + (r48_z % $(136)) + $(35904)), dtype=tl.bfloat16) + (tl.load($(vaddmm_19_ptr) + (((r48_z % $(136)) * $(264)) + ((r48_z / $(136)) % $(264))), dtype=tl.bfloat16) * -0.03249615361854385))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(35904)])
    tl.store($(vadd_1_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(107712))
    r49_value = tl.where((((r49_z / $(35904)) % $(3)) == $(1)), tl.load($(vadd_1_ptr) + ((((r49_z / $(136)) % $(264)) * $(136)) + (r49_z % $(136))), dtype=tl.bfloat16), tl.load($(vselect_scatter_4_ptr) + ((((r49_z / $(35904)) % $(3)) * $(35904)) + (((r49_z / $(136)) % $(264)) * $(136)) + (r49_z % $(136))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(107712)])
    tl.store($(vselect_scatter_5_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(35904))
    r50_value = (tl.load($(vselect_scatter_3_ptr) + ((((r50_z / $(136)) % $(264)) * $(136)) + (r50_z % $(136)) + $(71808)), dtype=tl.bfloat16) + (tl.load($(input0_ptr) + ((((r50_z / $(136)) % $(264)) * $(136)) + (r50_z % $(136)) + $(71808)), dtype=tl.bfloat16) - tl.load($(vselect_scatter_3_ptr) + ((((r50_z / $(136)) % $(264)) * $(136)) + (r50_z % $(136)) + $(71808)), dtype=tl.bfloat16)) * 0.050000000000000044)
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(35904)])
    tl.store($(vlerp_4_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(107712))
    r51_value = tl.where((((r51_z / $(35904)) % $(3)) == $(2)), tl.load($(vlerp_4_ptr) + ((((r51_z / $(136)) % $(264)) * $(136)) + (r51_z % $(136))), dtype=tl.bfloat16), tl.load($(vselect_scatter_3_ptr) + ((((r51_z / $(35904)) % $(3)) * $(35904)) + (((r51_z / $(136)) % $(264)) * $(136)) + (r51_z % $(136))), dtype=tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(107712)])
    tl.store($(vselect_scatter_6_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(35904))
    r52_value = (tl.load($(input0_ptr) + ((((r52_z / $(136)) % $(264)) * $(136)) + (r52_z % $(136)) + $(71808)), dtype=tl.bfloat16) + (tl.load($(vselect_scatter_6_ptr) + ((((r52_z / $(136)) % $(264)) * $(136)) + (r52_z % $(136)) + $(71808)), dtype=tl.bfloat16) - tl.load($(input0_ptr) + ((((r52_z / $(136)) % $(264)) * $(136)) + (r52_z % $(136)) + $(71808)), dtype=tl.bfloat16)) * 0.95)
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(35904)])
    tl.store($(vlerp_5_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(1))
    r53_lane = tl.arange($(35904))
    r53_value = tl.sqrt(tl.sum(tl.load($(vlerp_5_ptr) + (((r53_lane[None, :] % $(264)) * $(136)) + ((r53_lane[None, :] / $(264)) % $(136))), dtype=tl.bfloat16) * tl.load($(vlerp_5_ptr) + (((r53_lane[None, :] % $(264)) * $(136)) + ((r53_lane[None, :] / $(264)) % $(136))), dtype=tl.bfloat16), axis=1))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(1)])
    tl.store($(vlinalg_vector_norm_2_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(1))
    r54_value = tl.maximum(tl.load($(vlinalg_vector_norm_2_ptr) + $(0), dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(1)])
    tl.store($(vclamp_2_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(35904))
    r55_value = (tl.load($(vlerp_5_ptr) + (((r55_z % $(264)) * $(136)) + ((r55_z / $(264)) % $(136))), dtype=tl.bfloat16) / tl.load($(vclamp_2_ptr) + $(0), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(35904)])
    tl.store($(vdiv_2_ptr) + r55_z, r55_value)
    r56_row = tl.arange($(136))
    r56_col = tl.arange($(136))
    r56_lane = tl.arange($(264))
    r56_left = tl.load($(vdiv_2_ptr) + (((r56_row[:, None] % $(136)) * $(264)) + (r56_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r56_right = tl.load($(vdiv_2_ptr) + (((r56_col[None, :] % $(136)) * $(264)) + (r56_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r56_value = tl.dot(r56_left, r56_right)
    r56_value = r56_value.to(tl.bfloat16)
    tl.store($(vmm_10_ptr) + ((r56_row[:, None] * $(136)) + r56_col[None, :]), r56_value)
    r57_row = tl.arange($(136))
    r57_col = tl.arange($(136))
    r57_lane = tl.arange($(136))
    r57_left = tl.load($(vmm_10_ptr) + (((r57_row[:, None] % $(136)) * $(136)) + (r57_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r57_right = tl.load($(vmm_10_ptr) + (((r57_lane[:, None] % $(136)) * $(136)) + (r57_col[None, :] % $(136))), dtype=tl.bfloat16)
    r57_value = tl.dot(r57_left, r57_right)
    r57_value = r57_value * 2.0315
    r57_value = r57_value + tl.load($(vmm_10_ptr) + (((r57_row[:, None] % $(136)) * $(136)) + (r57_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r57_value = r57_value.to(tl.bfloat16)
    tl.store($(vaddmm_20_ptr) + ((r57_row[:, None] * $(136)) + r57_col[None, :]), r57_value)
    r58_row = tl.arange($(136))
    r58_col = tl.arange($(264))
    r58_lane = tl.arange($(136))
    r58_left = tl.load($(vaddmm_20_ptr) + (((r58_row[:, None] % $(136)) * $(136)) + (r58_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r58_right = tl.load($(vdiv_2_ptr) + (((r58_lane[:, None] % $(136)) * $(264)) + (r58_col[None, :] % $(264))), dtype=tl.bfloat16)
    r58_value = tl.dot(r58_left, r58_right)
    r58_value = r58_value + tl.load($(vdiv_2_ptr) + (((r58_row[:, None] % $(136)) * $(264)) + (r58_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r58_value = r58_value.to(tl.bfloat16)
    tl.store($(vaddmm_21_ptr) + ((r58_row[:, None] * $(264)) + r58_col[None, :]), r58_value)
    r59_row = tl.arange($(136))
    r59_col = tl.arange($(136))
    r59_lane = tl.arange($(264))
    r59_left = tl.load($(vaddmm_21_ptr) + (((r59_row[:, None] % $(136)) * $(264)) + (r59_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r59_right = tl.load($(vaddmm_21_ptr) + (((r59_col[None, :] % $(136)) * $(264)) + (r59_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r59_value = tl.dot(r59_left, r59_right)
    r59_value = r59_value.to(tl.bfloat16)
    tl.store($(vmm_11_ptr) + ((r59_row[:, None] * $(136)) + r59_col[None, :]), r59_value)
    r60_row = tl.arange($(136))
    r60_col = tl.arange($(136))
    r60_lane = tl.arange($(136))
    r60_left = tl.load($(vmm_11_ptr) + (((r60_row[:, None] % $(136)) * $(136)) + (r60_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r60_right = tl.load($(vmm_11_ptr) + (((r60_lane[:, None] % $(136)) * $(136)) + (r60_col[None, :] % $(136))), dtype=tl.bfloat16)
    r60_value = tl.dot(r60_left, r60_right)
    r60_value = r60_value * 2.0315
    r60_value = r60_value + tl.load($(vmm_11_ptr) + (((r60_row[:, None] % $(136)) * $(136)) + (r60_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r60_value = r60_value.to(tl.bfloat16)
    tl.store($(vaddmm_22_ptr) + ((r60_row[:, None] * $(136)) + r60_col[None, :]), r60_value)
    r61_row = tl.arange($(136))
    r61_col = tl.arange($(264))
    r61_lane = tl.arange($(136))
    r61_left = tl.load($(vaddmm_22_ptr) + (((r61_row[:, None] % $(136)) * $(136)) + (r61_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r61_right = tl.load($(vaddmm_21_ptr) + (((r61_lane[:, None] % $(136)) * $(264)) + (r61_col[None, :] % $(264))), dtype=tl.bfloat16)
    r61_value = tl.dot(r61_left, r61_right)
    r61_value = r61_value + tl.load($(vaddmm_21_ptr) + (((r61_row[:, None] % $(136)) * $(264)) + (r61_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r61_value = r61_value.to(tl.bfloat16)
    tl.store($(vaddmm_23_ptr) + ((r61_row[:, None] * $(264)) + r61_col[None, :]), r61_value)
    r62_row = tl.arange($(136))
    r62_col = tl.arange($(136))
    r62_lane = tl.arange($(264))
    r62_left = tl.load($(vaddmm_23_ptr) + (((r62_row[:, None] % $(136)) * $(264)) + (r62_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r62_right = tl.load($(vaddmm_23_ptr) + (((r62_col[None, :] % $(136)) * $(264)) + (r62_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r62_value = tl.dot(r62_left, r62_right)
    r62_value = r62_value.to(tl.bfloat16)
    tl.store($(vmm_12_ptr) + ((r62_row[:, None] * $(136)) + r62_col[None, :]), r62_value)
    r63_row = tl.arange($(136))
    r63_col = tl.arange($(136))
    r63_lane = tl.arange($(136))
    r63_left = tl.load($(vmm_12_ptr) + (((r63_row[:, None] % $(136)) * $(136)) + (r63_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r63_right = tl.load($(vmm_12_ptr) + (((r63_lane[:, None] % $(136)) * $(136)) + (r63_col[None, :] % $(136))), dtype=tl.bfloat16)
    r63_value = tl.dot(r63_left, r63_right)
    r63_value = r63_value * 2.0315
    r63_value = r63_value + tl.load($(vmm_12_ptr) + (((r63_row[:, None] % $(136)) * $(136)) + (r63_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r63_value = r63_value.to(tl.bfloat16)
    tl.store($(vaddmm_24_ptr) + ((r63_row[:, None] * $(136)) + r63_col[None, :]), r63_value)
    r64_row = tl.arange($(136))
    r64_col = tl.arange($(264))
    r64_lane = tl.arange($(136))
    r64_left = tl.load($(vaddmm_24_ptr) + (((r64_row[:, None] % $(136)) * $(136)) + (r64_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r64_right = tl.load($(vaddmm_23_ptr) + (((r64_lane[:, None] % $(136)) * $(264)) + (r64_col[None, :] % $(264))), dtype=tl.bfloat16)
    r64_value = tl.dot(r64_left, r64_right)
    r64_value = r64_value + tl.load($(vaddmm_23_ptr) + (((r64_row[:, None] % $(136)) * $(264)) + (r64_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r64_value = r64_value.to(tl.bfloat16)
    tl.store($(vaddmm_25_ptr) + ((r64_row[:, None] * $(264)) + r64_col[None, :]), r64_value)
    r65_row = tl.arange($(136))
    r65_col = tl.arange($(136))
    r65_lane = tl.arange($(264))
    r65_left = tl.load($(vaddmm_25_ptr) + (((r65_row[:, None] % $(136)) * $(264)) + (r65_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r65_right = tl.load($(vaddmm_25_ptr) + (((r65_col[None, :] % $(136)) * $(264)) + (r65_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r65_value = tl.dot(r65_left, r65_right)
    r65_value = r65_value.to(tl.bfloat16)
    tl.store($(vmm_13_ptr) + ((r65_row[:, None] * $(136)) + r65_col[None, :]), r65_value)
    r66_row = tl.arange($(136))
    r66_col = tl.arange($(136))
    r66_lane = tl.arange($(136))
    r66_left = tl.load($(vmm_13_ptr) + (((r66_row[:, None] % $(136)) * $(136)) + (r66_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r66_right = tl.load($(vmm_13_ptr) + (((r66_lane[:, None] % $(136)) * $(136)) + (r66_col[None, :] % $(136))), dtype=tl.bfloat16)
    r66_value = tl.dot(r66_left, r66_right)
    r66_value = r66_value * 2.0315
    r66_value = r66_value + tl.load($(vmm_13_ptr) + (((r66_row[:, None] % $(136)) * $(136)) + (r66_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r66_value = r66_value.to(tl.bfloat16)
    tl.store($(vaddmm_26_ptr) + ((r66_row[:, None] * $(136)) + r66_col[None, :]), r66_value)
    r67_row = tl.arange($(136))
    r67_col = tl.arange($(264))
    r67_lane = tl.arange($(136))
    r67_left = tl.load($(vaddmm_26_ptr) + (((r67_row[:, None] % $(136)) * $(136)) + (r67_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r67_right = tl.load($(vaddmm_25_ptr) + (((r67_lane[:, None] % $(136)) * $(264)) + (r67_col[None, :] % $(264))), dtype=tl.bfloat16)
    r67_value = tl.dot(r67_left, r67_right)
    r67_value = r67_value + tl.load($(vaddmm_25_ptr) + (((r67_row[:, None] % $(136)) * $(264)) + (r67_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r67_value = r67_value.to(tl.bfloat16)
    tl.store($(vaddmm_27_ptr) + ((r67_row[:, None] * $(264)) + r67_col[None, :]), r67_value)
    r68_row = tl.arange($(136))
    r68_col = tl.arange($(136))
    r68_lane = tl.arange($(264))
    r68_left = tl.load($(vaddmm_27_ptr) + (((r68_row[:, None] % $(136)) * $(264)) + (r68_lane[None, :] % $(264))), dtype=tl.bfloat16)
    r68_right = tl.load($(vaddmm_27_ptr) + (((r68_col[None, :] % $(136)) * $(264)) + (r68_lane[:, None] % $(264))), dtype=tl.bfloat16)
    r68_value = tl.dot(r68_left, r68_right)
    r68_value = r68_value.to(tl.bfloat16)
    tl.store($(vmm_14_ptr) + ((r68_row[:, None] * $(136)) + r68_col[None, :]), r68_value)
    r69_row = tl.arange($(136))
    r69_col = tl.arange($(136))
    r69_lane = tl.arange($(136))
    r69_left = tl.load($(vmm_14_ptr) + (((r69_row[:, None] % $(136)) * $(136)) + (r69_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r69_right = tl.load($(vmm_14_ptr) + (((r69_lane[:, None] % $(136)) * $(136)) + (r69_col[None, :] % $(136))), dtype=tl.bfloat16)
    r69_value = tl.dot(r69_left, r69_right)
    r69_value = r69_value * 2.0315
    r69_value = r69_value + tl.load($(vmm_14_ptr) + (((r69_row[:, None] % $(136)) * $(136)) + (r69_col[None, :] % $(136))), dtype=tl.bfloat16) * -4.775
    r69_value = r69_value.to(tl.bfloat16)
    tl.store($(vaddmm_28_ptr) + ((r69_row[:, None] * $(136)) + r69_col[None, :]), r69_value)
    r70_row = tl.arange($(136))
    r70_col = tl.arange($(264))
    r70_lane = tl.arange($(136))
    r70_left = tl.load($(vaddmm_28_ptr) + (((r70_row[:, None] % $(136)) * $(136)) + (r70_lane[None, :] % $(136))), dtype=tl.bfloat16)
    r70_right = tl.load($(vaddmm_27_ptr) + (((r70_lane[:, None] % $(136)) * $(264)) + (r70_col[None, :] % $(264))), dtype=tl.bfloat16)
    r70_value = tl.dot(r70_left, r70_right)
    r70_value = r70_value + tl.load($(vaddmm_27_ptr) + (((r70_row[:, None] % $(136)) * $(264)) + (r70_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
    r70_value = r70_value.to(tl.bfloat16)
    tl.store($(vaddmm_29_ptr) + ((r70_row[:, None] * $(264)) + r70_col[None, :]), r70_value)
    r71_z = tl.arange($(35904))
    r71_value = (tl.load($(vselect_scatter_5_ptr) + ((((r71_z / $(136)) % $(264)) * $(136)) + (r71_z % $(136)) + $(71808)), dtype=tl.bfloat16) * tl.cast(0.9999, tl.bfloat16))
    r71_value = r71_value.to(tl.bfloat16)
    r71_value = tl.broadcast_to(r71_value, [$(35904)])
    tl.store($(vmul_2_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(107712))
    r72_value = tl.where((((r72_z / $(35904)) % $(3)) == $(2)), tl.load($(vmul_2_ptr) + ((((r72_z / $(136)) % $(264)) * $(136)) + (r72_z % $(136))), dtype=tl.bfloat16), tl.load($(vselect_scatter_5_ptr) + ((((r72_z / $(35904)) % $(3)) * $(35904)) + (((r72_z / $(136)) % $(264)) * $(136)) + (r72_z % $(136))), dtype=tl.bfloat16))
    r72_value = r72_value.to(tl.bfloat16)
    r72_value = tl.broadcast_to(r72_value, [$(107712)])
    tl.store($(vselect_scatter_7_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(35904))
    r73_value = (tl.load($(vselect_scatter_7_ptr) + ((((r73_z / $(136)) % $(264)) * $(136)) + (r73_z % $(136)) + $(71808)), dtype=tl.bfloat16) + (tl.load($(vaddmm_29_ptr) + (((r73_z % $(136)) * $(264)) + ((r73_z / $(136)) % $(264))), dtype=tl.bfloat16) * -0.03249615361854385))
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(35904)])
    tl.store($(vadd_2_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(107712))
    r74_value = tl.where((((r74_z / $(35904)) % $(3)) == $(2)), tl.load($(vadd_2_ptr) + ((((r74_z / $(136)) % $(264)) * $(136)) + (r74_z % $(136))), dtype=tl.bfloat16), tl.load($(vselect_scatter_7_ptr) + ((((r74_z / $(35904)) % $(3)) * $(35904)) + (((r74_z / $(136)) % $(264)) * $(136)) + (r74_z % $(136))), dtype=tl.bfloat16))
    r74_value = r74_value.to(tl.bfloat16)
    r74_value = tl.broadcast_to(r74_value, [$(107712)])
    tl.store($(vselect_scatter_8_ptr) + r74_z, r74_value)
    z = tl.arange($(107712))
    value = tl.load($(vselect_scatter_8_ptr) + ((((z / $(35904)) % $(3)) * $(35904)) + (((z / $(136)) % $(264)) * $(136)) + (z % $(136))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(107712)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vlinalg_vector_norm_ptr : RegionName := "vlinalg_vector_norm"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vadd_2_ptr : RegionName := "vadd_2"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vfma_ptr : RegionName := "candidate_vfma"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vsub_2_ptr : RegionName := "candidate_vsub_2"
  let candidate_vfma_1_ptr : RegionName := "candidate_vfma_1"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vlinalg_vector_norm_ptr : RegionName := "candidate_vlinalg_vector_norm"
  let candidate_vclamp_min_ptr : RegionName := "candidate_vclamp_min"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vle_ptr : RegionName := "candidate_vle"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vle_1_ptr : RegionName := "candidate_vle_1"
  let candidate_vwhere_1_ptr : RegionName := "candidate_vwhere_1"
  let candidate_vbaddbmm_ptr : RegionName := "candidate_vbaddbmm"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vle_2_ptr : RegionName := "candidate_vle_2"
  let candidate_vwhere_2_ptr : RegionName := "candidate_vwhere_2"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vle_3_ptr : RegionName := "candidate_vle_3"
  let candidate_vwhere_3_ptr : RegionName := "candidate_vwhere_3"
  let candidate_vbaddbmm_1_ptr : RegionName := "candidate_vbaddbmm_1"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmul_6_ptr : RegionName := "candidate_vmul_6"
  let candidate_vle_4_ptr : RegionName := "candidate_vle_4"
  let candidate_vwhere_4_ptr : RegionName := "candidate_vwhere_4"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vle_5_ptr : RegionName := "candidate_vle_5"
  let candidate_vwhere_5_ptr : RegionName := "candidate_vwhere_5"
  let candidate_vbaddbmm_2_ptr : RegionName := "candidate_vbaddbmm_2"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vle_6_ptr : RegionName := "candidate_vle_6"
  let candidate_vwhere_6_ptr : RegionName := "candidate_vwhere_6"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vle_7_ptr : RegionName := "candidate_vle_7"
  let candidate_vwhere_7_ptr : RegionName := "candidate_vwhere_7"
  let candidate_vbaddbmm_3_ptr : RegionName := "candidate_vbaddbmm_3"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vle_8_ptr : RegionName := "candidate_vle_8"
  let candidate_vwhere_8_ptr : RegionName := "candidate_vwhere_8"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vle_9_ptr : RegionName := "candidate_vle_9"
  let candidate_vwhere_9_ptr : RegionName := "candidate_vwhere_9"
  let candidate_vbaddbmm_4_ptr : RegionName := "candidate_vbaddbmm_4"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vto_copy_7_ptr : RegionName := "candidate_vto_copy_7"
  let candidate_vfma_2_ptr : RegionName := "candidate_vfma_2"
  let candidate_vto_copy_8_ptr : RegionName := "candidate_vto_copy_8"
  triton {
    r0_z = tl.arange($(107712))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(35904)) % $(3)) * $(35904)) + (((r0_z / $(136)) % $(264)) * $(136)) + (r0_z % $(136))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(107712)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(107712))
    r1_value = 0.0
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(107712)])
    tl.store($(candidate_vto_copy_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(107712))
    r2_value = (tl.load($(candidate_vto_copy_ptr) + ((((r2_z / $(35904)) % $(3)) * $(35904)) + (((r2_z / $(136)) % $(264)) * $(136)) + (r2_z % $(136))), dtype=tl.float32) - tl.load($(candidate_vto_copy_1_ptr) + ((((r2_z / $(35904)) % $(3)) * $(35904)) + (((r2_z / $(136)) % $(264)) * $(136)) + (r2_z % $(136))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(107712)])
    tl.store($(candidate_vsub_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(107712))
    r3_value = tl.fma(tl.cast(0.050000000000000044, tl.float32), tl.cast(tl.load($(candidate_vsub_ptr) + ((((r3_z / $(35904)) % $(3)) * $(35904)) + (((r3_z / $(136)) % $(264)) * $(136)) + (r3_z % $(136))), dtype=tl.float32), tl.float32), tl.cast(tl.load($(candidate_vto_copy_1_ptr) + ((((r3_z / $(35904)) % $(3)) * $(35904)) + (((r3_z / $(136)) % $(264)) * $(136)) + (r3_z % $(136))), dtype=tl.float32), tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(107712)])
    tl.store($(candidate_vfma_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(107712))
    r4_value = (1.0 - 0.95)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(107712)])
    tl.store($(candidate_vsub_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(107712))
    r5_value = -tl.load($(candidate_vsub_1_ptr) + ((((r5_z / $(35904)) % $(3)) * $(35904)) + (((r5_z / $(136)) % $(264)) * $(136)) + (r5_z % $(136))), dtype=tl.float32)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(107712)])
    tl.store($(candidate_vneg_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(107712))
    r6_value = (tl.load($(candidate_vfma_ptr) + ((((r6_z / $(35904)) % $(3)) * $(35904)) + (((r6_z / $(136)) % $(264)) * $(136)) + (r6_z % $(136))), dtype=tl.float32) - tl.load($(candidate_vto_copy_ptr) + ((((r6_z / $(35904)) % $(3)) * $(35904)) + (((r6_z / $(136)) % $(264)) * $(136)) + (r6_z % $(136))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(107712)])
    tl.store($(candidate_vsub_2_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(107712))
    r7_value = tl.fma(tl.cast(tl.load($(candidate_vneg_ptr) + ((((r7_z / $(35904)) % $(3)) * $(35904)) + (((r7_z / $(136)) % $(264)) * $(136)) + (r7_z % $(136))), dtype=tl.float32), tl.float32), tl.cast(tl.load($(candidate_vsub_2_ptr) + ((((r7_z / $(35904)) % $(3)) * $(35904)) + (((r7_z / $(136)) % $(264)) * $(136)) + (r7_z % $(136))), dtype=tl.float32), tl.float32), tl.cast(tl.load($(candidate_vfma_ptr) + ((((r7_z / $(35904)) % $(3)) * $(35904)) + (((r7_z / $(136)) % $(264)) * $(136)) + (r7_z % $(136))), dtype=tl.float32), tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(107712)])
    tl.store($(candidate_vfma_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(107712))
    r8_value = tl.load($(candidate_vfma_1_ptr) + ((((r8_z / $(35904)) % $(3)) * $(35904)) + (((r8_z / $(136)) % $(264)) * $(136)) + (r8_z % $(136))), dtype=tl.float32)
    r8_value = r8_value.to(tl.bfloat16)
    r8_value = tl.broadcast_to(r8_value, [$(107712)])
    tl.store($(candidate_vto_copy_3_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(3))
    r9_lane = tl.arange($(35904))
    r9_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_3_ptr) + ((r9_z[:, None] * $(35904)) + ((r9_lane[None, :] % $(264)) * $(136)) + ((r9_lane[None, :] / $(264)) % $(136))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_3_ptr) + ((r9_z[:, None] * $(35904)) + ((r9_lane[None, :] % $(264)) * $(136)) + ((r9_lane[None, :] / $(264)) % $(136))), dtype=tl.bfloat16), axis=1))
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(3)])
    tl.store($(candidate_vlinalg_vector_norm_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(3))
    r10_value = tl.maximum(tl.load($(candidate_vlinalg_vector_norm_ptr) + r10_z, dtype=tl.bfloat16), tl.cast(0.0000001, tl.bfloat16))
    r10_value = r10_value.to(tl.bfloat16)
    r10_value = tl.broadcast_to(r10_value, [$(3)])
    tl.store($(candidate_vclamp_min_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(107712))
    r11_value = (tl.load($(candidate_vto_copy_3_ptr) + ((((r11_z / $(35904)) % $(3)) * $(35904)) + ((r11_z % $(264)) * $(136)) + ((r11_z / $(264)) % $(136))), dtype=tl.bfloat16) / tl.load($(candidate_vclamp_min_ptr) + ((r11_z / $(35904)) % $(3)), dtype=tl.bfloat16))
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(107712)])
    tl.store($(candidate_vdiv_ptr) + r11_z, r11_value)
    r12_row = tl.arange($(136))
    r12_col = tl.arange($(136))
    r12_lane = tl.arange($(264))
    for r12_group in range($(3)) {
      r12_left = tl.load($(candidate_vdiv_ptr) + (((r12_group % $(3)) * $(35904)) + ((r12_row[:, None] % $(136)) * $(264)) + (r12_lane[None, :] % $(264))), dtype=tl.bfloat16)
      r12_right = tl.load($(candidate_vdiv_ptr) + (((r12_group % $(3)) * $(35904)) + ((r12_col[None, :] % $(136)) * $(264)) + (r12_lane[:, None] % $(264))), dtype=tl.bfloat16)
      r12_value = tl.dot(r12_left, r12_right)
      r12_value = r12_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r12_group * $(18496)) + (r12_row[:, None] * $(136)) + r12_col[None, :]), r12_value)
    }
    r13_z = tl.arange($(55488))
    r13_value = (tl.load($(candidate_vbmm_ptr) + ((((r13_z / $(18496)) % $(3)) * $(18496)) + (((r13_z / $(136)) % $(136)) * $(136)) + (r13_z % $(136))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(55488)])
    tl.store($(candidate_vmul_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(18496))
    r14_value = (tl.cast(((r14_z / $(136)) % $(136)), tl.int64) <= tl.cast((r14_z % $(136)), tl.int64))
    r14_value = tl.broadcast_to(r14_value, [$(18496)])
    tl.store($(candidate_vle_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(55488))
    r15_value = tl.where(tl.load($(candidate_vle_ptr) + ((((r15_z / $(136)) % $(136)) * $(136)) + (r15_z % $(136))), dtype=tl.int1), tl.load($(candidate_vmul_ptr) + ((((r15_z / $(18496)) % $(3)) * $(18496)) + (((r15_z / $(136)) % $(136)) * $(136)) + (r15_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vmul_ptr) + ((((r15_z / $(18496)) % $(3)) * $(18496)) + ((r15_z % $(136)) * $(136)) + ((r15_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(55488)])
    tl.store($(candidate_vwhere_ptr) + r15_z, r15_value)
    r16_row = tl.arange($(136))
    r16_col = tl.arange($(136))
    r16_lane = tl.arange($(136))
    for r16_group in range($(3)) {
      r16_left = tl.load($(candidate_vwhere_ptr) + (((r16_group % $(3)) * $(18496)) + ((r16_row[:, None] % $(136)) * $(136)) + (r16_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r16_right = tl.load($(candidate_vwhere_ptr) + (((r16_group % $(3)) * $(18496)) + ((r16_col[None, :] % $(136)) * $(136)) + (r16_lane[:, None] % $(136))), dtype=tl.bfloat16)
      r16_value = tl.dot(r16_left, r16_right)
      r16_value = r16_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_1_ptr) + ((r16_group * $(18496)) + (r16_row[:, None] * $(136)) + r16_col[None, :]), r16_value)
    }
    r17_z = tl.arange($(55488))
    r17_value = (tl.load($(candidate_vbmm_1_ptr) + ((((r17_z / $(18496)) % $(3)) * $(18496)) + (((r17_z / $(136)) % $(136)) * $(136)) + (r17_z % $(136))), dtype=tl.bfloat16) * tl.cast(2.0315, tl.bfloat16))
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(55488)])
    tl.store($(candidate_vmul_1_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(55488))
    r18_value = (tl.load($(candidate_vwhere_ptr) + ((((r18_z / $(18496)) % $(3)) * $(18496)) + (((r18_z / $(136)) % $(136)) * $(136)) + (r18_z % $(136))), dtype=tl.bfloat16) * tl.cast(-4.775, tl.bfloat16))
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(55488)])
    tl.store($(candidate_vmul_2_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(55488))
    r19_value = (tl.load($(candidate_vmul_1_ptr) + ((((r19_z / $(18496)) % $(3)) * $(18496)) + (((r19_z / $(136)) % $(136)) * $(136)) + (r19_z % $(136))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_2_ptr) + ((((r19_z / $(18496)) % $(3)) * $(18496)) + (((r19_z / $(136)) % $(136)) * $(136)) + (r19_z % $(136))), dtype=tl.bfloat16))
    r19_value = r19_value.to(tl.bfloat16)
    r19_value = tl.broadcast_to(r19_value, [$(55488)])
    tl.store($(candidate_vadd_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(18496))
    r20_value = (tl.cast(((r20_z / $(136)) % $(136)), tl.int64) <= tl.cast((r20_z % $(136)), tl.int64))
    r20_value = tl.broadcast_to(r20_value, [$(18496)])
    tl.store($(candidate_vle_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(55488))
    r21_value = tl.where(tl.load($(candidate_vle_1_ptr) + ((((r21_z / $(136)) % $(136)) * $(136)) + (r21_z % $(136))), dtype=tl.int1), tl.load($(candidate_vadd_ptr) + ((((r21_z / $(18496)) % $(3)) * $(18496)) + (((r21_z / $(136)) % $(136)) * $(136)) + (r21_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vadd_ptr) + ((((r21_z / $(18496)) % $(3)) * $(18496)) + ((r21_z % $(136)) * $(136)) + ((r21_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r21_value = r21_value.to(tl.bfloat16)
    r21_value = tl.broadcast_to(r21_value, [$(55488)])
    tl.store($(candidate_vwhere_1_ptr) + r21_z, r21_value)
    r22_row = tl.arange($(136))
    r22_col = tl.arange($(264))
    r22_lane = tl.arange($(136))
    for r22_group in range($(3)) {
      r22_left = tl.load($(candidate_vwhere_1_ptr) + (((r22_group % $(3)) * $(18496)) + ((r22_row[:, None] % $(136)) * $(136)) + (r22_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r22_right = tl.load($(candidate_vdiv_ptr) + (((r22_group % $(3)) * $(35904)) + ((r22_lane[:, None] % $(136)) * $(264)) + (r22_col[None, :] % $(264))), dtype=tl.bfloat16)
      r22_value = tl.dot(r22_left, r22_right)
      r22_value = r22_value + tl.load($(candidate_vdiv_ptr) + (((r22_group % $(3)) * $(35904)) + ((r22_row[:, None] % $(136)) * $(264)) + (r22_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
      r22_value = r22_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_ptr) + ((r22_group * $(35904)) + (r22_row[:, None] * $(264)) + r22_col[None, :]), r22_value)
    }
    r23_row = tl.arange($(136))
    r23_col = tl.arange($(136))
    r23_lane = tl.arange($(264))
    for r23_group in range($(3)) {
      r23_left = tl.load($(candidate_vbaddbmm_ptr) + (((r23_group % $(3)) * $(35904)) + ((r23_row[:, None] % $(136)) * $(264)) + (r23_lane[None, :] % $(264))), dtype=tl.bfloat16)
      r23_right = tl.load($(candidate_vbaddbmm_ptr) + (((r23_group % $(3)) * $(35904)) + ((r23_col[None, :] % $(136)) * $(264)) + (r23_lane[:, None] % $(264))), dtype=tl.bfloat16)
      r23_value = tl.dot(r23_left, r23_right)
      r23_value = r23_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r23_group * $(18496)) + (r23_row[:, None] * $(136)) + r23_col[None, :]), r23_value)
    }
    r24_z = tl.arange($(55488))
    r24_value = (tl.load($(candidate_vbmm_2_ptr) + ((((r24_z / $(18496)) % $(3)) * $(18496)) + (((r24_z / $(136)) % $(136)) * $(136)) + (r24_z % $(136))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(55488)])
    tl.store($(candidate_vmul_3_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(18496))
    r25_value = (tl.cast(((r25_z / $(136)) % $(136)), tl.int64) <= tl.cast((r25_z % $(136)), tl.int64))
    r25_value = tl.broadcast_to(r25_value, [$(18496)])
    tl.store($(candidate_vle_2_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(55488))
    r26_value = tl.where(tl.load($(candidate_vle_2_ptr) + ((((r26_z / $(136)) % $(136)) * $(136)) + (r26_z % $(136))), dtype=tl.int1), tl.load($(candidate_vmul_3_ptr) + ((((r26_z / $(18496)) % $(3)) * $(18496)) + (((r26_z / $(136)) % $(136)) * $(136)) + (r26_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vmul_3_ptr) + ((((r26_z / $(18496)) % $(3)) * $(18496)) + ((r26_z % $(136)) * $(136)) + ((r26_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(55488)])
    tl.store($(candidate_vwhere_2_ptr) + r26_z, r26_value)
    r27_row = tl.arange($(136))
    r27_col = tl.arange($(136))
    r27_lane = tl.arange($(136))
    for r27_group in range($(3)) {
      r27_left = tl.load($(candidate_vwhere_2_ptr) + (((r27_group % $(3)) * $(18496)) + ((r27_row[:, None] % $(136)) * $(136)) + (r27_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r27_right = tl.load($(candidate_vwhere_2_ptr) + (((r27_group % $(3)) * $(18496)) + ((r27_col[None, :] % $(136)) * $(136)) + (r27_lane[:, None] % $(136))), dtype=tl.bfloat16)
      r27_value = tl.dot(r27_left, r27_right)
      r27_value = r27_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_3_ptr) + ((r27_group * $(18496)) + (r27_row[:, None] * $(136)) + r27_col[None, :]), r27_value)
    }
    r28_z = tl.arange($(55488))
    r28_value = (tl.load($(candidate_vbmm_3_ptr) + ((((r28_z / $(18496)) % $(3)) * $(18496)) + (((r28_z / $(136)) % $(136)) * $(136)) + (r28_z % $(136))), dtype=tl.bfloat16) * tl.cast(2.0315, tl.bfloat16))
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(55488)])
    tl.store($(candidate_vmul_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(55488))
    r29_value = (tl.load($(candidate_vwhere_2_ptr) + ((((r29_z / $(18496)) % $(3)) * $(18496)) + (((r29_z / $(136)) % $(136)) * $(136)) + (r29_z % $(136))), dtype=tl.bfloat16) * tl.cast(-4.775, tl.bfloat16))
    r29_value = r29_value.to(tl.bfloat16)
    r29_value = tl.broadcast_to(r29_value, [$(55488)])
    tl.store($(candidate_vmul_5_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(55488))
    r30_value = (tl.load($(candidate_vmul_4_ptr) + ((((r30_z / $(18496)) % $(3)) * $(18496)) + (((r30_z / $(136)) % $(136)) * $(136)) + (r30_z % $(136))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_5_ptr) + ((((r30_z / $(18496)) % $(3)) * $(18496)) + (((r30_z / $(136)) % $(136)) * $(136)) + (r30_z % $(136))), dtype=tl.bfloat16))
    r30_value = r30_value.to(tl.bfloat16)
    r30_value = tl.broadcast_to(r30_value, [$(55488)])
    tl.store($(candidate_vadd_1_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(18496))
    r31_value = (tl.cast(((r31_z / $(136)) % $(136)), tl.int64) <= tl.cast((r31_z % $(136)), tl.int64))
    r31_value = tl.broadcast_to(r31_value, [$(18496)])
    tl.store($(candidate_vle_3_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(55488))
    r32_value = tl.where(tl.load($(candidate_vle_3_ptr) + ((((r32_z / $(136)) % $(136)) * $(136)) + (r32_z % $(136))), dtype=tl.int1), tl.load($(candidate_vadd_1_ptr) + ((((r32_z / $(18496)) % $(3)) * $(18496)) + (((r32_z / $(136)) % $(136)) * $(136)) + (r32_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vadd_1_ptr) + ((((r32_z / $(18496)) % $(3)) * $(18496)) + ((r32_z % $(136)) * $(136)) + ((r32_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r32_value = r32_value.to(tl.bfloat16)
    r32_value = tl.broadcast_to(r32_value, [$(55488)])
    tl.store($(candidate_vwhere_3_ptr) + r32_z, r32_value)
    r33_row = tl.arange($(136))
    r33_col = tl.arange($(264))
    r33_lane = tl.arange($(136))
    for r33_group in range($(3)) {
      r33_left = tl.load($(candidate_vwhere_3_ptr) + (((r33_group % $(3)) * $(18496)) + ((r33_row[:, None] % $(136)) * $(136)) + (r33_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r33_right = tl.load($(candidate_vbaddbmm_ptr) + (((r33_group % $(3)) * $(35904)) + ((r33_lane[:, None] % $(136)) * $(264)) + (r33_col[None, :] % $(264))), dtype=tl.bfloat16)
      r33_value = tl.dot(r33_left, r33_right)
      r33_value = r33_value + tl.load($(candidate_vbaddbmm_ptr) + (((r33_group % $(3)) * $(35904)) + ((r33_row[:, None] % $(136)) * $(264)) + (r33_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
      r33_value = r33_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_1_ptr) + ((r33_group * $(35904)) + (r33_row[:, None] * $(264)) + r33_col[None, :]), r33_value)
    }
    r34_row = tl.arange($(136))
    r34_col = tl.arange($(136))
    r34_lane = tl.arange($(264))
    for r34_group in range($(3)) {
      r34_left = tl.load($(candidate_vbaddbmm_1_ptr) + (((r34_group % $(3)) * $(35904)) + ((r34_row[:, None] % $(136)) * $(264)) + (r34_lane[None, :] % $(264))), dtype=tl.bfloat16)
      r34_right = tl.load($(candidate_vbaddbmm_1_ptr) + (((r34_group % $(3)) * $(35904)) + ((r34_col[None, :] % $(136)) * $(264)) + (r34_lane[:, None] % $(264))), dtype=tl.bfloat16)
      r34_value = tl.dot(r34_left, r34_right)
      r34_value = r34_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_4_ptr) + ((r34_group * $(18496)) + (r34_row[:, None] * $(136)) + r34_col[None, :]), r34_value)
    }
    r35_z = tl.arange($(55488))
    r35_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r35_z / $(18496)) % $(3)) * $(18496)) + (((r35_z / $(136)) % $(136)) * $(136)) + (r35_z % $(136))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r35_value = r35_value.to(tl.bfloat16)
    r35_value = tl.broadcast_to(r35_value, [$(55488)])
    tl.store($(candidate_vmul_6_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(18496))
    r36_value = (tl.cast(((r36_z / $(136)) % $(136)), tl.int64) <= tl.cast((r36_z % $(136)), tl.int64))
    r36_value = tl.broadcast_to(r36_value, [$(18496)])
    tl.store($(candidate_vle_4_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(55488))
    r37_value = tl.where(tl.load($(candidate_vle_4_ptr) + ((((r37_z / $(136)) % $(136)) * $(136)) + (r37_z % $(136))), dtype=tl.int1), tl.load($(candidate_vmul_6_ptr) + ((((r37_z / $(18496)) % $(3)) * $(18496)) + (((r37_z / $(136)) % $(136)) * $(136)) + (r37_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vmul_6_ptr) + ((((r37_z / $(18496)) % $(3)) * $(18496)) + ((r37_z % $(136)) * $(136)) + ((r37_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(55488)])
    tl.store($(candidate_vwhere_4_ptr) + r37_z, r37_value)
    r38_row = tl.arange($(136))
    r38_col = tl.arange($(136))
    r38_lane = tl.arange($(136))
    for r38_group in range($(3)) {
      r38_left = tl.load($(candidate_vwhere_4_ptr) + (((r38_group % $(3)) * $(18496)) + ((r38_row[:, None] % $(136)) * $(136)) + (r38_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r38_right = tl.load($(candidate_vwhere_4_ptr) + (((r38_group % $(3)) * $(18496)) + ((r38_col[None, :] % $(136)) * $(136)) + (r38_lane[:, None] % $(136))), dtype=tl.bfloat16)
      r38_value = tl.dot(r38_left, r38_right)
      r38_value = r38_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r38_group * $(18496)) + (r38_row[:, None] * $(136)) + r38_col[None, :]), r38_value)
    }
    r39_z = tl.arange($(55488))
    r39_value = (tl.load($(candidate_vbmm_5_ptr) + ((((r39_z / $(18496)) % $(3)) * $(18496)) + (((r39_z / $(136)) % $(136)) * $(136)) + (r39_z % $(136))), dtype=tl.bfloat16) * tl.cast(2.0315, tl.bfloat16))
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(55488)])
    tl.store($(candidate_vmul_7_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(55488))
    r40_value = (tl.load($(candidate_vwhere_4_ptr) + ((((r40_z / $(18496)) % $(3)) * $(18496)) + (((r40_z / $(136)) % $(136)) * $(136)) + (r40_z % $(136))), dtype=tl.bfloat16) * tl.cast(-4.775, tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(55488)])
    tl.store($(candidate_vmul_8_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(55488))
    r41_value = (tl.load($(candidate_vmul_7_ptr) + ((((r41_z / $(18496)) % $(3)) * $(18496)) + (((r41_z / $(136)) % $(136)) * $(136)) + (r41_z % $(136))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_8_ptr) + ((((r41_z / $(18496)) % $(3)) * $(18496)) + (((r41_z / $(136)) % $(136)) * $(136)) + (r41_z % $(136))), dtype=tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(55488)])
    tl.store($(candidate_vadd_2_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(18496))
    r42_value = (tl.cast(((r42_z / $(136)) % $(136)), tl.int64) <= tl.cast((r42_z % $(136)), tl.int64))
    r42_value = tl.broadcast_to(r42_value, [$(18496)])
    tl.store($(candidate_vle_5_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(55488))
    r43_value = tl.where(tl.load($(candidate_vle_5_ptr) + ((((r43_z / $(136)) % $(136)) * $(136)) + (r43_z % $(136))), dtype=tl.int1), tl.load($(candidate_vadd_2_ptr) + ((((r43_z / $(18496)) % $(3)) * $(18496)) + (((r43_z / $(136)) % $(136)) * $(136)) + (r43_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vadd_2_ptr) + ((((r43_z / $(18496)) % $(3)) * $(18496)) + ((r43_z % $(136)) * $(136)) + ((r43_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(55488)])
    tl.store($(candidate_vwhere_5_ptr) + r43_z, r43_value)
    r44_row = tl.arange($(136))
    r44_col = tl.arange($(264))
    r44_lane = tl.arange($(136))
    for r44_group in range($(3)) {
      r44_left = tl.load($(candidate_vwhere_5_ptr) + (((r44_group % $(3)) * $(18496)) + ((r44_row[:, None] % $(136)) * $(136)) + (r44_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r44_right = tl.load($(candidate_vbaddbmm_1_ptr) + (((r44_group % $(3)) * $(35904)) + ((r44_lane[:, None] % $(136)) * $(264)) + (r44_col[None, :] % $(264))), dtype=tl.bfloat16)
      r44_value = tl.dot(r44_left, r44_right)
      r44_value = r44_value + tl.load($(candidate_vbaddbmm_1_ptr) + (((r44_group % $(3)) * $(35904)) + ((r44_row[:, None] % $(136)) * $(264)) + (r44_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
      r44_value = r44_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_2_ptr) + ((r44_group * $(35904)) + (r44_row[:, None] * $(264)) + r44_col[None, :]), r44_value)
    }
    r45_row = tl.arange($(136))
    r45_col = tl.arange($(136))
    r45_lane = tl.arange($(264))
    for r45_group in range($(3)) {
      r45_left = tl.load($(candidate_vbaddbmm_2_ptr) + (((r45_group % $(3)) * $(35904)) + ((r45_row[:, None] % $(136)) * $(264)) + (r45_lane[None, :] % $(264))), dtype=tl.bfloat16)
      r45_right = tl.load($(candidate_vbaddbmm_2_ptr) + (((r45_group % $(3)) * $(35904)) + ((r45_col[None, :] % $(136)) * $(264)) + (r45_lane[:, None] % $(264))), dtype=tl.bfloat16)
      r45_value = tl.dot(r45_left, r45_right)
      r45_value = r45_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r45_group * $(18496)) + (r45_row[:, None] * $(136)) + r45_col[None, :]), r45_value)
    }
    r46_z = tl.arange($(55488))
    r46_value = (tl.load($(candidate_vbmm_6_ptr) + ((((r46_z / $(18496)) % $(3)) * $(18496)) + (((r46_z / $(136)) % $(136)) * $(136)) + (r46_z % $(136))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(55488)])
    tl.store($(candidate_vmul_9_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(18496))
    r47_value = (tl.cast(((r47_z / $(136)) % $(136)), tl.int64) <= tl.cast((r47_z % $(136)), tl.int64))
    r47_value = tl.broadcast_to(r47_value, [$(18496)])
    tl.store($(candidate_vle_6_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(55488))
    r48_value = tl.where(tl.load($(candidate_vle_6_ptr) + ((((r48_z / $(136)) % $(136)) * $(136)) + (r48_z % $(136))), dtype=tl.int1), tl.load($(candidate_vmul_9_ptr) + ((((r48_z / $(18496)) % $(3)) * $(18496)) + (((r48_z / $(136)) % $(136)) * $(136)) + (r48_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vmul_9_ptr) + ((((r48_z / $(18496)) % $(3)) * $(18496)) + ((r48_z % $(136)) * $(136)) + ((r48_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(55488)])
    tl.store($(candidate_vwhere_6_ptr) + r48_z, r48_value)
    r49_row = tl.arange($(136))
    r49_col = tl.arange($(136))
    r49_lane = tl.arange($(136))
    for r49_group in range($(3)) {
      r49_left = tl.load($(candidate_vwhere_6_ptr) + (((r49_group % $(3)) * $(18496)) + ((r49_row[:, None] % $(136)) * $(136)) + (r49_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r49_right = tl.load($(candidate_vwhere_6_ptr) + (((r49_group % $(3)) * $(18496)) + ((r49_col[None, :] % $(136)) * $(136)) + (r49_lane[:, None] % $(136))), dtype=tl.bfloat16)
      r49_value = tl.dot(r49_left, r49_right)
      r49_value = r49_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_7_ptr) + ((r49_group * $(18496)) + (r49_row[:, None] * $(136)) + r49_col[None, :]), r49_value)
    }
    r50_z = tl.arange($(55488))
    r50_value = (tl.load($(candidate_vbmm_7_ptr) + ((((r50_z / $(18496)) % $(3)) * $(18496)) + (((r50_z / $(136)) % $(136)) * $(136)) + (r50_z % $(136))), dtype=tl.bfloat16) * tl.cast(2.0315, tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(55488)])
    tl.store($(candidate_vmul_10_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(55488))
    r51_value = (tl.load($(candidate_vwhere_6_ptr) + ((((r51_z / $(18496)) % $(3)) * $(18496)) + (((r51_z / $(136)) % $(136)) * $(136)) + (r51_z % $(136))), dtype=tl.bfloat16) * tl.cast(-4.775, tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(55488)])
    tl.store($(candidate_vmul_11_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(55488))
    r52_value = (tl.load($(candidate_vmul_10_ptr) + ((((r52_z / $(18496)) % $(3)) * $(18496)) + (((r52_z / $(136)) % $(136)) * $(136)) + (r52_z % $(136))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_11_ptr) + ((((r52_z / $(18496)) % $(3)) * $(18496)) + (((r52_z / $(136)) % $(136)) * $(136)) + (r52_z % $(136))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(55488)])
    tl.store($(candidate_vadd_3_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(18496))
    r53_value = (tl.cast(((r53_z / $(136)) % $(136)), tl.int64) <= tl.cast((r53_z % $(136)), tl.int64))
    r53_value = tl.broadcast_to(r53_value, [$(18496)])
    tl.store($(candidate_vle_7_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(55488))
    r54_value = tl.where(tl.load($(candidate_vle_7_ptr) + ((((r54_z / $(136)) % $(136)) * $(136)) + (r54_z % $(136))), dtype=tl.int1), tl.load($(candidate_vadd_3_ptr) + ((((r54_z / $(18496)) % $(3)) * $(18496)) + (((r54_z / $(136)) % $(136)) * $(136)) + (r54_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vadd_3_ptr) + ((((r54_z / $(18496)) % $(3)) * $(18496)) + ((r54_z % $(136)) * $(136)) + ((r54_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(55488)])
    tl.store($(candidate_vwhere_7_ptr) + r54_z, r54_value)
    r55_row = tl.arange($(136))
    r55_col = tl.arange($(264))
    r55_lane = tl.arange($(136))
    for r55_group in range($(3)) {
      r55_left = tl.load($(candidate_vwhere_7_ptr) + (((r55_group % $(3)) * $(18496)) + ((r55_row[:, None] % $(136)) * $(136)) + (r55_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r55_right = tl.load($(candidate_vbaddbmm_2_ptr) + (((r55_group % $(3)) * $(35904)) + ((r55_lane[:, None] % $(136)) * $(264)) + (r55_col[None, :] % $(264))), dtype=tl.bfloat16)
      r55_value = tl.dot(r55_left, r55_right)
      r55_value = r55_value + tl.load($(candidate_vbaddbmm_2_ptr) + (((r55_group % $(3)) * $(35904)) + ((r55_row[:, None] % $(136)) * $(264)) + (r55_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
      r55_value = r55_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_3_ptr) + ((r55_group * $(35904)) + (r55_row[:, None] * $(264)) + r55_col[None, :]), r55_value)
    }
    r56_row = tl.arange($(136))
    r56_col = tl.arange($(136))
    r56_lane = tl.arange($(264))
    for r56_group in range($(3)) {
      r56_left = tl.load($(candidate_vbaddbmm_3_ptr) + (((r56_group % $(3)) * $(35904)) + ((r56_row[:, None] % $(136)) * $(264)) + (r56_lane[None, :] % $(264))), dtype=tl.bfloat16)
      r56_right = tl.load($(candidate_vbaddbmm_3_ptr) + (((r56_group % $(3)) * $(35904)) + ((r56_col[None, :] % $(136)) * $(264)) + (r56_lane[:, None] % $(264))), dtype=tl.bfloat16)
      r56_value = tl.dot(r56_left, r56_right)
      r56_value = r56_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_8_ptr) + ((r56_group * $(18496)) + (r56_row[:, None] * $(136)) + r56_col[None, :]), r56_value)
    }
    r57_z = tl.arange($(55488))
    r57_value = (tl.load($(candidate_vbmm_8_ptr) + ((((r57_z / $(18496)) % $(3)) * $(18496)) + (((r57_z / $(136)) % $(136)) * $(136)) + (r57_z % $(136))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(55488)])
    tl.store($(candidate_vmul_12_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(18496))
    r58_value = (tl.cast(((r58_z / $(136)) % $(136)), tl.int64) <= tl.cast((r58_z % $(136)), tl.int64))
    r58_value = tl.broadcast_to(r58_value, [$(18496)])
    tl.store($(candidate_vle_8_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(55488))
    r59_value = tl.where(tl.load($(candidate_vle_8_ptr) + ((((r59_z / $(136)) % $(136)) * $(136)) + (r59_z % $(136))), dtype=tl.int1), tl.load($(candidate_vmul_12_ptr) + ((((r59_z / $(18496)) % $(3)) * $(18496)) + (((r59_z / $(136)) % $(136)) * $(136)) + (r59_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vmul_12_ptr) + ((((r59_z / $(18496)) % $(3)) * $(18496)) + ((r59_z % $(136)) * $(136)) + ((r59_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(55488)])
    tl.store($(candidate_vwhere_8_ptr) + r59_z, r59_value)
    r60_row = tl.arange($(136))
    r60_col = tl.arange($(136))
    r60_lane = tl.arange($(136))
    for r60_group in range($(3)) {
      r60_left = tl.load($(candidate_vwhere_8_ptr) + (((r60_group % $(3)) * $(18496)) + ((r60_row[:, None] % $(136)) * $(136)) + (r60_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r60_right = tl.load($(candidate_vwhere_8_ptr) + (((r60_group % $(3)) * $(18496)) + ((r60_col[None, :] % $(136)) * $(136)) + (r60_lane[:, None] % $(136))), dtype=tl.bfloat16)
      r60_value = tl.dot(r60_left, r60_right)
      r60_value = r60_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_9_ptr) + ((r60_group * $(18496)) + (r60_row[:, None] * $(136)) + r60_col[None, :]), r60_value)
    }
    r61_z = tl.arange($(55488))
    r61_value = (tl.load($(candidate_vbmm_9_ptr) + ((((r61_z / $(18496)) % $(3)) * $(18496)) + (((r61_z / $(136)) % $(136)) * $(136)) + (r61_z % $(136))), dtype=tl.bfloat16) * tl.cast(2.0315, tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(55488)])
    tl.store($(candidate_vmul_13_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(55488))
    r62_value = (tl.load($(candidate_vwhere_8_ptr) + ((((r62_z / $(18496)) % $(3)) * $(18496)) + (((r62_z / $(136)) % $(136)) * $(136)) + (r62_z % $(136))), dtype=tl.bfloat16) * tl.cast(-4.775, tl.bfloat16))
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(55488)])
    tl.store($(candidate_vmul_14_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(55488))
    r63_value = (tl.load($(candidate_vmul_13_ptr) + ((((r63_z / $(18496)) % $(3)) * $(18496)) + (((r63_z / $(136)) % $(136)) * $(136)) + (r63_z % $(136))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_14_ptr) + ((((r63_z / $(18496)) % $(3)) * $(18496)) + (((r63_z / $(136)) % $(136)) * $(136)) + (r63_z % $(136))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(55488)])
    tl.store($(candidate_vadd_4_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(18496))
    r64_value = (tl.cast(((r64_z / $(136)) % $(136)), tl.int64) <= tl.cast((r64_z % $(136)), tl.int64))
    r64_value = tl.broadcast_to(r64_value, [$(18496)])
    tl.store($(candidate_vle_9_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(55488))
    r65_value = tl.where(tl.load($(candidate_vle_9_ptr) + ((((r65_z / $(136)) % $(136)) * $(136)) + (r65_z % $(136))), dtype=tl.int1), tl.load($(candidate_vadd_4_ptr) + ((((r65_z / $(18496)) % $(3)) * $(18496)) + (((r65_z / $(136)) % $(136)) * $(136)) + (r65_z % $(136))), dtype=tl.bfloat16), tl.load($(candidate_vadd_4_ptr) + ((((r65_z / $(18496)) % $(3)) * $(18496)) + ((r65_z % $(136)) * $(136)) + ((r65_z / $(136)) % $(136))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(55488)])
    tl.store($(candidate_vwhere_9_ptr) + r65_z, r65_value)
    r66_row = tl.arange($(136))
    r66_col = tl.arange($(264))
    r66_lane = tl.arange($(136))
    for r66_group in range($(3)) {
      r66_left = tl.load($(candidate_vwhere_9_ptr) + (((r66_group % $(3)) * $(18496)) + ((r66_row[:, None] % $(136)) * $(136)) + (r66_lane[None, :] % $(136))), dtype=tl.bfloat16)
      r66_right = tl.load($(candidate_vbaddbmm_3_ptr) + (((r66_group % $(3)) * $(35904)) + ((r66_lane[:, None] % $(136)) * $(264)) + (r66_col[None, :] % $(264))), dtype=tl.bfloat16)
      r66_value = tl.dot(r66_left, r66_right)
      r66_value = r66_value + tl.load($(candidate_vbaddbmm_3_ptr) + (((r66_group % $(3)) * $(35904)) + ((r66_row[:, None] % $(136)) * $(264)) + (r66_col[None, :] % $(264))), dtype=tl.bfloat16) * 3.4445
      r66_value = r66_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_4_ptr) + ((r66_group * $(35904)) + (r66_row[:, None] * $(264)) + r66_col[None, :]), r66_value)
    }
    r67_z = tl.arange($(107712))
    r67_value = tl.load($(parameter0_ptr) + ((((r67_z / $(35904)) % $(3)) * $(35904)) + (((r67_z / $(136)) % $(264)) * $(136)) + (r67_z % $(136))), dtype=tl.bfloat16)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(107712)])
    tl.store($(candidate_vto_copy_4_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(107712))
    r68_value = (tl.load($(candidate_vto_copy_4_ptr) + ((((r68_z / $(35904)) % $(3)) * $(35904)) + (((r68_z / $(136)) % $(264)) * $(136)) + (r68_z % $(136))), dtype=tl.float32) * 0.9999)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(107712)])
    tl.store($(candidate_vmul_15_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(107712))
    r69_value = tl.load($(candidate_vmul_15_ptr) + ((((r69_z / $(35904)) % $(3)) * $(35904)) + (((r69_z / $(136)) % $(264)) * $(136)) + (r69_z % $(136))), dtype=tl.float32)
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(107712)])
    tl.store($(candidate_vto_copy_5_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(107712))
    r70_value = tl.load($(candidate_vto_copy_5_ptr) + ((((r70_z / $(35904)) % $(3)) * $(35904)) + (((r70_z / $(136)) % $(264)) * $(136)) + (r70_z % $(136))), dtype=tl.bfloat16)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(107712)])
    tl.store($(candidate_vto_copy_6_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(107712))
    r71_value = tl.load($(candidate_vbaddbmm_4_ptr) + ((((r71_z / $(35904)) % $(3)) * $(35904)) + ((r71_z % $(136)) * $(264)) + ((r71_z / $(136)) % $(264))), dtype=tl.bfloat16)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(107712)])
    tl.store($(candidate_vto_copy_7_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(107712))
    r72_value = tl.fma(tl.cast(-0.03249615361854385, tl.float32), tl.cast(tl.load($(candidate_vto_copy_7_ptr) + ((((r72_z / $(35904)) % $(3)) * $(35904)) + (((r72_z / $(136)) % $(264)) * $(136)) + (r72_z % $(136))), dtype=tl.float32), tl.float32), tl.cast(tl.load($(candidate_vto_copy_6_ptr) + ((((r72_z / $(35904)) % $(3)) * $(35904)) + (((r72_z / $(136)) % $(264)) * $(136)) + (r72_z % $(136))), dtype=tl.float32), tl.float32))
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(107712)])
    tl.store($(candidate_vfma_2_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(107712))
    r73_value = tl.load($(candidate_vfma_2_ptr) + ((((r73_z / $(35904)) % $(3)) * $(35904)) + (((r73_z / $(136)) % $(264)) * $(136)) + (r73_z % $(136))), dtype=tl.float32)
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(107712)])
    tl.store($(candidate_vto_copy_8_ptr) + r73_z, r73_value)
    z = tl.arange($(107712))
    value = tl.load($(candidate_vto_copy_8_ptr) + ((((z / $(35904)) % $(3)) * $(35904)) + (((z / $(136)) % $(264)) * $(136)) + (z % $(136))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(107712)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_muon_fused_vs_loop
