import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Target

/-! Complete candidate JIT quotation. Signed int64 conversions and range bounds
are retained. The early return is represented by its equivalent active guard;
implicit atomic operand conversion is explicit. -/
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.LegacySurface
open Lean VeriTile.Triton
syntax "physicalShrinkSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(physicalShrinkSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scalarLeft do
      Macro.throwError "expected original or scalarLeft"
    let scale : TSyntax `tritonExpr ←
      if version.getId == `scalarLeft then `(tritonExpr| scaling * accumulator)
      else `(tritonExpr| accumulator * scaling)
    `(fun (out_dtype : FloatDType) (input_ptr lora_ptr out_ptr : RegionName)
        (N K : Nat) (lora_indices : Region .int) (scaling : ℝ)
        (xm_stride _xk_stride l0_stride lora_k_stride lora_n_stride cm_stride
          cn_stride BLOCK_N BLOCK_K SPLIT_K : Nat) =>
    ComputeKernel.mk [input_ptr, lora_ptr, out_ptr, Region.cast lora_indices] [out_ptr]
    ((triton {
      tl.region input_ptr = $storage, lora_ptr = $storage, out_ptr = OUT_DTYPE
      scaling = $((0 : ℝ)).to(tl.float32)
      pid_sk = tl.program_id(axis=0).to(tl.int64)
      cur_batch = tl.program_id(axis=1).to(tl.int64)
      lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
      if lora_index != $((-1 : Int)) {
        offset_n = tl.arange(0, $(BLOCK_N)).to(tl.int64)
        offset_k = tl.arange(0, $(BLOCK_K)).to(tl.int64) + pid_sk * $(BLOCK_K)
        a_ptr = input_ptr + cur_batch * $(xm_stride)
        b_ptr = lora_ptr + $(l0_stride) * lora_index
        accumulator = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
        for k in range(($((0 : Int))).to(tl.int64), ($((K : Int))).to(tl.int64), $(BLOCK_K * SPLIT_K)) {
          current_k = k + offset_k
          current_k_c = tl.max_contiguous(current_k, $(BLOCK_K))
          tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < $(K), other=0.0)
          b_ptr_mask = (offset_n[:, None] < $(N)) & (current_k[None, :] < $(K))
          tiled_b = tl.load(
            b_ptr + offset_n[:, None] * $(lora_k_stride) +
              current_k[None, :] * $(lora_n_stride),
            mask=b_ptr_mask, other=0.0)
          accumulator += tl.sum(tiled_a * tiled_b, 1)
        }
        accumulator = $scale
        offset_cn = tl.arange(0, $(BLOCK_N)).to(tl.int64)
        c_ptr = out_ptr + cur_batch * $(cm_stride) + offset_cn * $(cn_stride)
        c_mask = offset_cn < $(N)
        if $(SPLIT_K) == $(1) {
          tl.store(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
        } else {
          tl.atomic_add(c_ptr, (accumulator).to(OUT_DTYPE), mask=c_mask)
        }
      }
    }).surfaceBody.drop 1))


def source (h : Call) : ComputeKernel :=
  match h.storage with
  | .fp16 => (physicalShrinkSource%[tl.float16, original]) h.outputType "X" "W" "Y"
      h.N h.K ⟨"Indices"⟩ h.scaling h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64
  | .bf16 => (physicalShrinkSource%[tl.bfloat16, original]) h.outputType "X" "W" "Y"
      h.N h.K ⟨"Indices"⟩ h.scaling h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.BN 256 64

#auditModuleAxioms
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.LegacySurface
