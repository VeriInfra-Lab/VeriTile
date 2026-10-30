import bench.tritonbench_g.softmax_flaggems.Kernels
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.Source
open Lean VeriTile Triton
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems

syntax "repairFlaggems2%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(repairFlaggems2%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `address_commuted do
      Macro.throwError "expected original or address_commuted"
    let programBase : TSyntax `tritonExpr ←
      if version.getId == `address_commuted then
        `(tritonExpr| $(TILE_K) * pid_k)
      else `(tritonExpr| pid_k * $(TILE_K))
    `(fun (out_ptr out_grad_ptr in_grad_ptr : RegionName)
        (M N K TILE_N TILE_K : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region out_ptr = $inputType, out_grad_ptr = $inputType, in_grad_ptr = $outputType
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  offsets_k = $programBase + tl.arange(0, $(TILE_K))
  if ONE_TILE_PER_CTA {
    offsets_n = tl.arange(0, $(TILE_N))
    offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
    mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
    out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
    out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
    scale = tl.sum(out_tile * out_grad_tile, axis=0)
    in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
    tl.store(in_grad_ptr + offsets, (in_grad_tile).to($outputType:tritonDType), mask=mask)
  } else {
    offsets_n = tl.arange(0, $(TILE_N))
    offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
    scale = tl.zeros([$(TILE_N), $(TILE_K)], dtype=tl.float32)
    for unused_start_n in range($(0), $(N), $(TILE_N)) {
      mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
      out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
      out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
      scale += out_tile * out_grad_tile
      offsets_n += $(TILE_N)
      offsets += $(TILE_N) * $(K)
    }
    scale = tl.sum(scale, axis=0)
    offsets_n = tl.arange(0, $(TILE_N))
    offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
    for unused_start_n in range($(0), $(N), $(TILE_N)) {
      mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
      out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
      out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
      in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
      tl.store(in_grad_ptr + offsets, (in_grad_tile).to($outputType:tritonDType), mask=mask)
      offsets_n += $(TILE_N)
      offsets += $(TILE_N) * $(K)
    }
  }
})

def softmax_backward_kernel_non_inner_fp32 := repairFlaggems2%[tl.float32, tl.float32, original]

syntax "repairFlaggems3%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(repairFlaggems3%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `address_commuted do
      Macro.throwError "expected original or address_commuted"
    let programBase : TSyntax `tritonExpr ←
      if version.getId == `address_commuted then
        `(tritonExpr| $(TILE_M) * pid_m)
      else `(tritonExpr| pid_m * $(TILE_M))
    `(fun (out_ptr out_grad_ptr in_grad_ptr : RegionName)
        (M N TILE_M TILE_N : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region out_ptr = $inputType, out_grad_ptr = $inputType, in_grad_ptr = $outputType
  pid_m = tl.program_id(0)
  m_offsets = $programBase + tl.arange(0, $(TILE_M))
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offsets = m_offsets[:, None] * $(N) + n_offsets
    mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
    out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
    out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
    scale = tl.sum(out_tile * out_grad_tile, 1)
    in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
    tl.store(in_grad_ptr + offsets, (in_grad_tile).to($outputType:tritonDType), mask=mask)
  } else {
    scale = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
    n_offsets = tl.arange(0, $(TILE_N))
    offsets = m_offsets[:, None] * $(N) + n_offsets
    for unused_start_n in range($(0), $(N), $(TILE_N)) {
      mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
      out_tile = tl.load(out_ptr + offsets, mask=mask, other=0, eviction_policy="evict_last")
      out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
      scale += out_tile * out_grad_tile
      n_offsets += $(TILE_N)
      offsets += $(TILE_N)
    }
    scale = tl.sum(scale, 1)
    n_offsets = tl.arange(0, $(TILE_N))
    offsets = m_offsets[:, None] * $(N) + n_offsets
    for unused_start_n in range($(0), $(N), $(TILE_N)) {
      mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
      out_tile = tl.load(out_ptr + offsets, mask=mask, other=0, eviction_policy="evict_first")
      out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
      in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
      tl.store(in_grad_ptr + offsets, (in_grad_tile).to($outputType:tritonDType), mask=mask)
      n_offsets += $(TILE_N)
      offsets += $(TILE_N)
    }
  }
})

def softmax_backward_kernel_inner_fp32 := repairFlaggems3%[tl.float32, tl.float32, original]


/-- Forward kernels are unchanged. Only the selected source FP32 profile is
claimed here; wider storage support remains outside this repair review. -/
abbrev forward_inner := FullSource.softmax_kernel_inner .fp32 .fp32
abbrev forward_noninner := FullSource.softmax_kernel_non_inner .fp32 .fp32

end VeriTile.Bench.Repairs.SoftmaxFlaggems.Source
