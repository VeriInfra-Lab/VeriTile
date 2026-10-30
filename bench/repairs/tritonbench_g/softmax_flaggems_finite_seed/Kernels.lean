import bench.tritonbench_g.softmax_flaggems.Kernels
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Source
open Lean VeriTile Triton

syntax "finiteSeedFlaggems0%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(finiteSeedFlaggems0%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `address_commuted do
      Macro.throwError "expected original or address_commuted"
    let programBase : TSyntax `tritonExpr ←
      if version.getId == `address_commuted then
        `(tritonExpr| $(TILE_K) * pid_k)
      else `(tritonExpr| pid_k * $(TILE_K))
    `(fun (output_ptr input_ptr : RegionName)
        (M N K TILE_N TILE_K : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region output_ptr = $outputType, input_ptr = $inputType
  pid_k = tl.program_id(1)
  pid_m = tl.program_id(0)
  k_offsets = $programBase + tl.arange(0, $(TILE_K))
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
    mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    m = tl.max(inp, 0)
    e = tl.exp(inp - m[None, :])
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, (out).to($outputType:tritonDType), mask=mask)
  } else {
    m = tl.full([$(TILE_N), $(TILE_K)], value=-340282346638528859811704183484516925440.0, dtype=tl.float32)
    z = tl.full([$(TILE_N), $(TILE_K)], value=0.0, dtype=tl.float32)
    for start_n in range($(0), $(N), $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N))
      offsets = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
      mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
      inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
      m_new = tl.maximum(m, inp)
      alpha = tl.exp(m - m_new)
      z = z * alpha + tl.exp(inp - m_new)
      m = m_new
    }
    m_reduced = tl.max(m, 0)
    z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)
    m = m_reduced
    previous_multiple = tl.cdiv($(N), $(TILE_N)) * $(TILE_N) - $(TILE_N)
    for start_n in range($(0), $(N), $(TILE_N)) {
      n_offsets = previous_multiple - start_n + tl.arange(0, $(TILE_N))
      offsets = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
      mask = (n_offsets[:, None] < $(N)) & (k_offsets[None, :] < $(K))
      inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
      o = tl.exp(inp - m[None, :]) / z[None, :]
      tl.store(output_ptr + offsets, (o).to($outputType:tritonDType), mask=mask)
    }
  }
})

abbrev forward_noninner := finiteSeedFlaggems0%[tl.float32, tl.float32, original]

syntax "finiteSeedFlaggems1%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(finiteSeedFlaggems1%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `address_commuted do
      Macro.throwError "expected original or address_commuted"
    let programBase : TSyntax `tritonExpr ←
      if version.getId == `address_commuted then
        `(tritonExpr| $(N) * pid_m)
      else `(tritonExpr| pid_m * $(N))
    `(fun (output_ptr input_ptr : RegionName)
        (M N TILE_N : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region output_ptr = $outputType, input_ptr = $inputType
  pid_m = tl.program_id(0)
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to($outputType:tritonDType)
    m = tl.max(inp, 0)
    e = tl.exp(inp - m)
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, (out).to($outputType:tritonDType), mask=mask)
  } else {
    m = tl.full([$(TILE_N)], value=-340282346638528859811704183484516925440.0, dtype=tl.float32)
    z = tl.full([$(TILE_N)], value=0.0, dtype=tl.float32)
    input_ptr += $programBase
    output_ptr += $programBase
    previous_multiple = tl.cdiv($(N), $(TILE_N)) * $(TILE_N) - $(TILE_N)
    for start_n in range($(0), previous_multiple, $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N))
      inp = tl.load(input_ptr + n_offsets)
      m_new = tl.maximum(m, inp)
      z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
      m = m_new
    }
    for start_n in range(previous_multiple, $(N), $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N))
      mask = n_offsets < $(N)
      inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
      m_new = tl.maximum(m, inp)
      z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
      m = m_new
    }
    m_reduced = tl.max(m, 0)
    z = tl.sum(z * tl.exp(m - m_reduced), 0)
    m = m_reduced
    previous_multiple = tl.cdiv($(N), $(TILE_N)) * $(TILE_N) - $(TILE_N)
    for start_n in range($(0), $(TILE_N), $(TILE_N)) {
      n_offsets = previous_multiple - start_n + tl.arange(0, $(TILE_N))
      mask = n_offsets < $(N)
      inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"), eviction_policy="evict_first")
      o = tl.exp(inp - m) / z
      tl.store(output_ptr + n_offsets, (o).to($outputType:tritonDType), mask=mask)
    }
    for start_n in range($(TILE_N), $(N), $(TILE_N)) {
      n_offsets = previous_multiple - start_n + tl.arange(0, $(TILE_N))
      inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
      o = tl.exp(inp - m) / z
      tl.store(output_ptr + n_offsets, (o).to($outputType:tritonDType))
    }
  }
})

abbrev forward_inner := finiteSeedFlaggems1%[tl.float32, tl.float32, original]

syntax "finiteSeedFlaggems2%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(finiteSeedFlaggems2%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
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

def softmax_backward_kernel_non_inner_fp32 := finiteSeedFlaggems2%[tl.float32, tl.float32, original]

syntax "finiteSeedFlaggems3%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(finiteSeedFlaggems3%[$inputType:tritonDType, $outputType:tritonDType, $version:ident]) => do
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

def softmax_backward_kernel_inner_fp32 := finiteSeedFlaggems3%[tl.float32, tl.float32, original]


end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Source
