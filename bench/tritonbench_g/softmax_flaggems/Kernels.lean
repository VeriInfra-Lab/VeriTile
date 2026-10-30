import bench.tritonbench_g.softmax_flaggems.Sources.Kernels

/-! Complete ordinary quotations for the selected FP32 finite-seed source and
its existing address-commuted candidate. All original one/multiple-tile branches,
reduction seeds, loads, FP32 operations, casts and store order are preserved.
Proofs/ReadableSource.lean proves complete function equality to the old definitions.
-/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2
open Lean VeriTile Triton
set_option linter.unusedVariables false
def forward_noninner :=
  fun (output_ptr input_ptr : RegionName)
        (M N K TILE_N TILE_K : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region output_ptr = tl.float32, input_ptr = tl.float32
  pid_k = tl.program_id(1)
  pid_m = tl.program_id(0)
  k_offsets = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))
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
    tl.store(output_ptrs, (out).to(tl.float32), mask=mask)
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
      tl.store(output_ptr + offsets, (o).to(tl.float32), mask=mask)
    }
  }
}

def forward_inner :=
  fun (output_ptr input_ptr : RegionName)
        (M N TILE_N : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region output_ptr = tl.float32, input_ptr = tl.float32
  pid_m = tl.program_id(0)
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(inp, 0)
    e = tl.exp(inp - m)
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, (out).to(tl.float32), mask=mask)
  } else {
    m = tl.full([$(TILE_N)], value=-340282346638528859811704183484516925440.0, dtype=tl.float32)
    z = tl.full([$(TILE_N)], value=0.0, dtype=tl.float32)
    input_ptr += pid_m * $(N)
    output_ptr += pid_m * $(N)
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
      tl.store(output_ptr + n_offsets, (o).to(tl.float32), mask=mask)
    }
    for start_n in range($(TILE_N), $(N), $(TILE_N)) {
      n_offsets = previous_multiple - start_n + tl.arange(0, $(TILE_N))
      inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
      o = tl.exp(inp - m) / z
      tl.store(output_ptr + n_offsets, (o).to(tl.float32))
    }
  }
}

def backward_noninner :=
  fun (out_ptr out_grad_ptr in_grad_ptr : RegionName)
        (M N K TILE_N TILE_K : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region out_ptr = tl.float32, out_grad_ptr = tl.float32, in_grad_ptr = tl.float32
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  offsets_k = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))
  if ONE_TILE_PER_CTA {
    offsets_n = tl.arange(0, $(TILE_N))
    offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
    mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
    out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
    out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
    scale = tl.sum(out_tile * out_grad_tile, axis=0)
    in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
    tl.store(in_grad_ptr + offsets, (in_grad_tile).to(tl.float32), mask=mask)
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
      tl.store(in_grad_ptr + offsets, (in_grad_tile).to(tl.float32), mask=mask)
      offsets_n += $(TILE_N)
      offsets += $(TILE_N) * $(K)
    }
  }
}

def backward_inner :=
  fun (out_ptr out_grad_ptr in_grad_ptr : RegionName)
        (M N TILE_M TILE_N : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region out_ptr = tl.float32, out_grad_ptr = tl.float32, in_grad_ptr = tl.float32
  pid_m = tl.program_id(0)
  m_offsets = pid_m * $(TILE_M) + tl.arange(0, $(TILE_M))
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offsets = m_offsets[:, None] * $(N) + n_offsets
    mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
    out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
    out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
    scale = tl.sum(out_tile * out_grad_tile, 1)
    in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
    tl.store(in_grad_ptr + offsets, (in_grad_tile).to(tl.float32), mask=mask)
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
      tl.store(in_grad_ptr + offsets, (in_grad_tile).to(tl.float32), mask=mask)
      n_offsets += $(TILE_N)
      offsets += $(TILE_N)
    }
  }
}

namespace AddressCandidate

/-- Existing exact natural-address multiplication commutations only. -/
def forward_noninner :=
  fun (output_ptr input_ptr : RegionName)
        (M N K TILE_N TILE_K : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region output_ptr = tl.float32, input_ptr = tl.float32
  pid_k = tl.program_id(1)
  pid_m = tl.program_id(0)
  k_offsets = $(TILE_K) * pid_k + tl.arange(0, $(TILE_K))
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
    tl.store(output_ptrs, (out).to(tl.float32), mask=mask)
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
      tl.store(output_ptr + offsets, (o).to(tl.float32), mask=mask)
    }
  }
}

def forward_inner :=
  fun (output_ptr input_ptr : RegionName)
        (M N TILE_N : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region output_ptr = tl.float32, input_ptr = tl.float32
  pid_m = tl.program_id(0)
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) + n_offsets
    input_ptrs = input_ptr + offset
    mask = n_offsets < $(N)
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(inp, 0)
    e = tl.exp(inp - m)
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, (out).to(tl.float32), mask=mask)
  } else {
    m = tl.full([$(TILE_N)], value=-340282346638528859811704183484516925440.0, dtype=tl.float32)
    z = tl.full([$(TILE_N)], value=0.0, dtype=tl.float32)
    input_ptr += $(N) * pid_m
    output_ptr += $(N) * pid_m
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
      tl.store(output_ptr + n_offsets, (o).to(tl.float32), mask=mask)
    }
    for start_n in range($(TILE_N), $(N), $(TILE_N)) {
      n_offsets = previous_multiple - start_n + tl.arange(0, $(TILE_N))
      inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
      o = tl.exp(inp - m) / z
      tl.store(output_ptr + n_offsets, (o).to(tl.float32))
    }
  }
}

def backward_noninner :=
  fun (out_ptr out_grad_ptr in_grad_ptr : RegionName)
        (M N K TILE_N TILE_K : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region out_ptr = tl.float32, out_grad_ptr = tl.float32, in_grad_ptr = tl.float32
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  offsets_k = $(TILE_K) * pid_k + tl.arange(0, $(TILE_K))
  if ONE_TILE_PER_CTA {
    offsets_n = tl.arange(0, $(TILE_N))
    offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
    mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
    out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
    out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
    scale = tl.sum(out_tile * out_grad_tile, axis=0)
    in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
    tl.store(in_grad_ptr + offsets, (in_grad_tile).to(tl.float32), mask=mask)
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
      tl.store(in_grad_ptr + offsets, (in_grad_tile).to(tl.float32), mask=mask)
      offsets_n += $(TILE_N)
      offsets += $(TILE_N) * $(K)
    }
  }
}

def backward_inner :=
  fun (out_ptr out_grad_ptr in_grad_ptr : RegionName)
        (M N TILE_M TILE_N : Nat) (ONE_TILE_PER_CTA : Bool) => triton {
  tl.region out_ptr = tl.float32, out_grad_ptr = tl.float32, in_grad_ptr = tl.float32
  pid_m = tl.program_id(0)
  m_offsets = $(TILE_M) * pid_m + tl.arange(0, $(TILE_M))
  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offsets = m_offsets[:, None] * $(N) + n_offsets
    mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
    out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
    out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
    scale = tl.sum(out_tile * out_grad_tile, 1)
    in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
    tl.store(in_grad_ptr + offsets, (in_grad_tile).to(tl.float32), mask=mask)
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
      tl.store(in_grad_ptr + offsets, (in_grad_tile).to(tl.float32), mask=mask)
      n_offsets += $(TILE_N)
      offsets += $(TILE_N)
    }
  }
}

end AddressCandidate
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2
