import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Source

/-! Complete original Python function bodies with caller storage/runtime metadata
outside. Implicit destination conversions are inferred from the pointers.
The authored Python counterpart preserves the existing index-sharing candidate. -/
namespace VeriTile.Bench.Repairs.RopeTransform.Kernels
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("q_ptr",.inputFloat), ("k_ptr",.inputFloat), ("cos",.outputFloat), ("sin",.outputFloat)]
  scalars := [("q_row_stride",.integer), ("k_row_stride",.integer), ("cos_row_stride",.integer), ("sin_row_stride",.integer), ("sl",.integer)]
  constexpr := ["bs","n_qh","n_kh","hd","pad_n_qh","pad_n_kh","pad_hd","BLOCK_SIZE","BACKWARD_PASS"]
  readPorts := ["q_ptr","k_ptr","cos","sin"]
  writePorts := ["q_ptr","k_ptr"]

def source (dtype out_dtype : FloatDType)
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) :
    ComputeKernel :=
  with_triton_call callSignature call_args [q_row_stride := q_row_stride, k_row_stride := k_row_stride, cos_row_stride := cos_row_stride, sin_row_stride := sin_row_stride, sl := sl] in triton {
    pid = tl.program_id(0)

    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    cos_row_idx = pid % (sl)
    cos = cos + cos_row_idx * cos_row_stride
    sin = sin + cos_row_idx * sin_row_stride
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)

    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    if not BACKWARD_PASS:
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
  }

def indexShared (dtype out_dtype : FloatDType)
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) :
    ComputeKernel :=
  with_triton_call callSignature call_args [q_row_stride := q_row_stride, k_row_stride := k_row_stride, cos_row_stride := cos_row_stride, sin_row_stride := sin_row_stride, sl := sl] in triton {
    pid = tl.program_id(0)

    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    cos_row_idx = pid % (sl)
    cos = cos + cos_row_idx * cos_row_stride
    sin = sin + cos_row_idx * sin_row_stride
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)

    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + cos_offsets[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + cos_offsets[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & cos_mask[None, :]
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & cos_mask[None, :]
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    if not BACKWARD_PASS:
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
  }

end VeriTile.Bench.Repairs.RopeTransform.Kernels
