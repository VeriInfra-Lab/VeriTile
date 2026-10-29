import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Complete original and existing shifted-offset Python bodies. The unchanged
wrapper supplies FP32 pointer/scalar types, runtime strides and sizes, and
constexpr tile/flag values outside the quotation. Every cast, boundary check,
branch, dot option and candidate static assertion remains in the source. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton
noncomputable section

def callSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("h",.fp32),("o",.fp32)]
  scalars := [("s_qh",.integer),("s_qt",.integer),("s_qd",.integer),("s_hh",.integer),("s_ht",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BT","BD","NT","STORE","IFCOND"]
  readPorts := ["q","k","v","h","o"]

def source
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) : ComputeKernel :=
  with_triton_call callSignature call_args [
    s_qh := s_qh,
    s_qt := s_qt,
    s_qd := s_qd,
    s_hh := s_hh,
    s_ht := s_ht,
    T := T,
    scale := scale] in triton {
    i_bh = tl.program_id(0).to(tl.int64)
    b_h = tl.zeros([BD, BD], dtype=tl.float32)
    for i in range(0, NT):
        # make_block_ptr offsets are signed32. The host checks both padded
        # temporal coordinates and every pre-update H snapshot coordinate.
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i * BT), (BD, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (NT * BD, BD), (s_ht, 1), (i * BD, 0), (BD, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        if STORE:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty))
        b_q = tl.load(p_q, boundary_check=(0,), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.load(p_k, boundary_check=(1,), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0,), padding_option="zero")
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        if IFCOND:
            if i == 0:
                b_h = tl.dot(b_k, b_v, allow_tf32=False)
            else:
                b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
                b_h += tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h += tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
  }

def shiftedSource
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BD NT : Nat) (STORE IFCOND : Bool) : ComputeKernel :=
  let BT : Nat := 32
  with_triton_call callSignature call_args [
    s_qh := s_qh,
    s_qt := s_qt,
    s_qd := s_qd,
    s_hh := s_hh,
    s_ht := s_ht,
    T := T,
    scale := scale] in triton {
    tl.static_assert(BT == 32)
    i_bh = tl.program_id(0).to(tl.int64)
    b_h = tl.zeros([BD, BD], dtype=tl.float32)
    for i in range(0, NT):
        # make_block_ptr offsets are signed32. The host checks both padded
        # temporal coordinates and every pre-update H snapshot coordinate.
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i << 5, 0), (BT, BD), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i << 5), (BD, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i << 5, 0), (BT, BD), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (NT * BD, BD), (s_ht, 1), (i * BD, 0), (BD, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i << 5, 0), (BT, BD), (1, 0))
        if STORE:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty))
        b_q = tl.load(p_q, boundary_check=(0,), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.load(p_k, boundary_check=(1,), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0,), padding_option="zero")
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        if IFCOND:
            if i == 0:
                b_h = tl.dot(b_k, b_v, allow_tf32=False)
            else:
                b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
                b_h += tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h += tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
  }

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
