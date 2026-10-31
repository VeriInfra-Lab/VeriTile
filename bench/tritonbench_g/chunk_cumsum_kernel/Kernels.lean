import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Source

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel
open VeriTile.Triton

def callSignature (dtype out_dtype : FloatDType) : CallSignature where
  pointers := []
  floatingPointers := [("s",dtype),("o",out_dtype)]
  scalars := []
  constexpr := ["T","BT"]

def repairedKernel (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  with_triton_signature (callSignature dtype out_dtype) in triton {
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cast(0, tl.int64), tl.cast(tl.cdiv(T, BT), tl.int64)):
        times = i_t * BT + tl.arange(0, BT).to(tl.int64)
        p_s = s + i_bh.to(tl.int64) * T + times
        p_o = o + i_bh.to(tl.int64) * T + times
        b_s = tl.load(p_s, mask=times < T).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0) + b_z[None]
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=times < T)
  }

def sumLeftKernel (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  with_triton_signature (callSignature dtype out_dtype) in triton {
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cast(0, tl.int64), tl.cast(tl.cdiv(T, BT), tl.int64)):
        times = i_t * BT + tl.arange(0, BT).to(tl.int64)
        p_s = s + i_bh.to(tl.int64) * T + times
        p_o = o + i_bh.to(tl.int64) * T + times
        b_s = tl.load(p_s, mask=times < T).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0) + b_z[None]
        b_zz = tl.sum(b_s, axis=0)
        b_z = b_zz + b_z
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=times < T)
  }

inductive Config where
  | bt16w2 | bt32w4 | bt32w2 | bt64w8 | bt64w4
  deriving DecidableEq

def Config.block : Config → Nat
  | .bt16w2 => 16 | .bt32w4 | .bt32w2 => 32 | .bt64w8 | .bt64w4 => 64

def Config.warps : Config → Nat
  | .bt16w2 | .bt32w2 => 2 | .bt32w4 | .bt64w4 => 4 | .bt64w8 => 8


def configs : List Config := [.bt16w2,.bt32w4,.bt32w2,.bt64w8,.bt64w4]


end VeriTile.Bench.Repairs.ChunkCumsumKernel
