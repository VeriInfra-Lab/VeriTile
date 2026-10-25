import VeriTile.Triton.DSL
import VeriTile.Triton.Source

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.LegacyKernels
open VeriTile.Triton

/-- Actual minimal index repair. The loop is explicitly signed64 in Python;
the DSL retains its signed loop and casts. Explicit masked pointers avoid the
native block-pointer offset's int32 restriction. Unspecified padding stays
unspecified, with the same logical inactive address and original FP operations. -/
def repairedKernel (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[s], outputs=[o]
    tl.region s = DTYPE, o = OUT_DTYPE
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range($((0 : Int)), $((Int.ofNat ((T+BT-1)/BT) : Int)), $(1)) {
      times = i_t * $(BT) + tl.arange(0, BT).to(tl.int64)
      p_s = s+i_bh.to(tl.int64)*$(T)+times
      p_o = o+i_bh.to(tl.int64)*$(T)+times
      b_s = tl.load(p_s, mask=times<$(T)).to(tl.float32)
      b_o = tl.cumsum(b_s, axis=0) + b_z[None]
      b_zz = tl.sum(b_s, axis=0)
      b_z += b_zz
      tl.store(p_o, b_o.to(OUT_DTYPE), mask=times<$(T))
    }
  }

/-- The intended candidate changes only the FP32 carry-add operand order. -/
def sumLeftKernel (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[s], outputs=[o]
    tl.region s = DTYPE, o = OUT_DTYPE
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range($((0 : Int)), $((Int.ofNat ((T+BT-1)/BT) : Int)), $(1)) {
      times = i_t * $(BT) + tl.arange(0, BT).to(tl.int64)
      p_s = s+i_bh.to(tl.int64)*$(T)+times
      p_o = o+i_bh.to(tl.int64)*$(T)+times
      b_s = tl.load(p_s, mask=times<$(T)).to(tl.float32)
      b_o = tl.cumsum(b_s, axis=0) + b_z[None]
      b_zz = tl.sum(b_s, axis=0)
      b_z = b_zz + b_z
      tl.store(p_o, b_o.to(OUT_DTYPE), mask=times<$(T))
    }
  }

end VeriTile.Bench.Repairs.ChunkCumsumKernel.LegacyKernels
