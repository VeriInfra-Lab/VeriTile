import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.LegacyKernels

/-! The selected Python wrapper admits FP32 input/output. Both complete bodies
below are pasted without changes; pointer types and constexpr values are call
metadata. The old generic-dtype model remains explicit for compatibility
outside that Python contract. Proofs.SourcePreservation proves complete program
identity, including the signed loop, for every previous model instantiation. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumKernel
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("s",.fp32),("o",.fp32)]
  scalars := []
  constexpr := ["T","BT"]
  typedFP32Storage := Bool.true
  readPorts := ["s"]
  writePorts := ["o"]

def body (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
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

def sumLeftBody (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
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

/-- Compatibility of the earlier generic model interface; the admitted Python
FP32 call selects the actual pasted function body above. -/
def repairedKernel (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  if dtype = .fp32 ∧ out_dtype = .fp32 then body s o T BT
  else LegacyKernels.repairedKernel dtype out_dtype s o T BT

def sumLeftKernel (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  if dtype = .fp32 ∧ out_dtype = .fp32 then sumLeftBody s o T BT
  else LegacyKernels.sumLeftKernel dtype out_dtype s o T BT

inductive Config where
  | bt16w2 | bt32w4 | bt32w2 | bt64w8 | bt64w4
  deriving DecidableEq

def Config.block : Config → Nat
  | .bt16w2 => 16 | .bt32w4 | .bt32w2 => 32 | .bt64w8 | .bt64w4 => 64

def Config.warps : Config → Nat
  | .bt16w2 | .bt32w2 => 2 | .bt32w4 | .bt64w4 => 4 | .bt64w8 => 8


def configs : List Config := [.bt16w2,.bt32w4,.bt32w2,.bt64w8,.bt64w4]


end VeriTile.Bench.Repairs.ChunkCumsumKernel
