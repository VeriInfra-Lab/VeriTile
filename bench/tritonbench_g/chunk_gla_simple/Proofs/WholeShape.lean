import bench.tritonbench_g.chunk_gla_simple.Kernels
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple
open VeriTile.Triton
namespace Host
/-- Dimensions read by `chunk_fwd_o_fn` from K and V, and its explicit BT argument.
The tensor contract below uses contiguous Q/K `[B,H,T,K]`, V/O `[B,H,T,V]`,
G `[B,H,T]`, and packed history `[B,H,NT*K,V]`. H and G are caller inputs. -/
structure Shape where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  deriving DecidableEq, Repr

def nextPowerOfTwo (n : Nat) : Nat := 2 ^ Nat.clog 2 n
def Shape.BK (c : Shape) := min (nextPowerOfTwo c.K) 64
def Shape.BV (c : Shape) := min (nextPowerOfTwo c.V) 64
def Shape.NT (c : Shape) := (c.T + c.BT - 1) / c.BT
def Shape.NV (c : Shape) := (c.V + c.BV - 1) / c.BV
def Shape.heads (c : Shape) := c.B * c.H
def Shape.grid (c : Shape) : List Nat := [c.NV, c.NT, c.heads]
def Shape.historyShape (c : Shape) : List Nat := [c.B, c.H, c.NT*c.K, c.V]
def Shape.returnShape (c : Shape) : List Nat := [c.B, c.H, c.T, c.V]

/-- Default source precision is FP32. `empty_like(v)` returns the FP32 output
region with V's contiguous strides. The six stride arguments are exactly
`q.stride(1/2), v.stride(1/2), h.stride(1/2)`. -/
def kernel (c : Shape) (scale : ℝ) (swapped : Bool) : ComputeKernel :=
  source .fp32 .fp32 .fp32 swapped .fp32 "q" "k" "v" "h" "g" "o"
    (c.T*c.K) c.K (c.T*c.V) c.V (c.NT*c.K*c.V) c.V
    scale c.T c.K c.V c.BT c.BK c.BV
end Host
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple
