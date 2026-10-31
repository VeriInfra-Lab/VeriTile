import bench.tritonbench_g.chunk_gated_attention.Sources.Kernels
import bench.tritonbench_g.chunk_gla_simple.Sources.ScalarGraph
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000

/-- Every admitted operand is FP32; source dots use the original fp32
profile and retain pending results until consumed by the live accumulator. -/
def algebra : Algebra NativeGraph.Expr := ScalarGraph.algebra

/-- Padded physical dimensions used by one actual source specialization. -/
structure Shape where
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
  NT : Nat

abbrev HIndex (c : Shape) := TileIndex [c.BK,c.BV]
abbrev KIndex (c : Shape) := TileIndex [c.BK,c.BT]
abbrev VIndex (c : Shape) := TileIndex [c.BT,c.BV]

def linear {α : Type} (base : Nat) (s : State α) := s.pids 0+base
def vblock {α : Type} (c : Shape) (base : Nat) (s : State α) := linear base s%(c.V/c.BV)
def kblock {α : Type} (c : Shape) (base : Nat) (s : State α) := linear base s/(c.V/c.BV)%(c.K/c.BK)
def head {α : Type} (c : Shape) (base : Nat) (s : State α) := linear base s/((c.V/c.BV)*(c.K/c.BK))

def stateOffset {α : Type} (c : Shape) (base : Nat) (s : State α) (i : HIndex c) : Nat :=
  head c base s*c.K*c.V+kblock c base s*c.BK*c.V+vblock c base s*c.BV+i.1.val*c.V+i.2.1.val

def historyOffset {α : Type} (c : Shape) (base : Nat) (s : State α) (chunk : Nat) (i : HIndex c) : Nat :=
  head c base s*(c.NT*c.K*c.V)+chunk*c.K*c.V+kblock c base s*c.BK*c.V+vblock c base s*c.BV+i.1.val*c.V+i.2.1.val

def keyOffset {α : Type} (c : Shape) (base : Nat) (s : State α) (chunk : Nat) (i : KIndex c) : Nat :=
  head c base s*(c.T*c.K)+kblock c base s*c.BK+chunk*c.BT*c.K+i.1.val+i.2.1.val*c.K

def valueOffset {α : Type} (c : Shape) (base : Nat) (s : State α) (chunk : Nat) (i : VIndex c) : Nat :=
  head c base s*(c.T*c.V)+chunk*c.BT*c.V+vblock c base s*c.BV+i.1.val*c.V+i.2.1.val

def keyEndpoint {α : Type} (c : Shape) (base : Nat) (s : State α) (chunk : Nat) (i : Fin c.BK) : Nat :=
  head c base s*(c.T*c.K)+(chunk*c.BT+c.BT-1)*c.K+kblock c base s*c.BK+i.val

def valueEndpoint {α : Type} (c : Shape) (base : Nat) (s : State α) (chunk : Nat) (i : Fin c.BV) : Nat :=
  head c base s*(c.T*c.V)+(chunk*c.BT+c.BT-1)*c.V+vblock c base s*c.BV+i.val

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
