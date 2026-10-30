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

def source (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool) : ComputeKernel :=
  NativeSource.hOriginal .fp32 .fp32 .fp32 .fp32 .fp32 "k" "v" "g" "h" "h0" "ht"
    (c.T*c.K) c.K 1 (c.T*c.V) c.V 1 (c.NT*c.K*c.V) c.V 1
    c.T c.K c.V c.BT c.BK c.BV c.NT base gateK hasSeed storeFinal

def parts (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (source c base gateK hasSeed storeFinal).surfaceBody "i_t").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

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

/-- FP32 pointer loads and all explicit source casts remain observable. -/
def load {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (r : RegionName) (off : Nat) : α :=
  M.fp32Load ((s.mem r off).read .real)
def zero {α : Type} (M : Algebra α) : α := M.literal (some .fp32) .real 0

def initial {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (base : Nat)
    (s : State α) (hasSeed : Bool) : HIndex c → α := fun i =>
  if hasSeed then M.binary (some .fp32) .real .add (zero M)
    (M.cast (some .fp32) .fp32 .real ((s.mem "h0" (stateOffset c base s i)).read .fp32)) else zero M

def keyValue {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (base : Nat)
    (s : State α) (gateK : Bool) (chunk : Nat) : KIndex c → α := fun i =>
  let value:=load M s "k" (keyOffset c base s chunk i)
  if gateK then M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul value
    (M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub
      (load M s "g" (keyEndpoint c base s chunk i.1)) (load M s "g" (keyOffset c base s chunk i)))))
  else value

def valueValue {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (base : Nat)
    (s : State α) (gateK : Bool) (chunk : Nat) : VIndex c → α := fun i =>
  let value:=load M s "v" (valueOffset c base s chunk i)
  if gateK then value
  else M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul value
    (M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub
      (load M s "g" (valueEndpoint c base s chunk i.2.1)) (load M s "g" (valueOffset c base s chunk i)))))

def gated {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (base : Nat)
    (s : State α) (gateK : Bool) (chunk : Nat) (carry : HIndex c → α) : HIndex c → α := fun i =>
  M.binary (some .fp32) .real .mul (carry i) (M.unary (some .fp32) .exp (load M s "g"
    (if gateK then keyEndpoint c base s chunk i.1 else valueEndpoint c base s chunk i.2.1)))

/-- Staging consumes the actual right-hand pending dot with the gated carry.
There is no assumed split-rounding/fusion equation. -/
def update (c : Shape) (base : Nat) (s : State NativeGraph.Expr) (gateK : Bool)
    (chunk : Nat) (carry : HIndex c → NativeGraph.Expr) : HIndex c → NativeGraph.Expr := fun i =>
  .dotAcc (some .fp32) NativeGraph.options [] c.BK c.BT c.BV
    (keyValue algebra c base s gateK chunk)
    (valueValue algebra c base s gateK chunk)
    (gated algebra c base s gateK chunk carry i) i

def accumulated (c : Shape) (base : Nat) (s : State NativeGraph.Expr) (gateK hasSeed : Bool)
    (chunks : Nat) : HIndex c → NativeGraph.Expr :=
  foldRange (update c base s gateK) 0 chunks 1 (initial algebra c base s hasSeed)

def stored (c : Shape) (base : Nat) (s : State NativeGraph.Expr) (gateK hasSeed : Bool)
    (chunks : Nat) (i : HIndex c) : NativeGraph.Expr :=
  .cast (some .fp32) .real .real (accumulated c base s gateK hasSeed chunks i)

#specNonCircular accumulated avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
