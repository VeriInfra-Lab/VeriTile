import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.AutotunePlan

/-! Shared source configuration and geometry, without Real or FP execution. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostFP
open VeriTile.Triton

def chunks (n b : Nat) := (n+b-1)/b
def padded (n b : Nat) := chunks n b*b

structure Inner where
  dot : NativeSource.Storage
  gate : NativeSource.Storage
  history : NativeSource.Storage
  seed : FloatDType
  final : FloatDType
  k : RegionName
  v : RegionName
  g : RegionName
  h : RegionName
  h0 : RegionName
  ht : RegionName
  heads : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
  gateK : Bool
  hasSeed : Bool
  storeFinal : Bool

def Inner.NT (c : Inner) := chunks c.T c.BT
def Inner.PT (c : Inner) := padded c.T c.BT
def Inner.PK (c : Inner) := padded c.K c.BK
def Inner.PV (c : Inner) := padded c.V c.BV
def Inner.total (c : Inner) := c.heads*(c.PK/c.BK)*(c.PV/c.BV)
def Inner.empty (c : Inner) : Bool := c.heads*c.T*c.K*c.V==0
def Inner.plans (c : Inner) := if c.empty then [] else LaunchPlan.plans c.total

structure Cumulative where
  dtype : FloatDType
  input : RegionName
  output : RegionName
  heads : Nat
  T : Nat
  S : Nat
  BT : Nat
  BS : Nat

def Cumulative.PT (c : Cumulative) := padded c.T c.BT
def Cumulative.PS (c : Cumulative) := padded c.S 64
def Cumulative.total (c : Cumulative) := c.heads*(c.PT/c.BT)*(c.PS/c.BS)
def Cumulative.maximum (c : Cumulative) := c.heads*(c.PT/c.BT)*(c.PS/16)
def Cumulative.empty (c : Cumulative) : Bool := c.heads*c.T*c.S==0
def Cumulative.plans (c : Cumulative) :=
  if c.empty then [] else AutotunePlan.plans c.maximum c.total

end VeriTile.Bench.Repairs.ChunkGatedAttention.HostFP
