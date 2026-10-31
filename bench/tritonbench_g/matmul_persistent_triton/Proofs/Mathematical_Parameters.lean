import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
open VeriTile.Triton

/-- Logical dimensions, nonnegative source strides and a symbolic persistent
configuration. Native launch admission and fixed-width overflow are separate. -/
structure Parameters where
  A : RegionName
  B : RegionName
  C : RegionName
  M : Nat
  N : Nat
  K : Nat
  sam : Nat
  sak : Nat
  sbk : Nat
  sbn : Nat
  scm : Nat
  scn : Nat
  BM : Nat
  BN : Nat
  BK : Nat
  GM : Nat
  SMS : Nat

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
