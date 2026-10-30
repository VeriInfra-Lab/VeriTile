import bench.tritonbench_g.matmul_persistent_triton.Sources.Mathematical_Source

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

def Parameters.source (p : Parameters) (storage : TypedSource.Storage) (candidate : Bool) :=
  TypedSource.kernel storage candidate p.A p.B p.C p.M p.N p.K
    p.sam p.sak p.sbk p.sbn p.scm p.scn p.BM p.BN p.BK p.GM p.SMS

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
