import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural

structure Layout where
  X : RegionName
  W : RegionName
  Y : RegionName
  M : Nat
  N : Nat
  B : Nat
  complete : N ≤ B

def rowState (pid : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then pid else 0 }


end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
