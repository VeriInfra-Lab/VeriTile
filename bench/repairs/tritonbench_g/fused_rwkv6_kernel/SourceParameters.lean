namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel

/-- Exact source launch parameters, shared with the independent mathematical model. -/
structure Shape where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BK : Nat
  BV : Nat
  keyPitch : Nat
  valuePitch : Nat
  initial : Bool
  final : Bool
  reverse : Bool
  deriving DecidableEq

end VeriTile.Bench.Repairs.FusedRwkv6Kernel
