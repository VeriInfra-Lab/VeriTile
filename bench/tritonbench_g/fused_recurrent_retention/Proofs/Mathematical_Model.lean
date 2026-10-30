import VeriTile.Triton

/-! Parameters of the actual per-program retention kernels. The four unused
time/channel stride arguments remain in the source signature; addresses below
use the source's literal DK/DV increments. No host-layout claim is implicit. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full
open VeriTile.Triton

structure Layout where
  qHead : Nat
  qTime : Nat
  qChannel : Nat
  vHead : Nat
  vTime : Nat
  vChannel : Nat
  B : Nat
  H : Nat
  T : Nat
  BK : Nat
  BV : Nat
  DK : Nat
  DV : Nat
  scale : ℝ

structure Launch where
  value : Nat
  key : Nat
  head : Nat

def launch (s : BlockState) : Launch := ⟨s.pids 0, s.pids 1, s.pids 2⟩

structure ForwardRegions where
  q : RegionName
  k : RegionName
  v : RegionName
  o : RegionName
  initial : RegionName
  final : RegionName

structure BackwardRegions where
  q : RegionName
  k : RegionName
  v : RegionName
  do_ : RegionName
  dq : RegionName
  dk : RegionName
  dv : RegionName
  initial : RegionName

def activeKey (p : Layout) (l : Launch) (j : Nat) : Prop := l.key * p.BK + j < p.DK
def activeValue (p : Layout) (l : Launch) (j : Nat) : Prop := l.value * p.BV + j < p.DV
instance (p : Layout) (l : Launch) (j : Nat) : Decidable (activeKey p l j) :=
  inferInstanceAs (Decidable (l.key * p.BK + j < p.DK))
instance (p : Layout) (l : Launch) (j : Nat) : Decidable (activeValue p l j) :=
  inferInstanceAs (Decidable (l.value * p.BV + j < p.DV))

def keyAddress (p : Layout) (l : Launch) (t j : Nat) : Nat :=
  l.head * p.qHead + l.key * p.BK + j + t * p.DK
def valueAddress (p : Layout) (l : Launch) (t j : Nat) : Nat :=
  l.head * p.vHead + l.value * p.BV + j + t * p.DV
def stateAddress (p : Layout) (l : Launch) (jk jv : Nat) : Nat :=
  l.head * p.DK * p.DV + (l.key * p.BK + jk) * p.DV + (l.value * p.BV + jv)
def outputAddress (p : Layout) (l : Launch) (t j : Nat) : Nat :=
  (l.head + l.key * p.B * p.H) * p.vHead + l.value * p.BV + j + t * p.DV
def keyGradientAddress (p : Layout) (l : Launch) (t j : Nat) : Nat :=
  (l.head + l.value * p.B * p.H) * p.qHead + l.key * p.BK + j + t * p.DK

end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full
