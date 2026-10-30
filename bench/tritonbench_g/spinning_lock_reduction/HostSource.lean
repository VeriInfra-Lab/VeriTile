import bench.tritonbench_g.spinning_lock_reduction.Source

namespace VeriTile.Bench.TritonBenchG.SpinningLock
/-- Actual wrapper parameters. P is a flat buffer of tiles; C uses the two
supplied element strides. The wrapper allocates nothing and returns None. -/
structure HostArgs where
  numSms : Nat
  groupK : Nat
  m : Nat
  n : Nat
  strideM : Nat
  strideN : Nat
  bm : Nat
  bn : Nat
  deriving DecidableEq, Repr

def HostArgs.config (h : HostArgs) : Config :=
  ⟨h.numSms, h.bm, h.bn, BitVec.ofNat 32 h.numSms, BitVec.ofNat 32 h.groupK,
    BitVec.ofNat 32 h.m, BitVec.ofNat 32 h.n,
    BitVec.ofNat 32 h.strideM, BitVec.ofNat 32 h.strideN⟩

/-- Source wrapper return value, separate from its in-place C observation. -/
def wrapperReturn : Option Unit := none

end VeriTile.Bench.TritonBenchG.SpinningLock
