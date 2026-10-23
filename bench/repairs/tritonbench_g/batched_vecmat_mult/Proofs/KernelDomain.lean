import bench.repairs.tritonbench_g.batched_vecmat_mult.Contract
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.AdmissionCore

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelDomain
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult ValueProofs
abbrev Arguments := Host.Arguments

def parameters (a : Arguments) : Parameters :=
  ⟨"a","b","out",a.M,a.N,a.K,a.BM,a.BN,a.BK⟩

/-- Exactly the successful Python remainder/assert stage. A zero denominator
raises before launch; no dtype, count or payload restriction is added. -/
structure KernelContract (a : Arguments) : Prop where
  BM : 0<a.BM
  BN : 0<a.BN
  BK : 0<a.BK
  M : a.BM∣a.M
  N : a.BN∣a.N
  K : a.BK∣a.K

def checkKernel (a : Arguments) : Bool :=
  decide (a.BK≠0) && decide (a.K%a.BK=0) &&
    decide (a.BM≠0) && decide (a.M%a.BM=0) &&
    decide (a.BN≠0) && decide (a.N%a.BN=0)

theorem checkKernel_iff (a : Arguments) : checkKernel a=Bool.true ↔ KernelContract a := by
  simp only [checkKernel,Bool.and_eq_true,decide_eq_true_eq]
  constructor
  · rintro ⟨⟨⟨⟨⟨bk,k⟩,bm⟩,m⟩,bn⟩,n⟩
    exact ⟨Nat.pos_of_ne_zero bm,Nat.pos_of_ne_zero bn,Nat.pos_of_ne_zero bk,
      Nat.dvd_of_mod_eq_zero m,Nat.dvd_of_mod_eq_zero n,Nat.dvd_of_mod_eq_zero k⟩
  · intro h
    exact ⟨⟨⟨⟨⟨Nat.ne_of_gt h.BK,Nat.mod_eq_zero_of_dvd h.K⟩,Nat.ne_of_gt h.BM⟩,
      Nat.mod_eq_zero_of_dvd h.M⟩,Nat.ne_of_gt h.BN⟩,Nat.mod_eq_zero_of_dvd h.N⟩

/-- Necessary ordinary launch ABI facts already used by the accepted repair.
No grid-product/segmented-launch compatibility condition is added. -/
structure LaunchABI (a : Arguments) : Prop where
  blockM : a.BM≤2^31
  blockN : a.BN≤2^31
  blockK : a.BK≤2^31
  activeGrid : 0<gridCount (parameters a) → gridM (parameters a)≤2^31 ∧ gridN (parameters a)≤65535
  warps : ∃ e : Nat,a.numWarps=2^e

def launch (a : Arguments) (memory : Memory) : Host.Launch :=
  ⟨parameters a,memory,a.numWarps,a.numStages⟩

theorem launch_abi (a : Arguments) (h : LaunchABI a) (memory : Memory) :
    Whole.LaunchABI (launch a memory) :=
  ⟨h.blockM,h.blockN,h.blockK,h.activeGrid,h.warps⟩

#axiomsClean checkKernel_iff
#axiomsClean launch_abi
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelDomain
