import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawHostFacade
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeNativePlans

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBoundary
open VeriTile.Triton FP.Structural Math Program RawObserved Backend RawBackend
noncomputable section
set_option maxHeartbeats 2000000

/-- Physical allocation bases are byte addresses. Extents are FP32 words;
views into one allocation may alias, distinct allocations do not overlap. -/
structure AddressSpace where
  base : Nat → Nat
  extent : Nat → Nat
  aligned : ∀ allocation, base allocation % 4 = 0
  bounded : ∀ allocation, base allocation+4*extent allocation ≤ 2^64
  disjoint : ∀ a b, a ≠ b →
    base a+4*extent a ≤ base b ∨ base b+4*extent b ≤ base a

def Views.port (v : Views) (rg : RegionName) : View :=
  if rg = ports.dt then v.dt else if rg = ports.rate then v.rate
  else if rg = ports.bias then v.bias else if rg = ports.out then v.out else v.sum

def physicalAddress (A : AddressSpace) (v : View) (off : Nat) :=
  A.base v.allocation+4*(v.base+off)

def AccessBounds (A : AddressSpace) (v : Views) (c : Whole.Host) (hasBias : Bool) : Prop :=
  ∀ l, c.valid l → ∀ rg off, Whole.Access c ports hasBias l rg off →
    (v.port rg).base+off < A.extent (v.port rg).allocation

theorem full_physical_access (A : AddressSpace) (v : View) (off : Nat)
    (bound : v.base+off < A.extent v.allocation) :
    physicalAddress A v off % 4 = 0 ∧
    physicalAddress A v off+4 ≤ A.base v.allocation+4*A.extent v.allocation ∧
    physicalAddress A v off+3 < 2^64 ∧
    (BitVec.ofNat 64 (physicalAddress A v off)).toNat = physicalAddress A v off ∧
    (BitVec.ofNat 64 (physicalAddress A v off+3)).toNat = physicalAddress A v off+3 := by
  have total := A.bounded v.allocation
  have completeBytes : physicalAddress A v off+4 ≤ A.base v.allocation+4*A.extent v.allocation := by
    unfold physicalAddress
    omega
  have last : physicalAddress A v off+3 < 2^64 := by omega
  refine ⟨?_,completeBytes,last,Nat.mod_eq_of_lt (by omega),Nat.mod_eq_of_lt last⟩
  simp [physicalAddress,Nat.add_mod,A.aligned]

structure LaunchSafety (a : Allocated α call c entry preexisting) (p : LaunchPlan α c) where
  indices : Whole.Signed32Domain (reblock c p.block)
  addresses : AddressSpace
  bounds : AccessBounds addresses a.views (reblock c p.block) call.hasBias

def HostSafety (rt : Runtime α call entry preexisting) : Prop :=
  ∀ g a p, rt.setup g a = .ok p → ∀ launch ∈ p.selected::p.trials, Nonempty (LaunchSafety a launch)

def SourceSafety (rt : Runtime α call entry preexisting) : Prop :=
  ∀ g a p, rt.setup g a = .ok p → ∀ launch ∈ p.selected::p.trials,
    (∀ l, (reblock g.host launch.block).valid l →
      ∀ (i : Fin (reblock g.host launch.block).layout.BH) (j : Fin (reblock g.host launch.block).layout.BC),
      ∀ x ∈ Whole.integerTerms (reblock g.host launch.block) l i.val j.val,
        (BitVec.ofNat 32 x).toNat = x ∧ (BitVec.ofNat 32 x).toInt = (x : Int)) ∧
    ∃ A : AddressSpace, ∀ l, (reblock g.host launch.block).valid l →
      ∀ rg off, Whole.Access (reblock g.host launch.block) ports call.hasBias l rg off →
        physicalAddress A (a.views.port rg) off % 4 = 0 ∧
        physicalAddress A (a.views.port rg) off+4 ≤ A.base (a.views.port rg).allocation+4*A.extent (a.views.port rg).allocation ∧
        physicalAddress A (a.views.port rg) off+3 < 2^64 ∧
        (BitVec.ofNat 64 (physicalAddress A (a.views.port rg) off)).toNat = physicalAddress A (a.views.port rg) off ∧
        (BitVec.ofNat 64 (physicalAddress A (a.views.port rg) off+3)).toNat = physicalAddress A (a.views.port rg) off+3

theorem host_safety (rt : Runtime α call entry preexisting) (safe : HostSafety rt) : SourceSafety rt := by
  intro g a p setup launch member
  obtain ⟨proof⟩ := safe g a p setup launch member
  refine ⟨Whole.source_indices_exact _ proof.indices,proof.addresses,?_⟩
  intro l valid rg off access
  exact full_physical_access proof.addresses (a.views.port rg) off (proof.bounds l valid rg off access)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBoundary
