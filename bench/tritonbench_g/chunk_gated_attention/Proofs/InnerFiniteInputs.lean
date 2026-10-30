import bench.tritonbench_g.chunk_gated_attention.Proofs.RealStages

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath
noncomputable section
set_option maxHeartbeats 3000000

theorem key_address_bound (p : Layout) (heads : Nat) (l : Launch)
    (hp : StagedLayout.Canonical p) (hh : l.head<heads) (t k : Nat)
    (ht : t<p.T) (hk : k<p.K) : keyAddress p l k t < heads*p.T*p.K := by
  convert Address.flatten3_bounded hh ht hk using 1 <;>
    simp only [keyAddress,hp.keyHead,hp.keyTime,hp.keyFeature,Nat.mul_one] <;> ring

theorem value_address_bound (p : Layout) (heads : Nat) (l : Launch)
    (hp : StagedLayout.Canonical p) (hh : l.head<heads) (t v : Nat)
    (ht : t<p.T) (hv : v<p.V) : valueAddress p l t v < heads*p.T*p.V := by
  simpa only [valueAddress,hp.valueHead,hp.valueTime,hp.valueFeature,Nat.mul_one,Nat.mul_assoc,Nat.add_mul]
    using Address.flatten3_bounded hh ht hv

theorem state_address_bound (p : Layout) (heads : Nat) (l : Launch)
    (hh : l.head<heads) (k v : Nat) (hk : k<p.K) (hv : v<p.V) :
    stateAddress p l k v < heads*p.K*p.V := by
  simpa only [stateAddress,Nat.add_mul,Nat.mul_assoc] using Address.flatten3_bounded hh hk hv

/-- Finite role values agree on precisely the allocated staged input cells.
The seed condition is conditional on the source actually reading a seed. -/
structure FiniteInputs (p : Layout) (heads T K V : Nat) (gateK hasSeed : Bool)
    (x : Nat → Host.InnerInputs) (a : Inputs) : Prop where
  key : ∀ off, off<heads*p.T*p.K → a.key off=(Materialize.inputs p T K V gateK x).key off
  value : ∀ off, off<heads*p.T*p.V → a.value off=(Materialize.inputs p T K V gateK x).value off
  gate : ∀ off, off<heads*p.T*(if gateK then p.K else p.V) →
    a.gate off=(Materialize.inputs p T K V gateK x).gate off
  seed : hasSeed=Bool.true → ∀ off, off<heads*p.K*p.V →
    a.seed off=(Materialize.inputs p T K V gateK x).seed off

def effectiveInputs (a target : Inputs) (hasSeed : Bool) : Inputs :=
  {a with seed:=if hasSeed then a.seed else target.seed}

theorem effective_closed (p : Layout) (l : Launch) (a target : Inputs)
    (gateK hasSeed : Bool) (c k v : Nat) :
    closed p l a gateK hasSeed c k v =
      closed p l (effectiveInputs a target hasSeed) gateK hasSeed c k v := by
  cases hasSeed <;>
    simp [closed,seed,effectiveInputs,gateFactor,contribution,endpoint,gateValue,keyValue,valueValue]

theorem finite_staged (p : Layout) (heads T K V : Nat) (l : Launch)
    (gateK hasSeed : Bool) (x : Nat → Host.InnerInputs) (a : Inputs)
    (hp : StagedLayout.Canonical p) (hh : l.head<heads) (hk : K≤p.K) (hv : V≤p.V)
    (finite : FiniteInputs p heads T K V gateK hasSeed x a) :
    HostStaging.Staged p l (effectiveInputs a (Materialize.inputs p T K V gateK x) hasSeed)
      (x l.head) T K V gateK := by
  have old := Materialize.staged p l T K V gateK x hp hk hv
  refine ⟨old.key_le,old.value_le,old.time_extent,old.positive_time,old.key_stride,old.value_stride,?_,?_,?_,?_,?_⟩
  · intro t k ht hkl
    exact (finite.key _ (key_address_bound p heads l hp hh t k ht (lt_of_lt_of_le hkl hk))).trans (old.key t k ht hkl)
  · intro t v ht hvl
    exact (finite.value _ (value_address_bound p heads l hp hh t v ht (lt_of_lt_of_le hvl hv))).trans (old.value t v ht hvl)
  · intro h t k ht hkl
    subst gateK
    exact (finite.gate _ (key_address_bound p heads l hp hh t k ht (lt_of_lt_of_le hkl hk))).trans (old.gate_key rfl t k ht hkl)
  · intro h t v ht hvl
    subst gateK
    exact (finite.gate _ (value_address_bound p heads l hp hh t v ht (lt_of_lt_of_le hvl hv))).trans (old.gate_value rfl t v ht hvl)
  · intro k v hkl hvl
    cases hasSeed
    · exact old.seed k v hkl hvl
    · exact (finite.seed rfl _ (state_address_bound p heads l hh k v
        (lt_of_lt_of_le hkl hk) (lt_of_lt_of_le hvl hv))).trans (old.seed k v hkl hvl)

theorem finite_closed (p : Layout) (heads T K V : Nat) (l : Launch)
    (gateK hasSeed : Bool) (x : Nat → Host.InnerInputs) (a : Inputs)
    (hp : StagedLayout.Canonical p) (hh : l.head<heads) (hk : K≤p.K) (hv : V≤p.V)
    (finite : FiniteInputs p heads T K V gateK hasSeed x a)
    (c k v : Nat) (hc : c≤p.NT) (hkl : k<K) (hvl : v<V) :
    closed p l a gateK hasSeed c k v = Host.innerState (x l.head) T p.BT gateK hasSeed c k v :=
  (effective_closed p l a (Materialize.inputs p T K V gateK x) gateK hasSeed c k v).trans
    (HostStaging.closed_eq p l _ (x l.head) T K V gateK hasSeed
      (finite_staged p heads T K V l gateK hasSeed x a hp hh hk hv finite) c k v hc hkl hvl)

theorem inner_binding_finite (p : Layout) (heads T K V : Nat) (r : RealHProgram.Regions)
    (gateK hasSeed storeFinal : Bool) (x : Nat → Host.InnerInputs)
    (values : (innerIO p heads K V r gateK hasSeed storeFinal).Ctx)
    (domain : innerDomain p heads T K V gateK hasSeed x values) (before : BlockState)
    (binding : (innerIO p heads K V r gateK hasSeed storeFinal).InputBinding before values) :
    FiniteInputs p heads T K V gateK hasSeed x
      ⟨before.readMem r.key,before.readMem r.value,before.readMem r.gate,before.readMem r.seed⟩ := by
  constructor
  · intro off bound
    exact (binding (0 : Fin 4) ⟨off,bound⟩ (Or.inl (by decide))).trans
      (domain 0 ⟨off,bound⟩ (Or.inl (by decide)))
  · intro off bound
    exact (binding (1 : Fin 4) ⟨off,bound⟩ (Or.inl (by decide))).trans
      (domain 1 ⟨off,bound⟩ (Or.inl (by decide)))
  · intro off bound
    exact (binding (2 : Fin 4) ⟨off,bound⟩ (Or.inl (by decide))).trans
      (domain 2 ⟨off,bound⟩ (Or.inl (by decide)))
  · intro enabled off bound
    exact (binding (3 : Fin 4) ⟨off,bound⟩ (Or.inr enabled)).trans
      (domain 3 ⟨off,bound⟩ (Or.inr enabled))

#axiomsClean finite_closed
#axiomsClean inner_binding_finite
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface
