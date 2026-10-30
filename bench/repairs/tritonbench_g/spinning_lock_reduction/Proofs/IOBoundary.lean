import bench.repairs.tritonbench_g.spinning_lock_reduction.IO
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.FiniteBoundary
import VeriTile.Triton.Concurrent.EventLocality
import VeriTile.Triton.Concurrent.EventFrame

namespace VeriTile.Bench.Repairs.SpinningLock.IOProof
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock

theorem input_partials (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (input : (IO.io V h count stage).Inputs) (s : Concurrent.State V)
    (binding : (IO.io V h count stage).InputBinding input s) (i : Fin (Formula.partialCount h)) :
    s.mem 0 i.val = (IO.inputs h count stage input).partials i := by
  simpa [IO.io,IO.inputs,IO.partials,Concurrent.Window.address] using
    binding.1 ⟨0,by simp [IO.io]⟩ i rfl

theorem input_output (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (input : (IO.io V h count stage).Inputs) (s : Concurrent.State V)
    (binding : (IO.io V h count stage).InputBinding input s) (i : Fin count) :
    s.mem 1 i.val = (IO.inputs h count stage input).output i := by
  simpa [IO.io,IO.inputs,IO.output,Concurrent.Window.address] using
    binding.1 ⟨1,by simp [IO.io]⟩ i rfl

theorem input_locks (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (input : (IO.io V h count stage).Inputs) (s : Concurrent.State V)
    (binding : (IO.io V h count stage).InputBinding input s) (i : Fin h.numSms) :
    s.words 0 i.val = (IO.inputs h count stage input).locks i := by
  simpa [IO.io,IO.inputs,IO.locks,Concurrent.Window.address] using
    binding.2 ⟨0,by simp [IO.io]⟩ i rfl

/-- The declared IO has a concrete capacity witness, including empty outputs;
its universally quantified safety obligation is therefore nonvacuous. -/
theorem capacity (h : HostArgs) (count : Nat) (stage : Phased.Stage) :
    (IO.io V h count stage).InBounds (IO.bounds h count) := by
  simp [Concurrent.EventProgramIO.InBounds,IO.io,IO.partials,IO.output,IO.locks,
    Concurrent.Window.InBounds,Concurrent.Window.address,IO.bounds,Concurrent.InWindow]

private theorem prefix_size (name : String) (region count : Nat) (sizes : Nat → Nat)
    (bound : (show Concurrent.Window from {name:=name,region:=region,count:=count}).InBounds sizes) :
    count ≤ sizes region := by
  by_cases empty : count = 0
  · omega
  · have hk : count-1 < count := by omega
    have last := bound ⟨count-1,hk⟩ rfl
    simp only [Concurrent.Window.address,Int.mul_one,Int.zero_add,Concurrent.InWindow] at last
    omega

theorem bound_sizes (h : HostArgs) (count : Nat) (stage : Phased.Stage) (bounds : Concurrent.Bounds)
    (bound : (IO.io V h count stage).InBounds bounds) :
    Formula.partialCount h ≤ bounds.mem 0 ∧ count ≤ bounds.mem 1 ∧ h.numSms ≤ bounds.words 0 := by
  exact ⟨prefix_size "partials" 0 _ _ (bound.2.2.1 (IO.partials h) (by simp [IO.io])),
    prefix_size "output" 1 _ _ (bound.2.2.1 (IO.output count) (by simp [IO.io])),
    prefix_size "locks" 0 _ _ (bound.2.2.2.1 (IO.locks h) (by simp [IO.io]))⟩

def canonical (b : Backend V) (h : HostArgs) (input : Formula.Values V h count)
    (before : Concurrent.State V) : Concurrent.State V :=
  { before with
    mem := fun region => if region = 0 then FiniteBoundary.extend b.zero input.partials
      else if region = 1 then FiniteBoundary.extend b.zero input.output else before.mem region
    words := fun region => if region = 0 then FiniteBoundary.extend 0 input.locks else before.words region }

theorem canonical_memory (b : Backend V) (h : HostArgs) (input : Formula.Values V h count)
    (before : Concurrent.State V) :
    (ConcurrentSource.values (canonical b h input before)).mem = FiniteBoundary.memory b h input := rfl

theorem canonical_agree (b : Backend V) (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (input : (IO.io V h count stage).Inputs) (before : Concurrent.State V)
    (binding : (IO.io V h count stage).InputBinding input before) :
    Concurrent.Locality.Agree (IO.bounds h count)
      (canonical b h (IO.inputs h count stage input) before) before := by
  refine ⟨?_,?_,?_,rfl,rfl⟩
  · intro region address active
    exact False.elim (by simp [IO.bounds,Concurrent.InWindow] at active; omega)
  · intro region address active
    by_cases hp : region = 0
    · subst region
      have ha : 0 ≤ address ∧ address < Formula.partialCount h := by simpa [IO.bounds,Concurrent.InWindow] using active
      have hi : address.toNat < Formula.partialCount h := by omega
      have he := input_partials h count stage input before binding ⟨address.toNat,hi⟩
      simpa [canonical,FiniteBoundary.extend,ha,Int.toNat_of_nonneg ha.1] using he.symm
    · by_cases hc : region = 1
      · subst region
        have ha : 0 ≤ address ∧ address < count := by simpa [IO.bounds,Concurrent.InWindow] using active
        have hi : address.toNat < count := by omega
        have he := input_output h count stage input before binding ⟨address.toNat,hi⟩
        simpa [canonical,FiniteBoundary.extend,ha,Int.toNat_of_nonneg ha.1] using he.symm
      · exact False.elim (by simp [IO.bounds,Concurrent.InWindow,hp,hc] at active; omega)
  · intro region address active
    by_cases hp : region = 0
    · subst region
      have ha : 0 ≤ address ∧ address < h.numSms := by simpa [IO.bounds,Concurrent.InWindow] using active
      have hi : address.toNat < h.numSms := by omega
      have he := input_locks h count stage input before binding ⟨address.toNat,hi⟩
      simpa [canonical,FiniteBoundary.extend,ha,Int.toNat_of_nonneg ha.1] using he.symm
    · exact False.elim (by simp [IO.bounds,Concurrent.InWindow,hp] at active; omega)

end VeriTile.Bench.Repairs.SpinningLock.IOProof
