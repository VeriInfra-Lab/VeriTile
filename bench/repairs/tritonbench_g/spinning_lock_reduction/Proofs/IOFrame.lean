import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.IOBoundary

namespace VeriTile.Bench.Repairs.SpinningLock.IOProof
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock

theorem scalar_written (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (region : Nat) (address : Int) (active : Concurrent.InWindow ((IO.bounds h count).mem region) address) :
    (IO.io V h count stage).ScalarWritten region address := by
  by_cases hp : region = 0
  · subst region
    have ha : 0 ≤ address ∧ address < Formula.partialCount h := by simpa [Concurrent.InWindow,IO.bounds] using active
    refine ⟨IO.partials h,by simp [IO.io],⟨address.toNat,by change address.toNat < Formula.partialCount h; omega⟩,rfl,rfl,?_⟩
    simp [IO.partials,Concurrent.Window.address,Int.toNat_of_nonneg ha.1]
  · by_cases hc : region = 1
    · subst region
      have ha : 0 ≤ address ∧ address < count := by simpa [Concurrent.InWindow,IO.bounds] using active
      refine ⟨IO.output count,by simp [IO.io],⟨address.toNat,by change address.toNat < count; omega⟩,rfl,rfl,?_⟩
      simp [IO.output,Concurrent.Window.address,Int.toNat_of_nonneg ha.1]
    · exact False.elim (by simp [Concurrent.InWindow,IO.bounds,hp,hc] at active; omega)

theorem word_written (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (region : Nat) (address : Int) (active : Concurrent.InWindow ((IO.bounds h count).words region) address) :
    (IO.io V h count stage).WordWritten region address := by
  by_cases hp : region = 0
  · subst region
    have ha : 0 ≤ address ∧ address < h.numSms := by simpa [Concurrent.InWindow,IO.bounds] using active
    refine ⟨IO.locks h,by simp [IO.io],⟨address.toNat,by change address.toNat < h.numSms; omega⟩,rfl,rfl,?_⟩
    simp [IO.locks,Concurrent.Window.address,Int.toNat_of_nonneg ha.1]
  · exact False.elim (by simp [Concurrent.InWindow,IO.bounds,hp] at active; omega)

theorem frame (b : Concurrent.Backend V) (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (program : Concurrent.Program) (initial final : Concurrent.State V) (events : List Concurrent.Event)
    (ran : program.Runs b initial events final) (safe : Concurrent.TapeSafe (IO.bounds h count) events) :
    (IO.io V h count stage).Frame initial final := by
  apply Concurrent.EventProgramIO.frame_of_writes _ b program initial final events ran
  · intro e member region address write
    have bounds := safe e member
    cases e <;> simp only [Concurrent.Event.WritesMem] at write
    all_goals try contradiction
    all_goals rcases write with ⟨rfl,rfl⟩
    all_goals exact scalar_written h count stage _ _ bounds.1
  · intro e member region address write
    have bounds := safe e member
    cases e <;> simp only [Concurrent.Event.WritesWord] at write
    all_goals try contradiction
    all_goals rcases write with ⟨rfl,rfl⟩
    all_goals exact word_written h count stage _ _ bounds

theorem output_binding (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (expected : Formula.Values V h count) (final : Concurrent.State V)
    (partials : ∀ i : Fin (Formula.partialCount h),final.mem 0 i.val = expected.partials i)
    (output : ∀ i : Fin count,final.mem 1 i.val = expected.output i)
    (locks : ∀ i : Fin h.numSms,final.words 0 i.val = expected.locks i) :
    (IO.io V h count stage).OutputBinding (IO.outputs h count stage expected) final := by
  constructor
  · rintro ⟨port,hp⟩ lane active
    have bound : port < 2 := hp
    interval_cases port
    · simpa [IO.io,IO.outputs,IO.partials,Concurrent.Window.address] using partials lane
    · simpa [IO.io,IO.outputs,IO.output,Concurrent.Window.address] using output lane
  · rintro ⟨port,hp⟩ lane active
    have he : port = 0 := by change port < 1 at hp; omega
    subst port
    simpa [IO.io,IO.outputs,IO.locks,Concurrent.Window.address] using locks lane

end VeriTile.Bench.Repairs.SpinningLock.IOProof
