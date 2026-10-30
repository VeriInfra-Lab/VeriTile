import bench.tritonbench_g.cache_transform_triton.Proofs.KernelSource

namespace VeriTile.Bench.Repairs.CacheTransform.KernelInterface
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 100000

/-- The unchanged START=0 source launch, derived from raw grid bounds. -/
theorem launch_success [Inhabited α] (M : Algebra α) (p : Parameters)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit)
    (s : State α) (n : Nat) (hn : n≤blocks p.count) :
    KernelSource.launch M p s n=some (runTiles M p s n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have bound : n<gridLimit := by omega
    have selection : chunkStart n=0 ∧ localPid n=n := by
      rw [chunkStart_eq,localPid_eq,Nat.div_eq_of_lt bound,Nat.mod_eq_of_lt bound]
      simp
    have parameters : tileParameters p n=p := by
      cases p <;> simp_all only [tileParameters]
    have position : ∀ s : State α,tileState s n=KernelSource.positioned s n := by
      intro s
      simp only [tileState,KernelSource.positioned,selection.2]
    rw [KernelSource.launch,ih (by omega)]
    simpa only [parameters,position,runTiles] using
      tile_success M p n (runTiles M p s n)

theorem complete [Inhabited α] (M : Algebra α) (p : Parameters)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit)
    (s : State α) (n capacity : Nat) (rows cache : Nat → Nat)
    (input : 0<p.count → GatherInput p s n capacity rows cache) :
    ∃ t,KernelSource.launch M p s (blocks p.count)=some t ∧
      (∀ o,o<p.count → t.mem p.output o=.mk .nat (byteTarget p rows cache o)) ∧
      (∀ (r : RegionName) o,r≠p.output ∨ p.count≤o → t.mem r o=s.mem r o) := by
  obtain ⟨values,frame⟩ := gather_correct_total M p s n capacity rows cache input
  exact ⟨gather M p s,launch_success M p start block grid s _ (Nat.le_refl _),values,frame⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.CacheTransform.KernelInterface
