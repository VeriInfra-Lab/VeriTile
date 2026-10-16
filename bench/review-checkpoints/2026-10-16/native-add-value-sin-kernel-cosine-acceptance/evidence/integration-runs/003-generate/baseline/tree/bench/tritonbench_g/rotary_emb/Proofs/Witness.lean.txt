import bench.tritonbench_g.rotary_emb.Proofs.Contracts

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullIO
open VeriTile.Triton FullMath

/-- Four real allocations with room for a noncontiguous two-sequence,
two-head, two-pair block. This witness does not assume a successful run.
-/
def witnessAllocation : FlatAlloc :=
  FlatAlloc.ofList "flat" [("Q",32), ("K",32), ("Cos",32), ("Sin",32)]

theorem witness_disjoint : witnessAllocation.Disjoint :=
  FlatAlloc.ofList_disjoint _ _ (by decide)

theorem witness_bounds (s : BlockState) (h0 : s.pids 0 = 0) (h1 : s.pids 1 = 0) :
    exampleParameters.Bounds s witnessAllocation.extent "Q" "K" "Cos" "Sin" := by
  obtain ⟨hq, hk, ht⟩ := example_bounds s h0 h1
  exact ⟨hq, hk, ht⟩

/-- Every lane of the example block is active in both buffers, so the domain
does not conceal an empty shape or an all-false store mask. -/
theorem witness_active (s : BlockState) (h0 : s.pids 0 = 0) (h1 : s.pids 1 = 0)
    (i : exampleParameters.Lane) :
    exampleParameters.activeQ s i ∧ exampleParameters.activeK s i := by
  have hs := i.1.isLt
  have hh := i.2.1.isLt
  simpa only [exampleParameters, Parameters.activeQ, Parameters.activeK,
    FullMath.active, FullMath.sequence, FullMath.head, h0, h1, Nat.zero_mul, Nat.zero_add] using
      And.intro (And.intro hs hh) (And.intro hs hh)

/-- All six independent storage choices and both implementations have a
nonempty successful flat execution satisfying both output formulas and the
complete frame. The initial memory is arbitrary; only program IDs and the
flat bridge's undefined-value convention are fixed.
-/
theorem witness_execution (q k cos sin : FullSource.Storage) (v : Bool)
    (s : BlockState) (h0 : s.pids 0 = 0) (h1 : s.pids 1 = 0)
    (hu : s.undef = fun _ _ ↦ 0) :
    (∃ i : exampleParameters.Lane,
      exampleParameters.activeQ s i ∧ exampleParameters.activeK s i) ∧
    ∃ t, exec (witnessAllocation.flattenKernel
      (sourceIO exampleParameters q k cos sin v "Q" "K" "Cos" "Sin").kernel.eraseDType.toAlgKernel)
      (witnessAllocation.flattenState s) = some t ∧
      (sourceIO exampleParameters q k cos sin v "Q" "K" "Cos" "Sin").FlatResult
        witnessAllocation s t (expected exampleParameters "Q" "K" "Cos" "Sin") := by
  constructor
  · refine ⟨(⟨0, by decide⟩, ⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit), ?_⟩
    exact witness_active s h0 h1 _
  · have h := source_implements exampleParameters q k cos sin v "Q" "K" "Cos" "Sin"
      witnessAllocation witness_disjoint (FlatAlloc.ofList_closed _ _) s
      regions_nonempty (example_geometry s) (witness_bounds s h0 h1)
      (by simp [sourceIO, witnessAllocation]) (by simp [sourceIO, witnessAllocation])
    exact h.2.2 hu

/-- Closed existential witness: no state, allocation or successful-execution
premise is assumed. Both output masks contain active lanes. -/
theorem witness_exists (q k cos sin : FullSource.Storage) (v : Bool) :
    ∃ s : BlockState,
      (∃ i : exampleParameters.Lane,
        exampleParameters.activeQ s i ∧ exampleParameters.activeK s i) ∧
      ∃ t, exec (witnessAllocation.flattenKernel
        (sourceIO exampleParameters q k cos sin v "Q" "K" "Cos" "Sin").kernel.eraseDType.toAlgKernel)
        (witnessAllocation.flattenState s) = some t ∧
        (sourceIO exampleParameters q k cos sin v "Q" "K" "Cos" "Sin").FlatResult
          witnessAllocation s t (expected exampleParameters "Q" "K" "Cos" "Sin") :=
  ⟨default, witness_execution q k cos sin v default rfl rfl rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullIO
