import bench.tritonbench_g.adam_update_triton.Proofs.GridReuse

/-! A small extension of the frozen mathematical grid reuse. The existing
mergeFrames_mem_unwritten theorem preserves the complete memory cell, not only
its real projection. No kernel execution proof is repeated here. The host,
autotuner, typed-value, and native boundaries of GridReuse remain unchanged. -/

namespace VeriTile.Bench.TritonBenchG.AdamUpdateTriton.GridReuse
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Direct frame reuse for the example's constructed final memory. Every cell
outside the two active output prefixes retains its exact initial cell. This
also covers masked tail lanes, other regions, and the entire empty launch. -/
theorem merged_grid_frame (P G M : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B count : Nat)
    (distinct : P ≠ M) (initial : BlockState) (region : RegionName) (offset : Nat)
    (outside : (region ≠ P ∧ region ≠ M) ∨ n ≤ offset) :
    (Kernel.mergeFrames
      (VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamGrid count) initial
      (fun idx => VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamFrame
        P G M lr wd beta1 beta2 n B count distinct initial idx)).mem region offset =
      initial.mem region offset := by
  apply Kernel.mergeFrames_mem_unwritten
  rintro ⟨idx, written⟩
  change VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamWritesFP
    P M n B (initial.withGridIndex idx) (region, offset) at written
  unfold VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamWritesFP
    WriteFootprint.union WriteFootprint.activeTileImage at written
  rcases written with ⟨hr, i, active, address⟩ | ⟨hr, i, active, address⟩
  · rcases outside with other | tail
    · exact other.1 hr
    · dsimp at active address
      omega
  · rcases outside with other | tail
    · exact other.2 hr
    · dsimp at active address
      omega

/-- The same final memory carries both full-array Lion formulas and an exact
raw-cell frame. This is a mathematical launch helper, not a full Python or
native correctness/acceptance declaration. -/
theorem selected_config_real_grid_with_frame (P G M : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat)
    (selected : B = 128 ∨ B = 1024) (distinct : P ≠ M) (initial : BlockState) :
    ∃ final : BlockState,
      Nonempty (Kernel.GridLaunchedOrdinary
        (update_fn_kernel P G M lr wd beta1 beta2 n B).toAlgKernel
        (VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamGrid ((n+B-1)/B)) initial final) ∧
      (∀ k, k < n →
        final.readMem P k =
          TiledOptimizer.lionParam (initial.readMem P k) (initial.readMem M k)
            (initial.readMem G k) lr wd beta1 ∧
        final.readMem M k =
          TiledOptimizer.lionMomentum (initial.readMem M k) (initial.readMem G k) beta2) ∧
      (∀ region offset, (region ≠ P ∧ region ≠ M) ∨ n ≤ offset →
        final.mem region offset = initial.mem region offset) := by
  have positive : 0 < B := by rcases selected with h | h <;> omega
  let count := (n+B-1)/B
  let final := Kernel.mergeFrames
    (VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamGrid count) initial
    (fun idx => VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamFrame
      P G M lr wd beta1 beta2 n B count distinct initial idx)
  refine ⟨final, ?_, ?_, ?_⟩
  · rw [real_projection_eq]
    exact ⟨VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamLaunch
      P G M lr wd beta1 beta2 n B count positive distinct initial⟩
  · exact VeriTile.Bench.Examples.AdamUpdateGridLaunch.update_fn_kernel_grid_correct_unconditional
      P G M lr wd beta1 beta2 n B positive distinct initial
  · exact fun region offset outside =>
      merged_grid_frame P G M lr wd beta1 beta2 n B count distinct initial region offset outside

#axiomsClean merged_grid_frame
#axiomsClean selected_config_real_grid_with_frame
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton.GridReuse
