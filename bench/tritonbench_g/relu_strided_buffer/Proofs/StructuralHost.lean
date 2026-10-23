import bench.tritonbench_g.relu_strided_buffer.Proofs.StructuralControl
import bench.tritonbench_g.relu_strided_buffer.Proofs.StructuralTile
import bench.tritonbench_g.relu_strided_buffer.Proofs.HostPlan

/-!
Actual selected-FP32 source execution on an opaque floating carrier, from the
original wrapper preparation through both constexpr branches and all CTAs.
The local source, loop and launch proofs are discharged, and outputs use the
entry memory. Scalar casts and strict comparison remain explicit. Specializing
the algebra to raw FP32 words does not change this source or its control flow.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralHost
open VeriTile.Triton FP.Structural SignedView SignedGrid HostPlan
open StructuralGrid StructuralControl StructuralTile SourceSignedBranch
noncomputable section

theorem tile_implementation {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLt none .fp32 = some cmp) (c : SignedViews) :
    TileImplementation M (tileOutput M cmp c) (c.perTile.map ComputeStmt.alg) := by
  intro T s ht
  obtain ⟨final, hexec, hmem, hundef, hpid, hctas, _, _⟩ := perTile_run M cmp hc c T s ht
  exact ⟨final, hexec, ⟨hmem, hundef⟩, hpid, hctas⟩

theorem tile_contract {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : SignedViews) {capacity : Nat} (hout : c.outputView.Valid c.n capacity)
    (hstride : c.outputView.stride ≠ 0) :
    TileContract c.inputView c.outputView c.n c.B
      (fun x => Cell.mk .fp32 (scalarResult M cmp x)) (flatTile (tileOutput M cmp c)) := by
  intro T s
  refine ⟨flatState (tileOutput M cmp c) T s, rfl, ?_, ?_⟩
  · exact fun lane hactive => tileOutput_readback M cmp c T s.mem hout hstride lane hactive
  · exact fun r a outside => tileOutput_frame M cmp c T s.mem r a outside

def kernelIds {α : Type} (oneTile : Bool) (tiles : Nat) (s : State α) : List Nat :=
  if oneTile then [s.pids 0]
  else (List.range tiles).map (fun j => s.pids 0 + j * s.numPids 0)

/-- The structural source projects to SourceGridExecution.kernelBody, and this
theorem runs it. There is no per-tile, loop or complete-result premise. -/
theorem kernel_execution {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLt none .fp32 = some cmp)
    (c : SignedViews) (oneTile : Bool) (tiles : Nat) (s : State α) :
    ∃ actualFinal flatFinal,
      exec M (kernelSource c oneTile tiles) s = some actualFinal ∧
      runTiles (flatTile (tileOutput M cmp c)) (kernelIds oneTile tiles s) s = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
  cases oneTile
  · exact grid_branch_execution M (tileOutput M cmp c) (c.perTile.map ComputeStmt.alg)
      (tile_implementation M cmp hc c) c.n c.B tiles s
  · obtain ⟨actualFinal, hexec, hobs⟩ :=
      one_tile_execution M (tileOutput M cmp c) (c.perTile.map ComputeStmt.alg)
        (tile_implementation M cmp hc c) c.n c.B s
    exact ⟨actualFinal, flatState (tileOutput M cmp c) (s.pids 0) s, hexec, rfl, hobs⟩

theorem prepared_ids (input output : Buffer) (n : Nat) (sx sy : Int) :
    sourceTileIds (rankOneArguments input output n sx sy) = cappedTileIds n (block n) := by
  change (List.range (CappedGrid.numCTAs n (block n))).flatMap
    (fun pid => idsPerProgram (rankOneArguments input output n sx sy) pid) = _
  by_cases hone : CappedGrid.tilesPerCTA n (block n) = 1
  · simp [idsPerProgram, rankOneArguments, hone, cappedTileIds]
  · simp [idsPerProgram, rankOneArguments, hone, cappedTileIds]

theorem arguments_observations {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLt none .fp32 = some cmp)
    (a : Arguments) (s : State α) :
    ∃ actualFinal flatFinal, executeArguments M a s = some actualFinal ∧
      runTiles (flatTile (tileOutput M cmp (views a))) (sourceTileIds a) s = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
  have hbody : ∀ pid state, ∃ actualFinal flatFinal,
      run M (kernelSource (views a) a.oneTile a.T).surfaceBody
        (atProgram a.C pid state) = some actualFinal ∧
      runTiles (flatTile (tileOutput M cmp (views a))) (idsPerProgram a pid)
        (atProgram a.C pid state) = some flatFinal ∧ ObsEq actualFinal flatFinal := by
    intro pid state
    simpa only [FP.Structural.exec, kernelSource, ComputeKernel.surfaceBody, kernelIds, atProgram,
      if_true, idsPerProgram] using
      kernel_execution M cmp hc (views a) a.oneTile a.T (atProgram a.C pid state)
  exact programs_execution M (tileOutput M cmp (views a)) a.C
    (kernelSource (views a) a.oneTile a.T).surfaceBody (idsPerProgram a) hbody
    (List.range a.C) s s (ObsEq.refl s)

/-- Complete rank-one opaque source execution and exact raw-cell frame. The
applicability conditions are the original valid signed views, disjoint input
and output footprints, nonempty rank one, and successful device guard. -/
theorem rank_one_success {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLt none .fp32 = some cmp) (input output : Buffer)
    {n inputCapacity outputCapacity : Nat} (hn : 0 < n) (sx sy : Int)
    (hshapeIn : input.shape = [n]) (hshapeOut : output.shape = [n])
    (hstrideIn : input.strides = [sx]) (hstrideOut : output.strides = [sy])
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : FootprintsDisjoint (input.view sx) (output.view sy) n) (s : State α) :
    ∃ final, execute M input output Bool.true s = .returned output final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (scalarResult M cmp
          ((s.mem input.region ((input.view sx).address i)).read .fp32))) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧
        (output.view sy).address i < outputCapacity) := by
  let a := rankOneArguments input output n sx sy
  let c := views a
  obtain ⟨flat, hflat, hvalues, hinput, hframe, hbounds⟩ :=
    capped_grid_success hin hout hstride (block_pos hn) hsep
      (fun x => Cell.mk .fp32 (scalarResult M cmp x)) (flatTile (tileOutput M cmp c))
      (tile_contract M cmp c hout hstride) (cappedTileIds n (block n)) (List.Perm.refl _) s
  obtain ⟨actual, flat', hactual, hflat', hobs⟩ := arguments_observations M cmp hc a s
  have heq : flat = flat' := Option.some.inj (hflat.symm.trans (by
    simpa only [a, prepared_ids] using hflat'))
  subst flat'
  change executeArguments M (rankOneArguments input output n sx sy) s = some actual at hactual
  refine ⟨actual, ?_, ?_, ?_, ?_, hbounds⟩
  · simp only [execute, rank_one_prepare input output hn sx sy
      hshapeIn hshapeOut hstrideIn hstrideOut, hactual]
  · simpa only [hobs.1] using hvalues
  · simpa only [hobs.1] using hinput
  · simpa only [hobs.1] using hframe

theorem failure_preserves_memory {α : Type} [Inhabited α] (M : Algebra α)
    (input output : Buffer) (guardSucceeds : Bool) (s : State α) (failure : Failure)
    (h : prepare input output guardSucceeds = .error failure) :
    execute M input output guardSucceeds s = .hostFailure failure s := by
  simp only [execute, h]

theorem empty_failure {α : Type} [Inhabited α] (M : Algebra α)
    (input output : Buffer) (guardSucceeds : Bool) (s : State α)
    (hin : input.shape = [0]) (hout : output.shape = [0]) :
    execute M input output guardSucceeds s = .hostFailure .tileDivision s :=
  failure_preserves_memory M input output guardSucceeds s .tileDivision
    (empty_before_guard input output guardSucceeds hin hout)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralHost
