import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.HistoricalKernels

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

theorem forward_launch_output (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (base : Nat) (schedule : List Nat)
    (covers : schedule.Perm (List.range (OutputKind.forward.count s)))
    (fresh : ∀ i,i<OutputKind.forward.count s → base+i∉forwardInputCells s b)
    (m : Memory α) (i : Nat) (hi : i<OutputKind.forward.count s) :
    launch a p s b .forward base schedule m (base+i)=
      scratchValue a p s .forward (currentInputs a s b m) i := by
  apply actual_store_value (forwardInputCells s b) (fun i => base+i)
    (scratchCurrent a p s b .forward) (forward_scratch_local a p s b) schedule m
  · intro j hj
    exact fresh j (List.mem_range.mp (covers.mem_iff.mp hj))
  · exact covers.mem_iff.mpr (List.mem_range.mpr hi)
  · intro i j hij
    dsimp only at hij
    omega

theorem forward_whole_output (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (tree : SumPlan) (scratch output : Nat) (schedule : List Nat)
    (covers : schedule.Perm (List.range (OutputKind.forward.count s)))
    (scratch_fresh : ∀ i, i<OutputKind.forward.count s → scratch+i∉forwardInputCells s b)
    (output_fresh : ∀ i, i<OutputKind.forward.cells s → output+i∉scratchCells (OutputKind.forward.shards s) scratch (OutputKind.forward.cells s))
    (m : Memory α) (i : Nat) (hi : i<OutputKind.forward.cells s) :
    wholeRun a p s b OutputKind.forward tree scratch output schedule m (output+i)=
      wordReference a p s OutputKind.forward tree (currentInputs a s b m) i := by
  unfold wholeRun
  rw [host_output a tree (OutputKind.forward.shards s) scratch output (OutputKind.forward.cells s) _ output_fresh i hi]
  have eq : (fun shard => if shard<OutputKind.forward.shards s ∧ i<OutputKind.forward.cells s then
      launch a p s b OutputKind.forward scratch schedule m (scratch+shard*OutputKind.forward.cells s+i) else a.zero)=
    (fun shard => if shard<OutputKind.forward.shards s ∧ i<OutputKind.forward.cells s then
      scratchValue a p s OutputKind.forward (currentInputs a s b m) (shard*OutputKind.forward.cells s+i) else a.zero) := by
    funext shard
    split_ifs with h
    · have bound : shard*OutputKind.forward.cells s+i<OutputKind.forward.count s := by
        change shard*OutputKind.forward.cells s+i<OutputKind.forward.shards s*OutputKind.forward.cells s
        nlinarith
      simpa [Nat.add_assoc] using forward_launch_output a p s b scratch schedule covers scratch_fresh m _ bound
    · rfl
  simpa only [hostValue,wordReference,Nat.add_assoc] using
    congrArg (fun x : Nat → α => a.cast (tree.eval a x)) eq


/-- Source execution reads current input memory and the actual host scratch
reduction; this result is the returned tensor read, not a detached lane value. -/
theorem forward_api_word (abi : LocalRuntime α σ) (a : Ops α) (p : Plans)
    (s : Shape) (b : Bases) (tree : SumPlan) (layout : ForwardLayout s b)
    (saved : SavedInputs s b) (state : σ) (i : LogicalIndex s .forward) :
    let result := forwardAPI abi a p s b tree layout saved state
    readReturned s .forward result.output (abi.observe result.state) i=
      wordReference a p s .forward tree (currentInputs a s b (abi.observe state))
        (dense s.T s.DV i.bh i.time i.feature) := by
  dsimp [forwardAPI,forwardReturn,readReturned,outputHandle]
  rw [runtimeForward_observe]
  exact forward_whole_output a p s b tree layout.scratch layout.output (forwardSchedule s)
    (List.Perm.refl _) layout.scratchFresh layout.outputFresh (abi.observe state) _
    (dense_bound _ _ _ _ _ _ i.bh_bound i.time.isLt i.feature.isLt)

theorem backward_api_word (abi : LocalRuntime α σ) (a : Ops α) (p : Plans)
    (s : Shape) (b : Bases) (trees : OutputKind → SumPlan) (layout : BackwardLayout s b)
    (context : SavedContext) (state : σ) (kind : OutputKind) (active : kind≠.forward)
    (i : LogicalIndex s kind) :
    let result := backwardAPI abi a p s b trees layout context state
    readReturned s kind (result.handle kind) (abi.observe result.state) i=
      wordReference a p s kind (trees kind) (currentInputs a s b (abi.observe state))
        (dense s.T (kind.width s) i.bh i.time i.feature) := by
  dsimp only
  rw [backward_handle abi a p s b trees layout context state kind active]
  unfold readReturned
  dsimp only [backwardAPI,outputHandle]
  rw [runtimeBackward_observe]
  exact backward_whole_output a p s b trees layout.scratch layout.output
    (backwardSchedule s) (backwardHostSchedule s) (List.Perm.refl _) (List.Perm.refl _)
    layout.scratchSeparate layout.outputSeparate layout.scratchFresh layout.outputFresh
    (abi.observe state) ⟨kind,dense s.T (kind.width s) i.bh i.time i.feature⟩
    ((allCells_member s false _).mpr ⟨active,
      dense_bound _ _ _ _ _ _ i.bh_bound i.time.isLt i.feature.isLt⟩)

theorem forward_api_frame (abi : LocalRuntime α σ) (a : Ops α) (p : Plans)
    (s : Shape) (b : Bases) (tree : SumPlan) (layout : ForwardLayout s b)
    (saved : SavedInputs s b) (state : σ) (address : Nat)
    (scratchOutside : address<layout.scratch ∨ layout.scratch+OutputKind.forward.count s≤address)
    (outputOutside : address<layout.output ∨ layout.output+OutputKind.forward.cells s≤address) :
    abi.observe (forwardAPI abi a p s b tree layout saved state).state address=abi.observe state address := by
  dsimp only [forwardAPI]
  rw [runtimeForward_observe]
  exact whole_frame a p s b .forward tree layout.scratch layout.output (forwardSchedule s)
    (List.Perm.refl _) (abi.observe state) address scratchOutside outputOutside

theorem backward_api_frame (abi : LocalRuntime α σ) (a : Ops α) (p : Plans)
    (s : Shape) (b : Bases) (trees : OutputKind → SumPlan) (layout : BackwardLayout s b)
    (context : SavedContext) (state : σ) (address : Nat)
    (scratchOutside : ∀ cell∈allCells s true,cellsAddress layout.scratch cell≠address)
    (outputOutside : ∀ cell∈allCells s false,cellsAddress layout.output cell≠address) :
    abi.observe (backwardAPI abi a p s b trees layout context state).state address=abi.observe state address := by
  dsimp only [backwardAPI]
  rw [runtimeBackward_observe]
  exact backward_whole_frame a p s b trees layout.scratch layout.output
    (backwardSchedule s) (backwardHostSchedule s) (abi.observe state) address scratchOutside outputOutside


#axiomsClean forward_api_word
#axiomsClean backward_api_word
#axiomsClean forward_api_frame
#axiomsClean backward_api_frame

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
