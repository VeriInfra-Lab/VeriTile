import bench.tritonbench_g.triton_argmax.Proofs.TypedRealGlobal

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxRecDepth 30000

def atProgram (s : State Value) (pid : Nat) : State Value :=
  {s with pids:=fun ax => if ax=0 then pid else s.pids ax}

def snapshotStore (kind : TypedScratch.Kind) (V : RegionName) (I : Region .int)
    (c : GlobalConfig) (input : Nat → ℝ) (s : State Value) (pid : Nat) : State Value :=
  (s.write V pid (.mk (TypedScratch.channel kind).toTileDType
    (ofFloat (TypedScratch.channel kind) (input (Math.firstIndex c input pid) : Value)))).write
      (Region.cast I) pid (.mk .int (Int.ofNat (Math.firstIndex c input pid)))

theorem snapshot_memory (kind : TypedScratch.Kind) (V : RegionName) (I : Region .int)
    (c : GlobalConfig) (input : Nat → ℝ) (jobs : List Nat) (s t : State Value) (same : s.mem=t.mem) :
    (jobs.foldl (snapshotStore kind V I c input) s).mem=(jobs.foldl (snapshotStore kind V I c input) t).mem := by
  induction jobs generalizing s t with
  | nil => exact same
  | cons pid jobs ih =>
    simp only [List.foldl_cons]
    apply ih
    simp only [snapshotStore,State.write,same]

theorem snapshot_frame (kind : TypedScratch.Kind) (V : RegionName) (I : Region .int)
    (c : GlobalConfig) (input : Nat → ℝ) (jobs : List Nat) (s : State Value)
    (region : RegionName) (offset : Nat)
    (outside : ∀ pid,pid∈jobs → (region≠V ∨ offset≠pid) ∧ (region≠Region.cast I ∨ offset≠pid)) :
    (jobs.foldl (snapshotStore kind V I c input) s).mem region offset=s.mem region offset := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    rw [List.foldl_cons,ih _ (fun j hj => outside j (by simp [hj]))]
    obtain ⟨hv,hi⟩ := outside pid (by simp)
    unfold snapshotStore
    rw [State.write_other _ _ _ _ _ _ hi,State.write_other _ _ _ _ _ _ hv]

theorem snapshot_values (kind : TypedScratch.Kind) (V : RegionName) (I : Region .int)
    (apart : V≠Region.cast I) (c : GlobalConfig) (input : Nat → ℝ) (jobs : List Nat)
    (s : State Value) (pid : Nat) :
    (jobs.foldl (snapshotStore kind V I c input) s).mem V pid=
      if pid∈jobs then .mk (TypedScratch.channel kind).toTileDType
        (ofFloat (TypedScratch.channel kind) (input (Math.firstIndex c input pid) : Value)) else s.mem V pid := by
  induction jobs generalizing s with
  | nil => simp
  | cons j jobs ih =>
    rw [List.foldl_cons,ih]
    by_cases member : pid∈jobs
    · simp [member]
    · by_cases same : pid=j
      · subst pid; simp [member,snapshotStore,State.write,apart]
      · simp [member,same,snapshotStore,State.write,apart]

theorem snapshot_indices (kind : TypedScratch.Kind) (V : RegionName) (I : Region .int)
    (apart : V≠Region.cast I) (c : GlobalConfig) (input : Nat → ℝ) (jobs : List Nat)
    (s : State Value) (pid : Nat) :
    (jobs.foldl (snapshotStore kind V I c input) s).mem (Region.cast I) pid=
      if pid∈jobs then .mk .int (Int.ofNat (Math.firstIndex c input pid)) else s.mem (Region.cast I) pid := by
  induction jobs generalizing s with
  | nil => simp
  | cons j jobs ih =>
    rw [List.foldl_cons,ih]
    by_cases member : pid∈jobs
    · simp [member]
    · by_cases same : pid=j
      · subst pid; simp [member,snapshotStore,State.write,apart]
      · simp [member,same,snapshotStore,State.write,Ne.symm apart]

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
