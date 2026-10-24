import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedRealGlobal

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxRecDepth 30000

def atProgram (s : State Value) (pid : Nat) : State Value :=
  {s with pids:=fun ax => if ax=0 then pid else s.pids ax}

inductive GridRun (ops : Algebra Value) (code : ComputeKernel) : List Nat → State Value → State Value → Prop
  | nil (s) : GridRun ops code [] s s
  | cons {pid jobs s t u} : FP.Structural.exec ops code (atProgram s pid)=some t →
      GridRun ops code jobs t u → GridRun ops code (pid::jobs) s u

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

theorem first_snapshot (kind : TypedScratch.Kind) (rest : Algebra Value) (s : State Value)
    (X V : RegionName) (I : Region .int) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (input : Nat → ℝ) (pid : Nat) (active : pid<c.blocks) (index64 : Bool)
    (reads : ∀ i,i<c.size → (s.mem X i).read .real=(input i : Value)) :
    ∃ t, FP.Structural.exec (backend rest) (TypedScratch.firstKernel kind X V I c.size c.block 0 index64)
      (atProgram s pid)=some t ∧ t.mem=(snapshotStore kind V I c input s pid).mem := by
  obtain ⟨t,run,mem⟩ := stage1_run kind rest (atProgram s pid) X V I c.size c.block 0 index64 coverage.blockPositive
  have selected := first_reference (adapter:=.directCast kind.direct) (atProgram s pid) X c coverage 0 input
    (by simpa [atProgram] using active) reads
  refine ⟨t,run,mem.trans ?_⟩
  unfold firstWritten
  rw [selected.1]
  have valid := Math.block_valid c.size (pid*c.block) c.block input (c.firstTree pid)
    (coverage.firstComplete pid active) (coverage.eachActive pid active) coverage.blockPositive
  change Math.firstIndex c input pid<c.size at valid
  simp only [atProgram,ite_true,Nat.add_zero]
  rw [reads _ valid]
  rfl

theorem grid_run (kind : TypedScratch.Kind) (rest : Algebra Value) (s : State Value)
    (X V : RegionName) (I : Region .int) (inputValue : X≠V) (inputIndex : X≠Region.cast I)
    (c : GlobalConfig) (coverage : Math.GlobalCoverage c) (input : Nat → ℝ)
    (jobs : List Nat) (active : ∀ pid,pid∈jobs → pid<c.blocks) (index64 : Bool)
    (reads : ∀ i,i<c.size → (s.mem X i).read .real=(input i : Value)) :
    ∃ t, GridRun (backend rest) (TypedScratch.firstKernel kind X V I c.size c.block 0 index64) jobs s t ∧
      t.mem=(jobs.foldl (snapshotStore kind V I c input) s).mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,.nil s,rfl⟩
  | cons pid jobs ih =>
    obtain ⟨a,run,mem⟩ := first_snapshot kind rest s X V I c coverage input pid (active pid (by simp)) index64 reads
    have nextReads : ∀ i,i<c.size → (a.mem X i).read .real=(input i : Value) := by
      intro i valid
      rw [mem]
      simpa [snapshotStore,State.write,inputValue,inputIndex] using reads i valid
    obtain ⟨t,runs,mt⟩ := ih a (fun j hj => active j (by simp [hj])) nextReads
    exact ⟨t,.cons run runs,mt.trans (snapshot_memory kind V I c input jobs a _ mem)⟩

/-- Both global launches execute the complete shared quotations. Scratch
load facts are derived from the live first-grid stores. The target quantifies
over arbitrary logical Reals independently of the selected reduction trees. -/
theorem global_source_reference (kind : TypedScratch.Kind) (rest : Algebra Value) (s : State Value)
    (X V : RegionName) (I O : Region .int) (inputValue : X≠V) (inputIndex : X≠Region.cast I)
    (scratchApart : V≠Region.cast I) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (input : Nat → ℝ) (index64 : Bool)
    (reads : ∀ i,i<c.size → (s.mem X i).read .real=(input i : Value)) :
    ∃ middle final,
      GridRun (backend rest) (TypedScratch.firstKernel kind X V I c.size c.block 0 index64)
        (List.range c.blocks) s middle ∧
      FP.Structural.exec (backend rest) (TypedScratch.secondKernel kind V I O c.blocks c.midBlock) middle=some final ∧
      final.mem (Region.cast O) 0=.mk .int (Int.ofNat
        (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => input i.val)).val) ∧
      (∀ region offset,(region≠Region.cast O ∨ offset≠0) →
        (∀ pid,pid<c.blocks → (region≠V ∨ offset≠pid) ∧ (region≠Region.cast I ∨ offset≠pid)) →
        final.mem region offset=s.mem region offset) := by
  obtain ⟨middle,first,mem⟩ := grid_run kind rest s X V I inputValue inputIndex c coverage input
    (List.range c.blocks) (fun _ h => List.mem_range.mp h) index64 reads
  have values : ∀ i,i<c.blocks → toFloat (TypedScratch.channel kind)
      ((middle.mem V i).read (TypedScratch.channel kind).toTileDType)=(input (Math.firstIndex c input i) : Value) := by
    intro i active
    rw [mem,snapshot_values kind V I scratchApart c input _ s i,if_pos (List.mem_range.mpr active)]
    cases kind <;> rfl
  have indices : ∀ i,i<c.blocks → (middle.mem (Region.cast I) i).read .int=Int.ofNat (Math.firstIndex c input i) := by
    intro i active
    rw [mem,snapshot_indices kind V I scratchApart c input _ s i,if_pos (List.mem_range.mpr active)]
    rfl
  obtain ⟨final,second,output⟩ := second_reference kind rest middle V I O c coverage input values indices
  refine ⟨middle,final,first,second,?_,?_⟩
  · rw [output,State.write_same]
  · intro region offset out scratch
    rw [output,State.write_other _ _ _ _ _ _ out,mem]
    exact snapshot_frame kind V I c input _ s region offset (fun pid hp => scratch pid (List.mem_range.mp hp))

#axiomsClean grid_run
#axiomsClean global_source_reference
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
