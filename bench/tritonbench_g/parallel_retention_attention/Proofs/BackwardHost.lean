import bench.tritonbench_g.parallel_retention_attention.Proofs.Schedule
import bench.tritonbench_g.parallel_retention_attention.Proofs.Host

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

def cellCount (s : Shape) (scratch : Bool) (kind : OutputKind) :=
  if scratch then kind.count s else kind.cells s

theorem allCells_member (s : Shape) (scratch : Bool) (cell : Cell) :
    cell∈allCells s scratch ↔ cell.kind≠.forward ∧ cell.index<cellCount s scratch cell.kind := by
  rcases cell with ⟨kind,i⟩
  cases kind <;> simp [allCells,cellCount,OutputKind.cells]

def outputWindow (s : Shape) (scratch : Bool) (bases : OutputKind → Nat) (kind : OutputKind) : Window :=
  ⟨bases kind,cellCount s scratch kind⟩

def RegionDisjoint (s : Shape) (scratch : Bool) (bases : OutputKind → Nat) : Prop :=
  ∀ k l,k≠.forward → l≠.forward → k≠l → Disjoint (outputWindow s scratch bases k).Cells (outputWindow s scratch bases l).Cells

theorem cells_unique (s : Shape) (scratch : Bool) (bases : OutputKind → Nat)
    (separate : RegionDisjoint s scratch bases) (target cell : Cell)
    (ht : target∈allCells s scratch) (hc : cell∈allCells s scratch)
    (eq : cellsAddress bases cell=cellsAddress bases target) : cell=target := by
  have ht' := (allCells_member s scratch target).mp ht
  have hc' := (allCells_member s scratch cell).mp hc
  have hk : cell.kind=target.kind := by
    by_contra hn
    have dis := separate cell.kind target.kind hc'.1 ht'.1 hn
    rw [Set.disjoint_left] at dis
    apply dis (a := cellsAddress bases cell)
    · simp only [Window.Cells,outputWindow,Set.mem_setOf_eq,cellsAddress]
      omega
    · rw [eq]
      simp only [Window.Cells,outputWindow,Set.mem_setOf_eq,cellsAddress]
      omega
  have hi : cell.index=target.index := by unfold cellsAddress at eq;rw [hk] at eq;omega
  cases cell;cases target
  simp_all

def backwardScratchCells (s : Shape) (scratchBases : OutputKind → Nat) : Set Nat :=
  {address | ∃ kind,kind≠.forward ∧ address∈scratchCells (kind.shards s) (scratchBases kind) (kind.cells s)}

def backwardHostValue (a : Ops α) (s : Shape) (trees : OutputKind → SumPlan)
    (scratchBases : OutputKind → Nat) (m : Memory α) (cell : Cell) : α :=
  if cell.kind=.forward then a.zero else
    hostValue a (trees cell.kind) (cell.kind.shards s) (scratchBases cell.kind) (cell.kind.cells s) m cell.index

theorem backward_host_local (a : Ops α) (s : Shape) (trees : OutputKind → SumPlan)
    (scratchBases : OutputKind → Nat) (m m' : Memory α)
    (same : Set.EqOn m m' (backwardScratchCells s scratchBases)) (cell : Cell) :
    backwardHostValue a s trees scratchBases m cell=backwardHostValue a s trees scratchBases m' cell := by
  unfold backwardHostValue
  split_ifs with h
  · rfl
  · apply host_local
    intro address ha
    exact same ⟨cell.kind,h,ha⟩

def backwardHostRun (a : Ops α) (s : Shape) (trees : OutputKind → SumPlan)
    (scratchBases outputBases : OutputKind → Nat) (schedule : List Cell) (m : Memory α) :=
  runWrites (cellsAddress outputBases) (backwardHostValue a s trees scratchBases) schedule m

def backwardWhole (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (trees : OutputKind → SumPlan)
    (scratchBases outputBases : OutputKind → Nat) (scratchSchedule hostSchedule : List Cell) (m : Memory α) :=
  backwardHostRun a s trees scratchBases outputBases hostSchedule
    (backwardLaunch a p s b scratchBases scratchSchedule m)

/-- Three returned gradients consume the actual shared backward launch memory,
regardless of inter-program and inter-output store interleaving. -/
theorem backward_whole_output (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratchBases outputBases : OutputKind → Nat)
    (scratchSchedule hostSchedule : List Cell)
    (scratchCovers : scratchSchedule.Perm (allCells s true))
    (hostCovers : hostSchedule.Perm (allCells s false))
    (scratchSeparate : RegionDisjoint s true scratchBases)
    (outputSeparate : RegionDisjoint s false outputBases)
    (scratchFresh : ∀ cell∈allCells s true,cellsAddress scratchBases cell∉inputCells s b)
    (outputFresh : ∀ cell∈allCells s false,cellsAddress outputBases cell∉backwardScratchCells s scratchBases)
    (m : Memory α) (target : Cell) (active : target∈allCells s false) :
    backwardWhole a p s b trees scratchBases outputBases scratchSchedule hostSchedule m (cellsAddress outputBases target)=
      wordReference a p s target.kind (trees target.kind) (currentInputs a s b m) target.index := by
  unfold backwardWhole backwardHostRun
  rw [actual_write_value (backwardScratchCells s scratchBases) (cellsAddress outputBases)
    (backwardHostValue a s trees scratchBases) (backward_host_local a s trees scratchBases)
    hostSchedule _ (by intro cell hc;exact outputFresh cell (hostCovers.mem_iff.mp hc))
    target (hostCovers.mem_iff.mpr active) (by
      intro cell hc eq
      exact cells_unique s false outputBases outputSeparate target cell active (hostCovers.mem_iff.mp hc) eq)]
  have active' := (allCells_member s false target).mp active
  unfold backwardHostValue
  rw [if_neg active'.1]
  have eq : (fun shard => if shard<target.kind.shards s ∧ target.index<target.kind.cells s then
      backwardLaunch a p s b scratchBases scratchSchedule m (scratchBases target.kind+shard*target.kind.cells s+target.index) else a.zero)=
    (fun shard => if shard<target.kind.shards s ∧ target.index<target.kind.cells s then
      scratchValue a p s target.kind (currentInputs a s b m) (shard*target.kind.cells s+target.index) else a.zero) := by
    funext shard
    split_ifs with h
    · let cell : Cell := ⟨target.kind,shard*target.kind.cells s+target.index⟩
      have hc : cell∈allCells s true := by
        apply (allCells_member s true cell).mpr
        refine ⟨active'.1,?_⟩
        change shard*target.kind.cells s+target.index<target.kind.shards s*target.kind.cells s
        nlinarith
      simpa only [cell,cellsAddress,Nat.add_assoc] using
        backward_launch_output a p s b scratchBases scratchSchedule m
          (by intro c hm;exact scratchFresh c (scratchCovers.mem_iff.mp hm)) cell
          (scratchCovers.mem_iff.mpr hc) (by
            intro c hm heq
            exact cells_unique s true scratchBases scratchSeparate cell c hc (scratchCovers.mem_iff.mp hm) heq)
    · rfl
  simpa only [hostValue,wordReference,Nat.add_assoc] using
    congrArg (fun x : Nat → α => a.cast ((trees target.kind).eval a x)) eq

theorem backward_whole_frame (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratchBases outputBases : OutputKind → Nat)
    (scratchSchedule hostSchedule : List Cell) (m : Memory α) (address : Nat)
    (scratchOutside : ∀ cell∈scratchSchedule,cellsAddress scratchBases cell≠address)
    (outputOutside : ∀ cell∈hostSchedule,cellsAddress outputBases cell≠address) :
    backwardWhole a p s b trees scratchBases outputBases scratchSchedule hostSchedule m address=m address := by
  unfold backwardWhole backwardHostRun
  rw [runWrites_frame _ _ _ _ _ outputOutside]
  exact backward_launch_frame _ _ _ _ _ _ _ _ scratchOutside

#axiomsClean allCells_member
#axiomsClean cells_unique
#axiomsClean backward_host_local
#axiomsClean backward_whole_output
#axiomsClean backward_whole_frame
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
