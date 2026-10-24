import bench.tritonbench_g.triton_mul2.Proofs.FiniteSource

namespace TritonMul2Finite
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
noncomputable section
open Classical

/-- An independent request plan. Later blocks read the pointwise last-write
formula, which makes aliasing and in-place source order explicit. -/
def gridPlan (b : Numerics) (d : Storage) (p : Layout) (n block : Nat) :
    List Nat → ByteMemory → List Nat → Except Issue (List ByteWrite × List Nat)
  | [],snapshot,seen => .ok ([],seen)
  | pid::rest,snapshot,seen => do
      let (head,owners) ← programPlan b d p n block pid snapshot seen
      let (tail,finalOwners) ← gridPlan b d p n block rest (byteResult head snapshot) owners
      pure (head++tail,finalOwners)
def executeGrid (b : Numerics) (d : Storage) (p : Layout) (n block : Nat) :
    List Nat → State → Except Issue State
  | [],state => .ok state
  | pid::rest,state => do
      let next ← executeProgram b d p n block pid state
      executeGrid b d p n block rest next

theorem grid_correspondence (b : Numerics) (d : Storage) (p : Layout) (n block : Nat)
    (pids : List Nat) (state : State) : executeGrid b d p n block pids state=
    (gridPlan b d p n block pids state.memory state.owners).map
      (fun plan => ⟨applyWrites plan.1 state.memory,plan.2⟩) := by
  induction pids generalizing state with
  | nil => rfl
  | cons pid rest ih =>
    simp only [executeGrid,gridPlan,program_correspondence]
    cases hh : programPlan b d p n block pid state.memory state.owners with
    | error issue => rfl
    | ok head =>
      rcases head with ⟨head,owners⟩
      simp only [Except.map,bind_ok]
      rw [ih ⟨applyWrites head state.memory,owners⟩]
      simp only [writes_memory]
      cases gridPlan b d p n block rest (byteResult head state.memory) owners with
      | error issue => rfl
      | ok tail =>
        rcases tail with ⟨tail,finalOwners⟩
        simp only [Except.map,bind_ok,pure_ok]
        congr 1
        rw [←writes_memory,←writes_memory,←writes_append,writes_memory]

theorem grid_write_bounds (b : Numerics) (d : Storage) (p : Layout) (n block : Nat)
    (pids : List Nat) (snapshot : ByteMemory) (seen : List Nat) (writes : List ByteWrite) (owners : List Nat)
    (h : gridPlan b d p n block pids snapshot seen=.ok (writes,owners)) :
    ∀ w∈writes, p.output.Contains w.outputAddress := by
  induction pids generalizing snapshot seen writes owners with
  | nil => cases h; simp
  | cons pid rest ih =>
    simp only [gridPlan] at h
    cases hh : programPlan b d p n block pid snapshot seen with
    | error issue => simp [hh] at h
    | ok head =>
      rcases head with ⟨head,nextOwners⟩
      cases ht : gridPlan b d p n block rest (byteResult head snapshot) nextOwners with
      | error issue => simp [hh,ht] at h
      | ok tail =>
        rcases tail with ⟨tail,finalOwners⟩
        simp [hh,ht] at h
        rcases h with ⟨rfl,rfl⟩
        intro w hw
        rcases List.mem_append.mp hw with hw|hw
        · exact program_write_bounds b d p n block pid snapshot seen head nextOwners hh w hw
        · exact ih _ _ _ _ ht w hw

inductive Mode where | outplace | inplace deriving DecidableEq, Repr
def sourceGrid (b : Numerics) (d : Storage) (p : Layout) (mode : Mode)
    (input output : VeriTile.Triton.RegionName) (n block : Nat) : List Nat → State → Except Issue State
  | [],state => .ok state
  | pid::rest,state => do
      let next ← match mode with
        | .outplace => runOriginalProgram b d p input output n block pid state
        | .inplace => runInplaceProgram b d p input n block pid state
      sourceGrid b d p mode input output n block rest next
theorem source_grid_correspondence (b : Numerics) (d : Storage) (p : Layout) (mode : Mode)
    (input output : VeriTile.Triton.RegionName) (n block : Nat) (pids : List Nat) (state : State) :
    sourceGrid b d p mode input output n block pids state=executeGrid b d p n block pids state := by
  induction pids generalizing state with
  | nil => rfl
  | cons pid rest ih =>
    simp only [sourceGrid,executeGrid]
    cases mode <;> simp only [original_program_correspondence,inplace_program_correspondence]
    all_goals cases executeProgram b d p n block pid state <;> simp [ih]

/-- Every launched pid occurs exactly once. This rules out repeated in-place
doubling, while permitting any complete native program schedule. -/
def Schedule (n block : Nat) (pids : List Nat) : Prop := pids.Perm (List.range (grid n block))
def launchPlan (b : Numerics) (d : Storage) (p : Layout) (n block : Nat)
    (pids : List Nat) (initial : ByteMemory) : Except Issue (List ByteWrite) :=
  (gridPlan b d p n block pids initial []).map Prod.fst
def sourceLaunch (b : Numerics) (d : Storage) (p : Layout) (mode : Mode)
    (input output : VeriTile.Triton.RegionName) (n block : Nat)
    (pids : List Nat) (initial : ByteMemory) : Except Issue ByteMemory :=
  (sourceGrid b d p mode input output n block pids ⟨initial,[]⟩).map State.memory
theorem launch_correspondence (b : Numerics) (d : Storage) (p : Layout) (mode : Mode)
    (input output : VeriTile.Triton.RegionName) (n block : Nat)
    (pids : List Nat) (initial : ByteMemory) :
    sourceLaunch b d p mode input output n block pids initial=
    (launchPlan b d p n block pids initial).map (fun writes => byteResult writes initial) := by
  unfold sourceLaunch launchPlan
  rw [source_grid_correspondence,grid_correspondence]
  cases gridPlan b d p n block pids initial [] <;> simp [Except.map,writes_memory]
theorem launch_frame (b : Numerics) (d : Storage) (p : Layout) (n block : Nat)
    (pids : List Nat) (initial : ByteMemory) (writes : List ByteWrite)
    (h : launchPlan b d p n block pids initial=.ok writes)
    (address : Nat) (outside : ¬p.output.Contains address) :
    byteResult writes initial address=initial address := by
  unfold launchPlan at h
  cases hg : gridPlan b d p n block pids initial [] with
  | error issue => simp [hg,Except.map] at h
  | ok plan =>
    simp [hg,Except.map] at h
    subst writes
    rw [←writes_pointwise]
    apply writes_frame
    intro w hw he
    exact outside (he ▸ grid_write_bounds b d p n block pids initial [] plan.1 plan.2 hg w hw)
theorem empty_schedule (block : Nat) (positive : 0<block) (pids : List Nat)
    (schedule : Schedule 0 block pids) : pids=[] := by
  have hg : grid 0 block=0 := by unfold grid; simp only [Nat.zero_add]; exact Nat.div_eq_of_lt (by omega)
  simpa [Schedule,hg] using schedule

#specNonCircular launchPlan avoiding [sourceLaunch,sourceGrid,executeGrid,executeProgram,applyWrites,source,Expr.eval]
#axiomsClean grid_correspondence
#axiomsClean source_grid_correspondence
#axiomsClean launch_correspondence
#axiomsClean launch_frame
#axiomsClean empty_schedule
#guard_msgs (drop info) in
#auditModuleAxioms
end
end TritonMul2Finite
