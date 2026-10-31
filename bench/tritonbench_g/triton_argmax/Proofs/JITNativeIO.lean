import bench.tritonbench_g.triton_argmax.IO
import bench.tritonbench_g.triton_argmax.Proofs.JITNative

/-! Complete typed native schedule, reusing the exact source/codec results
which also justify BoundSource's physical first/second commits. -/
namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

variable (casts : Request.Casts) (kind : Request.Kind)
    (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (g : Native.GlobalPrograms c)

/-- Exact typed store state, including the requested scratch codec. -/
def snapshot (input : Nat → Word) (s : State Word) (pid : Nat) : State Word :=
  (s.write BoundSource.V pid (JITBinding.cell binding
    (Native.firstValue (Request.scratch casts kind) g input pid))).write
      (Region.cast BoundSource.I) pid (.mk .int (Int.ofNat (Native.firstIndex g input pid)))

theorem first_execution (s : State Word) (index64 : Bool) (positive : 0 < c.block) :
    ∃ t, exec (JITBinding.ops binding rest) (JITBinding.first binding c index64) (JITStage1.arguments s c.size 0) = some t ∧
      t.mem = (snapshot casts kind binding c g
        (fun i => (s.mem BoundSource.X i).read .real) s (s.pids 0)).mem := by
  cases binding with
  | floating k round codec =>
    simpa only [JITBinding.ops, JITBinding.first, snapshot,
      JITBinding.cell, Request.scratch, Nat.add_zero] using
      JITNative.first_float k round (casts.floating k) codec rest s
        BoundSource.X BoundSource.V BoundSource.I c g 0 index64 positive
  | integer k zero =>
    obtain ⟨t,ran,mem⟩ := JITInteger.source_run k (casts.integer k) rest s
      BoundSource.X BoundSource.V BoundSource.I c.size c.block 0 index64 positive
    refine ⟨t,ran,mem.trans ?_⟩
    rw [JITInteger.written,JITNative.first_index s BoundSource.X c g 0 positive]
    cases k <;> rfl

theorem first_index_congr (input other : Nat → Word)
    (same : ∀ i, i < c.size → input i = other i) (pid : Nat) :
    Native.firstIndex g input pid = Native.firstIndex g other pid := by
  have loads : c.loaded input 0xff800000 pid = c.loaded other 0xff800000 pid := by
    funext lane
    unfold GlobalConfig.loaded
    split
    · exact same _ (by assumption)
    · rfl
  simp only [Native.firstIndex, loads]

theorem first_valid (input : Nat → Word) (pid : Nat) (positive : 0 < c.block)
    (active : pid * c.block < c.size) : Native.firstIndex g input pid < c.size := by
  have valid := selected_valid c.size (pid*c.block) c.block input (c.firstTree pid)
    (g.firstNumeric pid ▸ (g.first pid).numericComplete) active positive
  simpa only [Native.firstIndex, Native.Sites.pick_eq, g.firstNumeric,
    GlobalConfig.loaded, selected, padded, Nat.add_comm] using valid

theorem first_value_congr (input other : Nat → Word)
    (same : ∀ i, i < c.size → input i = other i) (pid : Nat)
    (positive : 0 < c.block) (active : pid * c.block < c.size) :
    Native.firstValue (Request.scratch casts kind) g input pid =
      Native.firstValue (Request.scratch casts kind) g other pid := by
  unfold Native.firstValue
  rw [first_index_congr c g input other same pid, same _ (first_valid c g other pid positive active)]

theorem snapshot_memory (input : Nat → Word) (jobs : List Nat)
    (s t : State Word) (same : s.mem = t.mem) :
    (jobs.foldl (snapshot casts kind binding c g input) s).mem =
      (jobs.foldl (snapshot casts kind binding c g input) t).mem := by
  induction jobs generalizing s t with
  | nil => exact same
  | cons pid jobs ih =>
    simp only [List.foldl_cons]
    apply ih
    simp only [snapshot, State.write, same]

theorem snapshot_frame (input : Nat → Word) (jobs : List Nat) (s : State Word)
    (r : RegionName) (o : Nat)
    (outside : ∀ pid, pid ∈ jobs →
      (r ≠ BoundSource.V ∨ o ≠ pid) ∧ (r ≠ Region.cast BoundSource.I ∨ o ≠ pid)) :
    (jobs.foldl (snapshot casts kind binding c g input) s).mem r o = s.mem r o := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    rw [List.foldl_cons, ih _ (fun j hj => outside j (by simp [hj]))]
    obtain ⟨hv, hi⟩ := outside pid (by simp)
    unfold snapshot
    rw [State.write_other _ _ _ _ _ _ hi, State.write_other _ _ _ _ _ _ hv]

theorem snapshot_values (input : Nat → Word) (jobs : List Nat) (s : State Word) (pid : Nat) :
    (jobs.foldl (snapshot casts kind binding c g input) s).mem BoundSource.V pid =
      if pid ∈ jobs then JITBinding.cell binding (Native.firstValue (Request.scratch casts kind) g input pid)
      else s.mem BoundSource.V pid := by
  induction jobs generalizing s with
  | nil => simp
  | cons j jobs ih =>
    rw [List.foldl_cons, ih]
    by_cases member : pid ∈ jobs
    · simp [member]
    · by_cases same : pid = j
      · subst pid; simp [member, snapshot, State.write, BoundSource.V, BoundSource.I, Region.cast]
      · simp [member, same, snapshot, State.write, BoundSource.V, BoundSource.I, Region.cast]

theorem snapshot_indices (input : Nat → Word) (jobs : List Nat) (s : State Word) (pid : Nat) :
    (jobs.foldl (snapshot casts kind binding c g input) s).mem (Region.cast BoundSource.I) pid =
      if pid ∈ jobs then .mk .int (Int.ofNat (Native.firstIndex g input pid))
      else s.mem (Region.cast BoundSource.I) pid := by
  induction jobs generalizing s with
  | nil => simp
  | cons j jobs ih =>
    rw [List.foldl_cons, ih]
    by_cases member : pid ∈ jobs
    · simp [member]
    · by_cases same : pid = j
      · subst pid; simp [member, snapshot, State.write, BoundSource.V, BoundSource.I, Region.cast]
      · simp [member, same, snapshot, State.write, BoundSource.V, BoundSource.I, Region.cast]

theorem grid_execution (input : Nat → Word) (jobs : List Nat) (s : State Word)
    (index64 : Bool) (positive : 0 < c.block)
    (active : ∀ pid, pid ∈ jobs → pid * c.block < c.size)
    (reads : ∀ i, i < c.size → (s.mem BoundSource.X i).read .real = input i) :
    ∃ t, runSchedule (jobs.map (JITIO.firstLaunch (JITBinding.ops binding rest) (JITBinding.first binding c index64) c.size 0)) s = some t ∧
      t.mem = (jobs.foldl (snapshot casts kind binding c g input) s).mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s, rfl, rfl⟩
  | cons pid jobs ih =>
    let initial := (JITIO.atProgram (JITBinding.ops binding rest) (JITBinding.first binding c index64) pid).initial s
    obtain ⟨a, ran, mem⟩ := first_execution casts kind binding rest c g initial index64 positive
    have pin : ∀ i, i < c.size → (initial.mem BoundSource.X i).read .real = input i := by
      simpa [initial] using reads
    have localMem : a.mem = (snapshot casts kind binding c g input s pid).mem := by
      rw [mem]
      simp only [snapshot, initial, JITIO.atProgram_initial, ite_true]
      rw [first_index_congr c g _ input reads pid,
        first_value_congr casts kind c g _ input reads pid positive (active pid (by simp))]
      rfl
    have next : ∀ i, i < c.size → (a.mem BoundSource.X i).read .real = input i := by
      intro i hi
      rw [localMem]
      simpa [snapshot, State.write, BoundSource.X, BoundSource.V, BoundSource.I, Region.cast] using reads i hi
    obtain ⟨t, runs, mt⟩ := ih a (fun j hj => active j (by simp [hj])) next
    refine ⟨t, ?_, ?_⟩
    · simp only [List.map_cons, runSchedule, Invocation.run]
      change (exec (JITBinding.ops binding rest) (JITBinding.first binding c index64) (JITStage1.arguments initial c.size 0)).bind _ = _
      rw [ran]
      exact runs
    · rw [mt, List.foldl_cons]
      exact snapshot_memory casts kind binding c g input jobs a _ localMem

theorem second_execution (s : State Word) (input : Nat → Word)
    (positive : 0 < c.midBlock)
    (values : ∀ i, i < c.blocks → s.mem BoundSource.V i =
      JITBinding.cell binding (Native.firstValue (Request.scratch casts kind) g input i)) :
    ∃ t, exec (JITBinding.ops binding rest) (JITBinding.second binding c) (s.setReg "mid_size" .int [] (fun _ => (c.blocks : Int))) = some t ∧
      t.mem = (s.write (Region.cast BoundSource.O) 0
        (.mk .int ((s.mem (Region.cast BoundSource.I)
          (Native.secondSelected (Request.scratch casts kind) kind.isFloating g
            (Native.firstValue (Request.scratch casts kind) g input)).val).read .int))).mem := by
  cases binding with
  | floating k round codec =>
    apply JITNative.second_float k round (casts.floating k) codec rest s
      BoundSource.V BoundSource.I BoundSource.O c g positive
      (Native.firstValue (floatingScratch k (casts.floating k)) g input)
    · intro i _
      exact Or.inl ⟨input (Native.firstIndex g input i), rfl⟩
    · intro i hi
      rw [values i hi]
      cases k <;> rfl
  | integer k zero =>
    apply JITNative.second_integer k (casts.integer k) (JITBinding.ops (.integer k zero) rest)
      s BoundSource.V BoundSource.I BoundSource.O c g positive
      (Native.firstValue (integerScratch k (casts.integer k)) g input)
    intro i hi
    rw [values i hi]
    simpa only [JITBinding.cell, BitVec.cast_cast, BitVec.cast_eq] using
      JITInteger.read_cell k
        ((Native.firstValue (integerScratch k (casts.integer k)) g input i).cast (Formats.format_width k).symm)

theorem second_reference (input : Nat → Word) (positive : 0 < c.block)
    (midPositive : 0 < c.midBlock) (active : ∀ pid, pid < c.blocks → pid * c.block < c.size) :
    Native.secondSelected (Request.scratch casts kind) kind.isFloating g
        (Native.firstValue (Request.scratch casts kind) g input) =
      TritonArgmaxNanFirst.secondReference (Request.scratch casts kind) c positive midPositive input := by
  rw [Native.second_selected]
  have same : stage2Selected (Request.scratch casts kind).scalars c
      (Native.firstValue (Request.scratch casts kind) g input) =
      stage2Selected (Request.scratch casts kind).scalars c
        (stage1Value (Request.scratch casts kind).scalars c input) := by
    unfold stage2Selected
    congr 1
    funext lane
    unfold stage2Loaded
    split
    · exact Native.first_value (Request.scratch casts kind) g input lane.val positive (active _ (by assumption))
    · rfl
  rw [same]
  exact TritonArgmaxNanFirst.second_reference (Request.scratch casts kind) c positive midPositive input
    (fun pid _ => g.firstNumeric pid ▸ (g.first pid).numericComplete)
    (g.secondNumeric ▸ g.second.numericComplete)

include g in
/-- Full typed execution preserves exactly the old first-grid scratch stores
and final scalar store. Initial zero tail indices remain the original premise. -/
theorem global_execution (input : Nat → Word) (s : State Word) (index64 : Bool)
    (positive : 0 < c.block) (midPositive : 0 < c.midBlock)
    (active : ∀ pid, pid < c.blocks → pid * c.block < c.size)
    (reads : ∀ i, i < c.size → (s.mem BoundSource.X i).read .real = input i)
    (initialized : ∀ i, i < c.midBlock → (s.mem (Region.cast BoundSource.I) i).read .int = 0) :
    ∃ final, runSchedule (JITIO.nativeSchedule casts kind binding rest c index64) s = some final ∧
      final.mem (Region.cast BoundSource.O) 0 = .mk .int
        (Int.ofNat (globalReference (Request.scratch casts kind) c positive midPositive input)) ∧
      (∀ r o, (r ≠ Region.cast BoundSource.O ∨ o ≠ 0) →
        (∀ pid, pid < c.blocks → (r ≠ BoundSource.V ∨ o ≠ pid) ∧
          (r ≠ Region.cast BoundSource.I ∨ o ≠ pid)) → final.mem r o = s.mem r o) := by
  obtain ⟨middle, first, memory⟩ := grid_execution casts kind binding rest c g input
    (List.range c.blocks) s index64 positive (fun pid hp => active pid (List.mem_range.mp hp)) reads
  have values : ∀ i, i < c.blocks → middle.mem BoundSource.V i =
      JITBinding.cell binding (Native.firstValue (Request.scratch casts kind) g input i) := by
    intro i hi
    rw [memory, snapshot_values casts kind binding c g input (List.range c.blocks) s i,
      if_pos (List.mem_range.mpr hi)]
  obtain ⟨final, second, output⟩ := second_execution casts kind binding rest c g middle input midPositive values
  refine ⟨final, ?_, ?_, ?_⟩
  · rw [JITIO.nativeSchedule, runSchedule_append, first, Option.bind_some, runSchedule_singleton]
    simpa only [Invocation.run, JITIO.secondLaunch_initial, JITIO.secondLaunch] using second
  · rw [output, State.write_same]
    have selected := second_reference casts kind c g input positive midPositive active
    rw [memory, snapshot_indices casts kind binding c g input (List.range c.blocks)]
    simp only [List.mem_range]
    rw [selected]
    unfold globalReference
    split
    · rename_i inside
      simp only [Cell.read_same]
      rw [Native.first_index (Request.scratch casts kind) g,
        first_index_reference (Request.scratch casts kind) c positive input _
          (g.firstNumeric _ ▸ (g.first _).numericComplete)]
      simp only [if_pos inside]
    · rename_i outside
      rw [initialized _ (TritonArgmaxNanFirst.secondReference (Request.scratch casts kind)
        c positive midPositive input).isLt]
      simp [outside]
  · intro r o out scratch
    rw [output, State.write_other _ _ _ _ _ _ out, memory]
    exact snapshot_frame casts kind binding c g input (List.range c.blocks) s r o
      (fun pid hp => scratch pid (List.mem_range.mp hp))

#axiomsClean first_execution
#axiomsClean grid_execution
#axiomsClean second_execution
#axiomsClean global_execution
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNativeIO
