import bench.tritonbench_g.triton_argmax.Proofs.JITRealIO

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

variable (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig)
    (coverage : Math.GlobalCoverage c) (index64 : Bool)
include coverage

theorem first_safe (s : State Math.Value) (xs : Nat → ℝ) (pid : Nat) (active : pid < c.blocks)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ i, i < c.size → i < bounds X)
    (valueBounds : ∀ i, i < c.blocks → i < bounds V)
    (indexBounds : ∀ i, i < c.blocks → i < bounds (Region.cast I))
    (reads : ∀ i, i < c.size → (s.mem X i).read .real = (xs i : Math.Value)) :
    (JITIO.firstLaunch (TypedReal.backend rest)
      (JITSource.stage1 X V I (JITStage1.floatingType kind) c.block index64) c.size 0 pid).Safe bounds s := by
  let initial := TypedReal.atProgram s pid
  have selected := (TypedReal.first_reference (adapter := .directCast kind.direct) initial X c
    coverage 0 xs (by simpa [initial, TypedReal.atProgram] using active) reads).1
  have valid := Math.block_valid c.size (pid*c.block) c.block xs (c.firstTree pid)
    (coverage.firstComplete pid active) (coverage.eachActive pid active) coverage.blockPositive
  have hit : TypedReal.firstIndex initial X c.size c.block 0 coverage.blockPositive < bounds X := by
    rw [selected]
    exact inputBounds _ (by simpa [initial, TypedReal.atProgram] using valid)
  unfold Invocation.Safe
  rw [JITIO.firstLaunch_initial, JITIO.atProgram_initial]
  exact JITRealSafety.real_first kind rest initial X V I c.size c.block 0 index64
      coverage.blockPositive bounds inputBounds hit
      (by simpa [initial, TypedReal.atProgram] using valueBounds pid active)
      (by simpa [initial, TypedReal.atProgram] using indexBounds pid active)

theorem grid_safe (inputValue : X ≠ V) (inputIndex : X ≠ Region.cast I)
    (s : State Math.Value) (xs : Nat → ℝ) (jobs : List Nat)
    (active : ∀ pid, pid ∈ jobs → pid < c.blocks) (bounds : RegionName → Nat)
    (inputBounds : ∀ i, i < c.size → i < bounds X)
    (valueBounds : ∀ i, i < c.blocks → i < bounds V)
    (indexBounds : ∀ i, i < c.blocks → i < bounds (Region.cast I))
    (reads : ∀ i, i < c.size → (s.mem X i).read .real = (xs i : Math.Value)) :
    ScheduleSafe bounds (jobs.map (JITIO.firstLaunch (TypedReal.backend rest)
      (JITSource.stage1 X V I (JITStage1.floatingType kind) c.block index64) c.size 0)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid jobs ih =>
    refine ⟨first_safe kind rest X V I c coverage index64 s xs pid (active pid (by simp))
      bounds inputBounds valueBounds indexBounds reads, ?_⟩
    intro actual actualRun
    obtain ⟨t, ran, mem⟩ := JITRealGrid.first_snapshot kind rest s X V I c coverage xs pid
      (active pid (by simp)) index64 reads
    have same : actual = t := by
      have equal : some actual = some t := actualRun.symm.trans (by
        simp only [Invocation.run, JITIO.firstLaunch_initial, JITIO.atProgram_initial]
        change exec (TypedReal.backend rest)
          (JITSource.stage1 X V I (JITStage1.floatingType kind) c.block index64) (JITStage1.arguments (JITRealGrid.atProgram s pid) c.size 0) = some t
        exact ran)
      exact Option.some.inj equal
    subst actual
    apply ih t (fun j hj => active j (by simp [hj]))
    intro i hi
    rw [mem]
    simpa [JITRealGrid.snapshotStore, State.write, inputValue, inputIndex] using reads i hi

theorem second_safe (s : State Math.Value) (xs : Nat → ℝ)
    (bounds : RegionName → Nat)
    (valueBounds : ∀ i, i < c.blocks → i < bounds V)
    (indexBounds : ∀ i, i < c.blocks → i < bounds (Region.cast I))
    (outputBound : 0 < bounds (Region.cast O))
    (values : ∀ i, i < c.blocks → toFloat (TypedScratch.channel kind)
      ((s.mem V i).read (TypedScratch.channel kind).toTileDType) = (xs (Math.firstIndex c xs i) : Math.Value)) :
    (JITIO.secondLaunch (TypedReal.backend rest)
      (JITSource.stage2 V I O (JITStage1.floatingType kind) c.midBlock) c.blocks).Safe bounds s := by
  let positive := coverage.blocksPositive.trans_le coverage.midFits
  have loads : (fun lane : Fin c.midBlock => if lane.val < c.blocks then
      toFloat (TypedScratch.channel kind) ((s.mem V lane.val).read (TypedScratch.channel kind).toTileDType)
      else ⊥) = Math.midValues c xs := by
    funext lane
    by_cases active : lane.val < c.blocks
    · simp [Math.midValues, active, values _ active]
    · simp [Math.midValues, active]
  have selected : TypedReal.secondIndex kind s V c.blocks c.midBlock positive =
      (Math.globalSelected c xs).val := by
    unfold TypedReal.secondIndex
    rw [loads, TypedReal.best_tree positive _ c.secondTree coverage.secondComplete]
    rfl
  unfold Invocation.Safe
  rw [JITIO.secondLaunch_initial]
  exact JITRealSafety.real_second kind rest s V I O c.blocks c.midBlock positive bounds
      valueBounds (by rw [selected]; exact indexBounds _ (Math.mid_valid c coverage xs)) outputBound

theorem implements (inputValue : X ≠ V) (inputIndex : X ≠ Region.cast I)
    (scratchApart : V ≠ Region.cast I) :
    (JITIO.realGlobal kind rest X V I O c index64).Implements
      (mathematical kind rest X V I O c coverage index64) := by
  intro vals _ bounds bounded s binding
  let xs := input c vals
  have reads : ∀ i, i < c.size → (s.mem X i).read .real = (xs i : Math.Value) := by
    intro i hi
    have h := binding 0 (⟨i,hi⟩,PUnit.unit) trivial
    simpa [JITIO.realGlobal, mapValue, xs, input, hi] using h
  have inputBounds : ∀ i, i < c.size → i < bounds X := by
    intro i hi
    exact bounded.2.1 0 (⟨i,hi⟩,PUnit.unit) trivial
  have valueBounds : ∀ i, i < c.blocks → i < bounds V := by
    intro i hi
    exact bounded.2.2.2 0 (⟨i,hi⟩,PUnit.unit) trivial
  have indexBounds : ∀ i, i < c.blocks → i < bounds (Region.cast I) := by
    intro i hi
    exact bounded.2.2.2 1 (⟨i,hi⟩,PUnit.unit) trivial
  have outputBound : 0 < bounds (Region.cast O) :=
    bounded.2.2.1 0 (⟨0,by decide⟩,PUnit.unit) trivial
  constructor
  · change ScheduleSafe bounds (JITIO.realSchedule kind rest X V I O c index64) s
    rw [JITIO.realSchedule, scheduleSafe_append]
    refine ⟨grid_safe kind rest X V I c coverage index64 inputValue inputIndex s xs
      (List.range c.blocks) (fun pid hp => List.mem_range.mp hp) bounds
      inputBounds valueBounds indexBounds reads, ?_⟩
    intro middle runMiddle
    obtain ⟨t, oldRun, mem⟩ := JITRealGrid.grid_run kind rest s X V I inputValue inputIndex c
      coverage xs (List.range c.blocks) (fun pid hp => List.mem_range.mp hp) index64 reads
    have equal : middle = t := Option.some.inj (runMiddle.symm.trans
      (grid_execution kind rest X V I c index64 _ s t oldRun))
    subst middle
    rw [scheduleSafe_singleton]
    apply second_safe kind rest V I O c coverage t xs bounds valueBounds indexBounds outputBound
    intro i hi
    rw [mem, JITRealGrid.snapshot_values kind V I scratchApart c xs (List.range c.blocks) s i,
      if_pos (List.mem_range.mpr hi)]
    cases kind <;> rfl
  · obtain ⟨middle, final, first, second, output, frame⟩ := JITRealGrid.global_source_reference kind rest s
      X V I O inputValue inputIndex scratchApart c coverage xs index64 reads
    refine ⟨final, ?_, ?_, ?_⟩
    · change runSchedule (JITIO.realSchedule kind rest X V I O c index64) s = some final
      rw [JITIO.realSchedule, runSchedule_append,
        grid_execution kind rest X V I c index64 _ s middle first, Option.bind_some,
        runSchedule_singleton]
      simpa only [Invocation.run, JITIO.secondLaunch_initial, JITIO.secondLaunch] using second
    · intro o j _
      have ho : o = 0 := Fin.eq_zero o
      subst o
      change mapCell (fun _ => (0 : ℝ)) (final.mem (Region.cast O) 0) = _
      rw [output]
      have coordinates : (fun i : Fin c.size => xs i.val) =
          (fun i : Fin c.size => vals 0 (i,PUnit.unit)) := by
        funext i
        simp [xs, input, i.isLt]
      change Cell.mk .int (Int.ofNat (Math.argmax c.size coverage.sizePositive
        (fun i : Fin c.size => xs i.val)).val) =
        Cell.mk .int (Int.ofNat (Math.argmax c.size coverage.sizePositive
          (fun i : Fin c.size => vals 0 (i,PUnit.unit))).val)
      rw [coordinates]
    · intro r o outside
      apply frame r o
      · simpa [JITIO.realGlobal] using outside.1 0 (⟨0,by decide⟩,PUnit.unit) trivial
      · intro pid hp
        exact ⟨outside.2 0 (⟨pid,hp⟩,PUnit.unit) trivial,
          outside.2 1 (⟨pid,hp⟩,PUnit.unit) trivial⟩

#axiomsClean implements
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealIO
