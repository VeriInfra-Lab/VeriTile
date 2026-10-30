import bench.tritonbench_g.bmm_chunk_bwd.IO

namespace VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

theorem invocation_initial (h : HostCall) (c : Tuning) (q : HostGrid.Pid) (s : BlockState) :
    (invocation h c q).initial s = HostGrid.launch (HostContract.grid h c) q s := rfl

theorem schedule_eq (h : HostCall) (c : Tuning) (qs : List HostGrid.Pid) (s : BlockState) :
    runPreparedAlgorithmSchedule (qs.map (invocation h c)) s =
      HostGrid.run (Kernels.fp32Source h c).eraseDType.toAlgKernel (HostContract.grid h c) qs s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.map_cons,runPreparedAlgorithmSchedule,HostGrid.run,
      PreparedAlgorithmInvocation.run,invocation_initial]
    change (exec _ _).bind _ = (exec _ _).bind _
    congr 1
    funext next
    exact ih next

theorem input_bound (h : HostCall) (c : Tuning) (vals : (realIO h c).Ctx)
    (initial : BlockState) (bounds : RegionBounds) (hb : (realIO h c).InBounds bounds initial vals)
    (i : Fin 3) (j : TileIndex (inputShape h c i)) (active : inputMask h c i j) :
    inputWindow h c i j < bounds ((view h c).inputRegion i) := by
  simpa only [realIO,memory,get_tilePos] using hb.1 i (tilePos (inputShape h c i) j)
    (by simpa only [realIO,memory,get_tilePos] using active)

theorem output_bound (h : HostCall) (c : Tuning) (vals : (realIO h c).Ctx)
    (initial : BlockState) (bounds : RegionBounds) (hb : (realIO h c).InBounds bounds initial vals)
    (j : TileIndex (outputShape h c)) (active : outputMask h c j) :
    outputWindow h c j < bounds h.Out := by
  simpa only [realIO,memory,get_tilePos] using hb.2.1 (0 : Fin 1) (tilePos (outputShape h c) j)
    (by simpa only [realIO,memory,get_tilePos] using active)

theorem bounds_of_io (h : HostCall) (c : Tuning) (vals : (realIO h c).Ctx)
    (initial : BlockState) (bounds : RegionBounds) (hb : (realIO h c).InBounds bounds initial vals)
    (q : HostGrid.Pid) (hq : q ∈ jobs h c) (s : BlockState) :
    RealIO.Bounds (view h c) (HostGrid.launch (HostContract.grid h c) q s) bounds := by
  obtain ⟨qi,rfl⟩ := List.mem_iff_get.mp hq
  refine ⟨?_,?_,?_,?_⟩
  · intro i hi
    exact input_bound h c vals initial bounds hb 0 (qi,i) hi
  · intro i hi
    exact input_bound h c vals initial bounds hb 1 (qi,i) hi
  · intro hr i hi
    exact input_bound h c vals initial bounds hb 2 (qi,i) ⟨hr,hi⟩
  · intro i hi
    exact output_bound h c vals initial bounds hb (qi,i) hi

/-- Safety is proved along every actual predecessor in the complete serial
source launch. Bounds are finite declared IO capacities, never a postulated
source-execution or numerical assumption. -/
theorem schedule_safe (h : HostCall) (c : Tuning) (vals : (realIO h c).Ctx)
    (initial : BlockState) (bounds : RegionBounds) (hb : (realIO h c).InBounds bounds initial vals)
    (qs : List HostGrid.Pid) (subset : ∀ q ∈ qs,q ∈ jobs h c) (s : BlockState) :
    PreparedAlgorithmScheduleSafe bounds (qs.map (invocation h c)) s := by
  induction qs generalizing s with
  | nil => trivial
  | cons q qs ih =>
    refine ⟨?_,fun t _ => ih (fun q hq => subset q (by simp [hq])) t⟩
    change Kernel.TraceSafe bounds (Kernels.fp32Source h c).eraseDType.toAlgKernel
      ((invocation h c q).initial s)
    rw [source_real,invocation_initial]
    exact RealSafety.safe Bool.false _ c.BM c.BN c.BCS (HostContract.parameters h) bounds
      (RealContract.bounds_of_views Bool.false _ c.BM c.BN c.BCS (HostContract.parameters h)
        bounds (bounds_of_io h c vals initial bounds hb q (subset q (by simp)) s))

theorem input_read (h : HostCall) (c : Tuning) (vals : (realIO h c).Ctx)
    (s : BlockState) (bound : (realIO h c).InputBinding s vals)
    (i : Fin 3) (j : TileIndex (inputShape h c i)) (active : inputMask h c i j) :
    s.readMem ((view h c).inputRegion i) (inputWindow h c i j)=
      vals i (tilePos (inputShape h c i) j) := by
  simpa only [realIO,memory,get_tilePos,ChanTy.read] using
    bound i (tilePos (inputShape h c i) j)
      (by simpa only [realIO,memory,get_tilePos] using active)

/-- Every arbitrary old memory state supplies finite inputs; inactive padding
is unrestricted and overlapping input views retain their actual shared values. -/
def inputsOfState (h : HostCall) (c : Tuning) (s : BlockState) : (realIO h c).Ctx :=
  fun i j => s.readMem ((view h c).inputRegion i)
    (inputWindow h c i ((TileShape.allIndices _).get j))

theorem inputsOfState_binding (h : HostCall) (c : Tuning) (s : BlockState) :
    (realIO h c).InputBinding s (inputsOfState h c s) := by
  intro i j _
  rfl

theorem expected_eq (h : HostCall) (c : Tuning) (vals : (realIO h c).Ctx)
    (s : BlockState) (bound : (realIO h c).InputBinding s vals)
    (j : TileIndex (outputShape h c)) (active : outputMask h c j) :
    HostGrid.expected c.BM c.BN (HostContract.parameters h) s (pid h c j.1) j.2=
      mathematical h c vals 0 (tilePos (outputShape h c) j) := by
  have hrow : RealViews.row c.BM c.BN (HostContract.parameters h) (pid h c j.1).tile j.2.1 < h.chunk :=
    lt_of_lt_of_le active.1 (Nat.min_le_left _ _)
  unfold HostGrid.expected mathematical
  simp only [get_tilePos,HostContract.parameters]
  apply congrArg₂ (· + ·)
  · apply Finset.sum_congr rfl
    intro k _
    split_ifs with hk
    · exact congrArg₂ (· * ·)
        (input_read h c vals s bound 0 (j.1,j.2.1,k,()) ⟨hrow,hk⟩)
        (input_read h c vals s bound 1 (j.1,k,j.2.2.1,()) ⟨hk,active.2⟩)
    · rfl
  · split_ifs with hr
    · exact input_read h c vals s bound 2 j ⟨hr,active⟩
    · rfl

/-- The finite IO theorem reuses the original complete source-grid result.
Only geometric nonoverlap conditions are required for initial-input sums. -/
theorem implements (h : HostCall) (choice : Fin 9)
    (distinct : RealViews.OutputDistinct (tuning choice).BM (tuning choice).BN (HostContract.parameters h))
    (separate : HostGrid.InputsSeparate (HostContract.parameters h))
    (writes : HostGrid.SeparateWrites (tuning choice).BM (tuning choice).BN
      (HostContract.parameters h) (jobs h (tuning choice))) :
    (realIO h (tuning choice)).Implements (mathematical h (tuning choice)) := by
  apply PreparedAlgorithmProgramIO.Implements.intro
  · intro vals initial _ bounds hb
    exact schedule_safe h (tuning choice) vals initial bounds hb _ (fun _ hq => hq) initial
  · intro vals initial bound
    obtain ⟨final,executed,values,frame⟩ := HostGrid.run_correct Bool.false
      (tuning choice).BM (tuning choice).BN (tuning choice).BCS (HostContract.parameters h)
      (HostContract.tuning_positive choice).2.2 distinct separate
      (HostContract.grid h (tuning choice)) (jobs h (tuning choice))
      (HostGrid.programs_nodup _) writes initial
    refine ⟨final,?_,?_,?_⟩
    · change runPreparedAlgorithmSchedule ((jobs h (tuning choice)).map _) initial = _
      rw [schedule_eq,source_real]
      exact executed
    · intro o j active
      have hv := values (pid h (tuning choice) ((TileShape.allIndices _).get j).1)
        (List.get_mem _ _) ((TileShape.allIndices _).get j).2 active
      exact hv.trans (by
        simpa only [tilePos_get] using expected_eq h (tuning choice) vals initial bound
          ((TileShape.allIndices _).get j) active)
    · intro r off outside _
      apply frame r off
      intro q hq i hi
      obtain ⟨qi,rfl⟩ := List.mem_iff_get.mp hq
      simpa only [realIO,memory,outputWindow,get_tilePos] using
        outside (0 : Fin 1) (tilePos (outputShape h (tuning choice)) (qi,i))
          (by simpa only [realIO,memory,outputMask,get_tilePos] using hi)

/-- Exact bridge to the retained flattened-source certificate. This is the
same complete source schedule, not a second kernel or a new physical model. -/
theorem flatten_execution (h : HostCall) (choice : Fin 9) (alloc : FlatAlloc)
    (layout : HostContract.Layout h (tuning choice) alloc) (s : BlockState) :
    HostGrid.run (alloc.flattenKernel (Kernels.fp32Source h (tuning choice)).eraseDType.toAlgKernel)
      (HostContract.grid h (tuning choice)) (jobs h (tuning choice)) (alloc.flattenState s) =
      (runPreparedAlgorithmSchedule ((realIO h (tuning choice)).schedule
        (inputsOfState h (tuning choice) s)) s).map alloc.flattenState := by
  change HostGrid.run _ _ _ _ =
    (runPreparedAlgorithmSchedule ((jobs h (tuning choice)).map _) s).map _
  rw [schedule_eq,source_real]
  exact HostGrid.run_flatten Bool.false (tuning choice).BM (tuning choice).BN (tuning choice).BCS
    (HostContract.parameters h) (HostContract.grid h (tuning choice)) (jobs h (tuning choice))
    alloc layout.disjoint layout.closed layout.bounds s

#axiomsClean flatten_execution
#axiomsClean implements
#axiomsClean inputsOfState_binding
end
end VeriTile.Bench.Repairs.BmmChunkBwd.PublicIO
