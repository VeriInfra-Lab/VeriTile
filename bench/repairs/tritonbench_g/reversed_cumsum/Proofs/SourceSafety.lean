import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.RealExecution
import VeriTile.Triton.Memory.TraceProofs

/-! Access safety of the repaired masked-pointer source, including both tails
and the actual signed reverse loop. This does not reuse the old block-pointer
source's safety certificate. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.SourceSafety
open VeriTile.Triton RealProgram IndexProgram RealValues RealBridge
open VeriTile.Bench.TritonBenchG.ReversedCumsumScalar
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Bounds (h : Call) (row f : Nat) (bounds : RegionBounds) : Prop where
  input : ∀ (t : Nat) (j : Fin BS), t<h.T → f*BS+j.val<h.S →
    Geometry.address h row t (f*BS+j.val)<bounds h.input
  output : ∀ (t : Nat) (j : Fin BS), t<h.T → f*BS+j.val<h.S →
    Geometry.address h row t (f*BS+j.val)<bounds h.output

private theorem address_safe (h : Call) (c : Config) (row q : Nat)
    (f : Geometry.FeatureBlock h) (r : RegionName) (bounds : RegionBounds)
    (start : q*c.BT<h.T)
    (hb : ∀ (t : Nat) (j : Fin BS), t<h.T → f.val*BS+j.val<h.S →
      Geometry.address h row t (f.val*BS+j.val)<bounds r)
    (idx : TileIndex [c.BT,BS]) (ha : (mask h c q f.val).data idx=Bool.true) :
    ((pointer h c row q f.val r).data idx).2<bounds ((pointer h c row q f.val r).data idx).1 := by
  obtain ⟨i,j,⟨⟩⟩ := idx
  have active : Geometry.active h c q f.val i j := by simpa [mask] using ha
  have inRange := (Geometry.active_iff h c q f i j start).mp active
  simpa only [pointer,Geometry.time,Geometry.feature,if_pos active.1,if_pos active.2] using
    hb (q*c.BT+i.val) j inRange.1 inRange.2

private theorem load_safe (h : Call) (c : Config) (row q : Nat)
    (f : Geometry.FeatureBlock h) (bounds : RegionBounds) (u : BlockState)
    (start : q*c.BT<h.T) (hb : Bounds h row f.val bounds)
    (hp : u.regs .ptr [c.BT,BS] "p_s"=some (pointer h c row q f.val h.input))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c q f.val)) :
    (loadOp c).SafeAt bounds u := by
  rw [loadOp,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],by simp [Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he i active
  rw [evalOp_ref,hp] at he
  cases he
  apply address_safe h c row q f h.input bounds start hb.input i
  simpa [MaskOpt.Active,evalOp_ref,hm] using active

private theorem store_safe (h : Call) (c : Config) (row q : Nat)
    (f : Geometry.FeatureBlock h) (bounds : RegionBounds) (u : BlockState)
    (start : q*c.BT<h.T) (hb : Bounds h row f.val bounds)
    (hp : u.regs .ptr [c.BT,BS] "p_z"=some (pointer h c row q f.val h.output))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c q f.val)) :
    Stmt.TraceSafe bounds (.store .real [c.BT,BS] (.ptr (.ref .ptr [c.BT,BS] "p_z"))
      (.ref .real [c.BT,BS] "b_c") (.mask (.ref .bool [c.BT,BS] "active"))) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt],by simp [Op.SafeAt],by simp [MaskOpt.SafeAt,Op.SafeAt],?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he i active
  rw [evalOp_ref,hp] at he
  cases he
  apply address_safe h c row q f h.output bounds start hb.output i
  simpa [MaskOpt.Active,evalOp_ref,hm] using active

theorem chunk_safe (h : Call) (c : Config) (row q : Nat)
    (f : Geometry.FeatureBlock h) (bounds : RegionBounds) (hb : Bounds h row f.val bounds)
    (u : BlockState) (start : q*c.BT<h.T)
    (his : u.regs .int [] "i_s"=some (Tile.scalar (f.val:Int)))
    (hih : u.regs .int [] "i_bh"=some (Tile.scalar (row:Int)))
    (hit : u.regs .int [] "i_t"=some (Tile.scalar (q:Int))) :
    Stmt.TraceSafeList bounds (RealStep.chunkCode h c) u := by
  simp only [RealStep.chunkCode,loopBody,List.tail_cons]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (startT_eval c q u hit))]
  refine ⟨by simp [Stmt.TraceSafe,startTOp,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (startS_eval f.val _ (by simp [his])))]
  refine ⟨by simp [Stmt.TraceSafe,startSOp,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (.castNatToInt (.arange c.BT)) _=some (signedLanes c.BT) from by simp only [evalOp, Option.bind_some]; rfl))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (show evalOp (.castNatToInt (.arange BS)) _=some (signedLanes BS) from by simp only [evalOp, Option.bind_some]; rfl))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (activeT_eval h c q _ start (by simp) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,activeTOp,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (activeS_eval h f.val _ (by simp) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,activeSOp,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (time_eval h c q _ (by simp) (by simp) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,timeOp,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (feature_eval h f.val _ (by simp) (by simp) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,featureOp,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (mask_eval h c q f.val _ (by simp) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,maskOp,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ (RealStep.indexed h c q f.val u)
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval h c row q f.val h.input _
    (by simp [RealStep.indexed,hih]) (by simp [RealStep.indexed]) (by simp [RealStep.indexed])))]
  refine ⟨by simp [Stmt.TraceSafe,pointerOp,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval h c row q f.val h.output _
    (by simp [RealStep.indexed,hih]) (by simp [RealStep.indexed]) (by simp [RealStep.indexed])))]
  refine ⟨by simp [Stmt.TraceSafe,pointerOp,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  apply TraceProofs.assign_cons (load_safe h c row q f bounds _ start hb (by simp) (by simp [RealStep.indexed]))
  intro bs
  apply TraceProofs.assign_cons
  · simp [outputOp,Op.SafeAt,Op.SafeAt.eq_def]
  intro bc
  apply Stmt.TraceSafeList.cons_intro (store_safe h c row q f bounds _ start hb (by simp) (by simp [RealStep.indexed]))
  intro v _
  apply Stmt.TraceSafeList.cons_intro
  · rw [Stmt.TraceSafe]
    refine ⟨by simp [nonnegativeOp,Op.SafeAt],?_⟩
    split
    · split
      · apply TraceProofs.all_safe
        simp [Stmt.TraceSafe,carryOp,Op.SafeAt,Op.SafeAt.eq_def]
      · trivial
    · trivial
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem iteration_safe (initial : BlockState) (h : Call) (c : Config) (row : Nat)
    (f : Geometry.FeatureBlock h) (a : Nat → ℝ) (positive : 0<c.BT)
    (bounds : RegionBounds) (hb : Bounds h row f.val bounds) (j : Int) (u : BlockState)
    (hj : j<1) (inv : Holds initial h c row f.val a (SignedIndex.remaining j) u) :
    Stmt.TraceSafeList bounds (loopBody h c) (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) := by
  have count : (-j).toNat<Geometry.chunks h c := by
    have hn := inv.count_le
    rw [SignedIndex.current_chunk j hj] at hn
    change (-j).toNat+1≤Geometry.chunks h c at hn
    omega
  change Stmt.TraceSafeList bounds (.assign .int [] "i_t" indexOp :: RealStep.chunkCode h c) _
  unfold indexOp
  rw [TraceProofs.cons_of_run_iff (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealExecution.mirror_assign j u)]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  apply chunk_safe h c row (-j).toNat f bounds hb _ (Geometry.chunk_start h c positive _ count)
  · simpa [] using inv.feature
  · simpa [] using inv.head
  · rw [SignedIndex.int_index_of_nat j hj]
    simp

theorem loop_safe (initial : BlockState) (h : Call) (c : Config) (row : Nat)
    (f : Geometry.FeatureBlock h) (a : Nat → ℝ) (distinct : h.input≠h.output) (positive : 0<c.BT)
    (bounds : RegionBounds) (hb : Bounds h row f.val bounds) (u : BlockState)
    (inv : Holds initial h c row f.val a (Geometry.chunks h c) u) :
    Stmt.TraceSafe bounds (statement h c) u := by
  rw [statement,Stmt.TraceSafe]
  refine ⟨by simp [reverseStart,stopOp,Op.SafeAt],by simp [Op.SafeAt],by simp [MaskOpt.SafeAt,Op.SafeAt],?_⟩
  rw [RealExecution.start_eval]
  simp only [evalOp]
  change Stmt.forRangeIntTraceSafe bounds "__rev_i_t" (1-(Geometry.chunks h c:Int)) 1 1 (loopBody h c) u
  apply VeriTile.Bench.TritonBenchG.ReversedCumsum.SignedRange.safe bounds "__rev_i_t" 1 1 (loopBody h c)
    (fun j v => Holds initial h c row f.val a (SignedIndex.remaining j) v)
  · intro j v hj hv
    exact iteration_safe initial h c row f a positive bounds hb j v hj hv
  · intro j v w hj hv he
    obtain ⟨w',he',hw⟩ := RealExecution.iteration initial h c row f a distinct positive j v hj hv
    rw [he] at he'
    cases he'
    exact hw
  · simpa only [SignedIndex.at_start] using inv

theorem source_safe (h : Call) (c : Config) (row : Nat) (f : Geometry.FeatureBlock h)
    (initial : BlockState) (distinct : h.input≠h.output) (positive : 0<c.BT)
    (pid0 : initial.pids 0=f.val) (pid1 : initial.pids 1=row)
    (bounds : RegionBounds) (hb : Bounds h row f.val bounds) :
    Kernel.TraceSafe bounds (Kernels.source h c).eraseDType.toAlgKernel initial := by
  have inv : Holds initial h c row f.val (initial.readMem h.input) (Geometry.chunks h c)
      (RealExecution.initialized c initial) := by
    simpa [pid0,pid1] using RealExecution.init_invariant h c initial positive
  rw [Kernel.TraceSafe,source_body,body]
  apply TraceProofs.append_safe
  · apply TraceProofs.all_safe
    intro st hs u
    simp only [initCode,List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl
    · rw [Stmt.TraceSafe]
      refine ⟨by simp [Op.SafeAt],?_⟩
      simp only [evalOp,Tile.scalar_data,Option.bind_some,if_true]
      apply TraceProofs.all_safe
      simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
    · simp [Stmt.TraceSafe,Op.SafeAt]
    · simp [Stmt.TraceSafe,triOp,Op.SafeAt,Op.SafeAt.eq_def]
    · simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
  · intro v he
    rw [RealExecution.init_executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro
    · exact loop_safe initial h c row f (initial.readMem h.input) distinct positive bounds hb _ inv
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ReversedCumsum.SourceSafety
