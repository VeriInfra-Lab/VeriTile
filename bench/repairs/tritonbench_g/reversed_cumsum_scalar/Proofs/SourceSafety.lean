import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealExecution
import VeriTile.Triton.Memory.TraceProofs
import VeriTile.Triton.Memory.FlattenR

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.SourceSafety
open VeriTile.Triton RealProgram RealValues RealExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem cons_of_run {bounds : RegionBounds} {st : Stmt} {rest : List Stmt}
    {s t : BlockState} (ran : stepStmt st s=some t) :
    Stmt.TraceSafeList bounds (st::rest) s ↔
      Stmt.TraceSafe bounds st s ∧ Stmt.TraceSafeList bounds rest t := by
  rw [Stmt.TraceSafeList,ran]

private theorem assign_cons {d sh name} {e : Op d sh} {s : BlockState} {bounds rest}
    (safe : e.SafeAt bounds s)
    (tail : ∀ v,Stmt.TraceSafeList bounds rest (s.setReg name d sh v)) :
    Stmt.TraceSafeList bounds (.assign d sh name e::rest) s := by
  apply Stmt.TraceSafeList.cons_intro (by simpa only [Stmt.TraceSafe] using safe)
  intro t ran
  obtain ⟨v,_,rfl⟩ := TraceProofs.assign_inv ran
  exact tail v

private theorem load_safe (h : Call) (c : Config) (row q : Nat)
    (bounds : RegionBounds) (u : BlockState) (start : q*c.BT<h.T)
    (hp : u.regs .ptr [c.BT] "p_s"=some (pointer h c row q h.input h.inputStrides))
    (hm : u.regs .bool [c.BT] "active"=some (mask h c q))
    (hb : ∀ t,t<h.T → h.inputStrides.address h.H row t<bounds h.input) :
    (loadOp c).SafeAt bounds u := by
  simp only [loadOp,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],by simp [Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ptrs ran i active
  simp only [evalOp_ref,hp] at ran
  cases ran
  rcases active with ⟨m,hmrun,hi⟩
  simp only [evalOp_ref,hm] at hmrun
  cases hmrun
  have ha : Geometry.active h c q i.1 := of_decide_eq_true hi
  exact hb _ (by rw [Geometry.active_time h c q i.1 ha]; exact (Geometry.active_iff h c q i.1 start).mp ha)

private theorem store_safe (h : Call) (c : Config) (row q : Nat)
    (bounds : RegionBounds) (u : BlockState) (start : q*c.BT<h.T)
    (hp : u.regs .ptr [c.BT] "p_o"=some (pointer h c row q h.output h.outputStrides))
    (hm : u.regs .bool [c.BT] "active"=some (mask h c q))
    (hb : ∀ t,t<h.T → h.outputStrides.address h.H row t<bounds h.output) :
    Stmt.TraceSafe bounds (.store .real [c.BT] (.ptr (.ref .ptr [c.BT] "p_o"))
      (.ref .real [c.BT] "b_o") (.mask (.ref .bool [c.BT] "active"))) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt],by simp [Op.SafeAt],
    by simp [MaskOpt.SafeAt,Op.SafeAt],?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ptrs ran i active
  simp only [evalOp_ref,hp] at ran
  cases ran
  rcases active with ⟨m,hmrun,hi⟩
  simp only [evalOp_ref,hm] at hmrun
  cases hmrun
  have ha : Geometry.active h c q i.1 := of_decide_eq_true hi
  exact hb _ (by rw [Geometry.active_time h c q i.1 ha]; exact (Geometry.active_iff h c q i.1 start).mp ha)

theorem iteration_safe (h : Call) (c : Config) (row : Nat)
    (bounds : RegionBounds)
    (hin : ∀ t,t<h.T → h.inputStrides.address h.H row t<bounds h.input)
    (hout : ∀ t,t<h.T → h.outputStrides.address h.H row t<bounds h.output)
    (j : Int) (u : BlockState) (hj : j<1) (start : (-j).toNat*c.BT<h.T)
    (hr : u.regs .int [] "__rev_i_t"=some (Tile.scalar j))
    (hh : u.regs .int [] "i_bh"=some (Tile.scalar (row:Int))) :
    Stmt.TraceSafeList bounds (loopBody h c) u := by
  unfold loopBody
  rw [cons_of_run (stepStmt_assign_eq_some (RealStep.index_eval j u hj hr))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [cons_of_run (stepStmt_assign_eq_some (RealValues.start_eval c (-j).toNat _ (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,startOp,Op.SafeAt],?_⟩
  rw [cons_of_run (stepStmt_assign_eq_some (show evalOp (.arange c.BT) _=some (lanes c) from by simpa [lanes,Tile.vec] using evalOp_arange c.BT _))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [cons_of_run (stepStmt_assign_eq_some (active_eval h c (-j).toNat _ start (by simp) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,activeOp,Op.SafeAt],?_⟩
  rw [cons_of_run (stepStmt_assign_eq_some (time_eval h c (-j).toNat _ (by simp) (by simp) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,timeOp,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  rw [cons_of_run (stepStmt_assign_eq_some (pointer_eval h c row (-j).toNat h.input h.inputStrides _ (by simp [hh]) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,pointerOp,baseOp,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  rw [cons_of_run (stepStmt_assign_eq_some (pointer_eval h c row (-j).toNat h.output h.outputStrides _ (by simp [hh]) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,pointerOp,baseOp,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  apply assign_cons (load_safe h c row (-j).toNat bounds _ start (by simp) (by simp) hin)
  intro bs
  apply assign_cons
  · simp [Op.SafeAt,Op.SafeAt.eq_def]
  intro sum
  apply assign_cons
  · simp [Op.SafeAt,Op.SafeAt.eq_def]
  intro carry
  apply assign_cons
  · simp [VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.outputOp,Op.SafeAt,Op.SafeAt.eq_def]
  intro out
  apply Stmt.TraceSafeList.cons_intro (store_safe h c row (-j).toNat bounds _ start (by simp) (by simp) hout)
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem loop_safe (initial : BlockState) (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (positive : 0<c.BT) (row : Nat) (hr : row<h.B*h.H)
    (input : Nat → ℝ) (u : BlockState) (inv : Holds initial h c row input (numChunks h c) u)
    (bounds : RegionBounds)
    (hin : ∀ t,t<h.T → h.inputStrides.address h.H row t<bounds h.input)
    (hout : ∀ t,t<h.T → h.outputStrides.address h.H row t<bounds h.output) :
    Stmt.TraceSafe bounds (statement h c) u := by
  rw [statement,Stmt.TraceSafe]
  refine ⟨by simp [startIndexOp,Op.SafeAt],
    by simp [Op.SafeAt],by simp [Op.SafeAt],?_⟩
  simp only [startIndex_eval,evalOp]
  change Stmt.forRangeIntTraceSafe bounds "__rev_i_t" (1-(numChunks h c:Int))
    1 1 (loopBody h c) u
  apply VeriTile.Bench.TritonBenchG.ReversedCumsum.SignedRange.safe bounds "__rev_i_t" 1 1 (loopBody h c)
    (fun j v => Holds initial h c row input
      (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.remaining j) v)
  · intro j v hj hv
    have hc := hv.count_le
    rw [VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.current_chunk j hj] at hc
    have start := Geometry.chunk_start h c positive (-j).toNat (Nat.lt_of_succ_le hc)
    exact iteration_safe h c row bounds hin hout j _ hj start (by simp) (by simpa using hv.head)
  · intro j v w hj hv ran
    obtain ⟨next,he,hi⟩ := loop_iteration initial h layout c positive row hr input j v hj hv
    rw [ran] at he
    cases he
    exact hi
  · simpa only [VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex.at_start] using inv


theorem source_safe (h : Call) (layout : Geometry.OutputLayout h) (c : Config)
    (positive : 0<c.BT) (u : BlockState) (hr : u.pids 0<h.B*h.H) (bounds : RegionBounds)
    (hin : ∀ t,t<h.T → h.inputStrides.address h.H (u.pids 0) t<bounds h.input)
    (hout : ∀ t,t<h.T → h.outputStrides.address h.H (u.pids 0) t<bounds h.output) :
    Kernel.TraceSafe bounds (Kernels.source h c).eraseDType.toAlgKernel u := by
  rw [Kernel.TraceSafe,source_body,body]
  apply TraceProofs.append_safe
  · apply TraceProofs.all_safe
    simp [initCode,VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealProgram.initCode,
      Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
  · intro v he
    rw [init_executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro
    · exact loop_safe u h layout c positive (u.pids 0) hr _ (initial u)
        (invariant h c positive u) bounds hin hout
    · intro _ _
      exact Stmt.TraceSafeList.nil_intro

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.SourceSafety
