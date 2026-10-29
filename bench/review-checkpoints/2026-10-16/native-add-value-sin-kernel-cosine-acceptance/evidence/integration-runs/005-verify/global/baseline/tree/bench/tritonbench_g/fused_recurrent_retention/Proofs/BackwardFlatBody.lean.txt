import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondSafety
import VeriTile.Triton.Memory.FlattenTolerant

/-! Exact ten-statement numerical/store core and six-statement pointer tail.
The tail is executed unchanged even when a final subtraction clips. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatBody
open VeriTile.Triton BackwardSource BackwardSecondState
noncomputable section
set_option maxHeartbeats 1300000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def core (p : Layout) : List Stmt := (secondBody p).take 10
def tail (p : Layout) : List Stmt := (secondBody p).drop 10

theorem split (p : Layout) : secondBody p = core p ++ tail p :=
  (List.take_append_drop 10 (secondBody p)).symm

theorem tail_shape (p : Layout) : StmtList.PtrDecTail (tail p) := by
  intro st hs
  simp only [tail,secondBody,List.drop,List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact ⟨_,_,.scalarR,_,rfl⟩

theorem core_flatten_ok (p : Layout) : StmtList.FlattenOkR (core p) := by
  simp [core,secondBody,ForwardSource.vectorLoad,dStateUpdate,dkOp,dvOp,
    StmtList.FlattenOkR,Stmt.FlattenOkR,Op.FlattenOkR.eq_def]

private theorem trace_take (bounds : RegionBounds) :
    ∀ (n : Nat) (code : List Stmt) (u : BlockState),
      Stmt.TraceSafeList bounds code u → Stmt.TraceSafeList bounds (code.take n) u
  | 0, _, _, _ => Stmt.TraceSafeList.nil_intro
  | _+1, [], _, _ => Stmt.TraceSafeList.nil_intro
  | n+1, st::rest, u, h => by
      rw [Stmt.TraceSafeList] at h
      apply Stmt.TraceSafeList.cons_intro h.1
      intro v hv
      have hr := h.2
      rw [hv] at hr
      exact trace_take bounds n rest v hr

private theorem trace_toR (bounds : RegionBounds) :
    ∀ (code : List Stmt) (u : BlockState),
      (∀ st ∈ code, ∀ v, Stmt.TraceSafe bounds st v → Stmt.TraceSafeR .triv bounds st v) →
      Stmt.TraceSafeList bounds code u → Stmt.TraceSafeListR .triv bounds code u
  | [], _, _, _ => Stmt.TraceSafeListR.nil_intro
  | st::rest, u, hc, h => by
      rw [Stmt.TraceSafeList] at h
      apply Stmt.TraceSafeListR.cons_intro (hc st List.mem_cons_self u h.1)
      intro v hv
      rw [stepStmtR_triv] at hv
      have hr := h.2
      rw [hv] at hr
      exact trace_toR bounds rest v (fun st hs => hc st (List.mem_cons_of_mem _ hs)) hr

theorem core_safe (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (bounds : RegionBounds) (hb : BackwardSafety.Bounds p l r hasSeed bounds)
    (t : Nat) (ht : t < p.T) (u : BlockState)
    (h : BackwardSecondLoop.Holds p l initial r hasSeed (t+1) u) :
    Stmt.TraceSafeListR .triv bounds (core p) u := by
  apply trace_toR bounds (core p) u
  · intro st hs v hv
    simp only [core,secondBody,List.take,List.mem_cons,List.not_mem_nil,or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals simpa [Stmt.TraceSafeR,Stmt.TraceSafe,ForwardSource.vectorLoad,
      dStateUpdate,dkOp,dvOp,Op.SafeAtR.eq_def,Op.SafeAt.eq_def,
      MemAccess.SafeAtR,MemAccess.SafeAt,MaskOpt.SafeAtR,MaskOpt.SafeAt,
      MemAccess.ActiveAddressSafeR,memAccessActiveAddressSafeR,MemAccess.ActiveAddressSafe,
      memAccessActiveAddressSafe,Op.PointerAddressesSafeOn,Op.MemorySafe,
      MaskOpt.ActiveR,MaskOpt.Active,evalOpR_triv] using hv
  · exact trace_take bounds 10 (secondBody p) u
      (BackwardSafety.second_iteration_safe p l initial r hasSeed bounds hb t ht u h)

def afterCore (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) : BlockState :=
  scatterV p l initial r t (scatterK p l initial r t (beforeStores p l initial r t u))

theorem core_executes (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) (ht : t < p.T) (u : BlockState)
    (h : BackwardSecondLoop.Holds p l initial r hasSeed (t+1) u) :
    stepStmts (core p) u = some (afterCore p l initial r t u) := by
  have hsplit : core p = (secondBody p).take 8 ++
      ((secondBody p).drop 8).take 2 := by rfl
  rw [hsplit,stepStmts.append_some (BackwardSecondValues.before_stores p l initial r t ht u
    h.ready h.qmem h.kmem h.vmem h.domem)]
  dsimp only [secondBody,List.drop,List.take]
  rw [stepStmts.cons_some (BackwardSecondStep.key_store p l initial r t u h.ready),
    stepStmts.cons_some (BackwardSecondStep.value_store p l initial r t u h.ready),stepStmts.nil]
  rfl

theorem tail_frame (p : Layout) (u v : BlockState)
    (h : stepStmts (tail p) u = some v) : v.ObsAgree u := by
  exact stepStmtsR_ptrDecTail_frame .triv (tail p) u v (tail_shape p)
    (by simpa only [stepStmtsR_triv] using h)

theorem tail_tolerant (A : FlatAlloc) (p : Layout) (u v flat : BlockState)
    (he : stepStmts (tail p) u = some v) (hd : RegsDefEq flat u)
    (ho : flat.ObsAgree (A.flattenState u)) :
    ∃ out, stepStmts (tail p) flat = some out ∧
      out.ObsAgree (A.flattenState v) ∧ RegsDefEq out v := by
  simpa only [stepStmtsR_triv] using
    A.stepStmtsR_ptrDecTail_tolerant .triv (tail p) u v flat (tail_shape p)
      (by simpa only [stepStmtsR_triv] using he) hd ho

#axiomsClean split
#axiomsClean tail_shape
#axiomsClean core_flatten_ok
#axiomsClean core_safe
#axiomsClean core_executes
#axiomsClean tail_frame
#axiomsClean tail_tolerant
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatBody
