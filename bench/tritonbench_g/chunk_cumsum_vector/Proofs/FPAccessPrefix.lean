import bench.tritonbench_g.chunk_cumsum_vector.Proofs.FPAccessStep

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPAccess
open VeriTile.Triton FP.Structural IndexProgram FPProgram FPValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The common actual-source prefix through the store. -/
def prefixCode (h : Call) (c : Config) : List ComputeStmt :=
  (FPProgram.loopBody h c).take 10

theorem prefix_safe (M : Algebra α) (dot : DotBackend α) (bounds : RegionBounds)
    (h : Call) (c : Config) (feature row q : Nat)
    (initial : State α) (acc : Values α .real [BS]) (u : State α)
    (ht : u.regs .nat [] "i_t"=some (fun _ => q))
    (hb : u.regs .nat [] "i_b"=some (fun _ => row/h.H))
    (hh : u.regs .nat [] "i_h"=some (fun _ => row%h.H))
    (hs : u.regs .bool [BS] "active_s"=some (axisMask h.S BS (feature*BS)))
    (hj : u.regs .nat [BS] "columns"=some (coordinate h.S BS (feature*BS)))
    (hm : u.regs .real [c.BT,c.BT] "m_s"=some (triangular M c))
    (ha : u.regs .real [BS] "b_z"=some acc)
    (read : ∀ off, u.mem h.input off=initial.mem h.input off) (padding : u.undef=initial.undef)
    (boundedIn : ∀ i : TileIndex [c.BT,BS], Geometry.active h c feature q i.1.val i.2.1.val →
      h.inputStrides.address h.H row (Geometry.time h c q i.1.val)
        (Geometry.column h feature i.2.1.val) < bounds h.input)
    (boundedOut : ∀ i : TileIndex [c.BT,BS], Geometry.active h c feature q i.1.val i.2.1.val →
      h.outputStrides.address h.H row (Geometry.time h c q i.1.val)
        (Geometry.column h feature i.2.1.val) < bounds h.output)
    (tail : List ComputeStmt) (tailSafe : ∀ v, Safety.traceSafeList (machine M dot) bounds tail v) :
    Safety.traceSafeList (machine M dot) bounds (prefixCode h c ++ tail) u := by
  simp only [prefixCode, FPProgram.loopBody, List.take, List.cons_append, List.nil_append]
  refine assign_alg _ bounds _ _ _ _ _ _ _ ?_ (start_eval (machine M dot) "i_t" c.BT q u ht) ?_
  · simp [Safety.safeOp_unfold, startOp]
  refine assign_alg _ bounds _ _ _ _ _ _ _ ?_
    (show FP.Structural.evalOp (machine M dot) none (.arange c.BT) _ = some (lanes c.BT) from rfl) ?_
  · simp [Safety.safeOp_unfold]
  refine assign_alg _ bounds _ _ _ _ _ _ _ ?_
    (active_eval (machine M dot) h.T c.BT (q*c.BT) "lanes_t" "start_t" _ (by simp) (by simp)) ?_
  · simp [Safety.safeOp_unfold, activeOp]
  refine assign_alg _ bounds _ _ _ _ _ _ _ ?_
    (coordinate_eval (machine M dot) h.T c.BT (q*c.BT) "start_t" "active_t" "lanes_t" _
      (by simp) (by simp) (by simp)) ?_
  · simp [Safety.safeOp_unfold, coordinateOp]
  refine assign_alg _ bounds _ _ _ _ _ _ _ ?_
    (mask_eval (machine M dot) h c feature q _ (by simp) (by simp [hs])) ?_
  · simp [Safety.safeOp_unfold, maskOp]
  refine assign_alg _ bounds _ _ _ _ _ _ _ ?_
    (pointer_eval (machine M dot) h c feature row q h.input h.inputStrides _
      (by simp [hb]) (by simp [hh]) (by simp) (by simp [hj])) ?_
  · simp [Safety.safeOp_unfold, pointerOp]
  refine assign_alg _ bounds _ _ _ _ _ _ _ ?_
    (pointer_eval (machine M dot) h c feature row q h.output h.outputStrides _
      (by simp [hb]) (by simp [hh]) (by simp) (by simp [hj])) ?_
  · simp [Safety.safeOp_unfold, pointerOp]
  refine assign_compute _ bounds _ _ _ _ _ _ ?_
    (load_eval M dot h c feature row q initial _ (by simp) (by simp)
      (by simpa using read) (by simpa using padding)) ?_
  · exact load_safe _ bounds h c feature row q _ (by simp) (by simp) boundedIn
  refine assign_compute _ bounds _ _ _ _ _ _ ?_
    (output_eval M dot c (loaded M h c feature row initial q) acc _
      (by simp [ha]) (by simp [hm]) (by simp)) ?_
  · simp [Safety.safeOp_unfold, outputOp]
  apply Safety.traceSafeList_cons_intro
  · exact store_safe _ bounds h c feature row q _ (by simp) (by simp) boundedOut
  · intro v _
    exact tailSafe v

/-- Both carry-update orders contain no memory accesses. -/
theorem carry_safe (M : Algebra α) (bounds : RegionBounds) (c : Config)
    (commuted : Bool) (u : State α) :
    Safety.traceSafeList M bounds
      [.ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
        [.assign .real [BS] "b_z" (.compute (.alg .fp32
          (if commuted then CandidateProgram.sumLeftCarryOp c else carryOp c)))]] u := by
  rw [Safety.traceSafeList_cons]
  refine ⟨?_, fun _ _ => Safety.traceSafeList_nil _ _ _⟩
  simp only [Safety.stepSafe_unfold, Safety.safeOp_unfold, true_and]
  split
  next condition =>
    split
    next =>
      cases commuted <;>
       simp [Safety.traceSafeList_cons, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
        Safety.safeCompute_unfold, Safety.safeOp_unfold, carryOp,
        CandidateProgram.sumLeftCarryOp]
    next => trivial
  next => trivial

#axiomsClean prefix_safe
#axiomsClean carry_safe
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPAccess
