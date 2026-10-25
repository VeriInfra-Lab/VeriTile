import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealAccessStep

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealAccess
open VeriTile.Triton IndexProgram RealProgram RealValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def prefixCode (h : Call) (c : Config) : List Stmt := (RealProgram.loopBody h c).take 10

theorem prefix_safe (bounds : RegionBounds) (h : Call) (c : Config) (feature row q : Nat)
    (f : Nat → Fin BS → ℝ) (acc : Fin BS → ℝ) (u : BlockState)
    (ht : u.regs .int [] "i_t"=some (Tile.scalar q))
    (hb : u.regs .int [] "i_b"=some (Tile.scalar (row/h.H)))
    (hh : u.regs .int [] "i_h"=some (Tile.scalar (row%h.H)))
    (hs : u.regs .bool [BS] "active_s"=some (axisMask h.S BS (feature*BS)))
    (hj : u.regs .int [BS] "columns"=some (coordinate h.S BS (feature*BS)))
    (hm : u.regs .real [c.BT,c.BT] "m_s"=some (lowerTriTile c.BT))
    (ha : u.regs .real [BS] "b_z"=some (Tile.vec fun j => some (acc j)))
    (read : ∀ (i : Fin c.BT) (j : Fin BS),
      (if Geometry.active h c feature q i.val j.val then
        u.readMem h.input (h.inputStrides.address h.H row
          (Geometry.time h c q i.val) (Geometry.column h feature j.val))
      else 0)=f (q*c.BT+i.val) j)
    (boundedIn : ∀ i : TileIndex [c.BT,BS], Geometry.active h c feature q i.1.val i.2.1.val →
      h.inputStrides.address h.H row (Geometry.time h c q i.1.val)
        (Geometry.column h feature i.2.1.val) < bounds h.input)
    (boundedOut : ∀ i : TileIndex [c.BT,BS], Geometry.active h c feature q i.1.val i.2.1.val →
      h.outputStrides.address h.H row (Geometry.time h c q i.1.val)
        (Geometry.column h feature i.2.1.val) < bounds h.output)
    (tail : List Stmt) (tailSafe : ∀ v, Stmt.TraceSafeListR .triv bounds tail v) :
    Stmt.TraceSafeListR .triv bounds (prefixCode h c ++ tail) u := by
  simp only [prefixCode, RealProgram.loopBody, List.take, List.cons_append, List.nil_append]
  refine assign bounds _ _ _ _ _ _ _ ?_ (timeStart_eval c q u ht) ?_
  · simp [Op.SafeAtR.eq_def, timeStartOp, startOp]
  refine assign bounds _ _ _ _ _ _ _ ?_
    (show evalOp (.castNatToInt (.arange c.BT)) _ = some (lanes c.BT) from by simp [evalOp,lanes,Tile.vec]) ?_
  · simp [Op.SafeAtR.eq_def]
  refine assign bounds _ _ _ _ _ _ _ ?_
    (active_eval h.T c.BT (q*c.BT) "lanes_t" "start_t" _ (by simp) (by simp)) ?_
  · simp [Op.SafeAtR.eq_def, activeOp]
  refine assign bounds _ _ _ _ _ _ _ ?_
    (coordinate_eval h.T c.BT (q*c.BT) "start_t" "active_t" "lanes_t" _
      (by simp) (by simp) (by simp)) ?_
  · simp [Op.SafeAtR.eq_def, coordinateOp]
  refine assign bounds _ _ _ _ _ _ _ ?_
    (mask_eval h c feature q _ (by simp) (by simp [hs])) ?_
  · simp [Op.SafeAtR.eq_def, maskOp]
  refine assign bounds _ _ _ _ _ _ _ ?_
    (pointer_eval h c feature row q h.input h.inputStrides _
      (by simp [hb]) (by simp [hh]) (by simp) (by simp [hj])) ?_
  · simp [Op.SafeAtR.eq_def, pointerOp]
  refine assign bounds _ _ _ _ _ _ _ ?_
    (pointer_eval h c feature row q h.output h.outputStrides _
      (by simp [hb]) (by simp [hh]) (by simp) (by simp [hj])) ?_
  · simp [Op.SafeAtR.eq_def, pointerOp]
  refine assign bounds _ _ _ _ _ _ _ ?_
    (load_eval h c feature row q f _ (by simp) (by simp) (by simpa using read)) ?_
  · exact load_safe bounds h c feature row q _ (by simp) (by simp) boundedIn
  refine assign bounds _ _ _ _ _ _ _ ?_
    (output_eval c f q acc _ (by simp [ha]) (by simp [hm]) (by simp)) ?_
  · simp [Op.SafeAtR.eq_def, outputOp]
  apply Stmt.TraceSafeListR.cons_intro
  · exact store_safe bounds h c feature row q _ (by simp) (by simp) boundedOut
  · intro v _
    exact tailSafe v

theorem carry_safe (bounds : RegionBounds) (c : Config) (u : BlockState) :
    Stmt.TraceSafeListR .triv bounds
      [.ifThen (.ge .int .nil (.ref .int [] "i_t") (.constInt 0))
        [.assign .real [BS] "b_z" (carryOp c)]] u := by
  apply Stmt.TraceSafeListR.cons_intro
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, true_and]
    split
    next condition =>
      split
      next =>
        apply Stmt.TraceSafeListR.of_forall
        intro st member s
        simp only [List.mem_singleton] at member
        subst st
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def, carryOp]
      next => trivial
    next => trivial
  · intro _ _
    exact Stmt.TraceSafeListR.nil_intro

#axiomsClean prefix_safe
#axiomsClean carry_safe
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealAccess
