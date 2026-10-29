import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Loop
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Stores

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedLoop
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001 TermBody SharedStep
noncomputable section
set_option maxHeartbeats 8000000

theorem output_earlier (c : Shape) (head chunk later : Nat) (i : Fin 32) (j : Fin c.D)
    (earlier : chunk<later) :
    mainAddress c head chunk i.val j.val<head*c.headSize+later*32*c.D := by
  have multiple := Nat.mul_le_mul_right 32 (Nat.succ_le_iff.mpr earlier)
  have hi := i.isLt
  have row : chunk*32+i.val<later*32 := by omega
  have cell := digit_bound _ _ _ _ row j.isLt
  unfold mainAddress
  omega

theorem history_earlier (c : Shape) (head chunk later : Nat) (i j : Fin c.D)
    (earlier : chunk<later) :
    historyAddress c head chunk i.val j.val<head*c.historyHead+later*c.D*c.D := by
  have multiple := Nat.mul_le_mul_right c.D (Nat.succ_le_iff.mpr earlier)
  simp only [Nat.succ_mul] at multiple
  have hi := i.isLt
  have row : chunk*c.D+i.val<later*c.D := by omega
  have cell := digit_bound _ _ _ _ row j.isLt
  unfold historyAddress
  omega

theorem output [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) (chunk : Nat) (reached : chunk<n)
    (i : Fin 32) (j : Fin c.D) (active : chunk*32+i.val<c.T) :
    (state c scale head storeFlag ifcond s n).mem "o" (mainAddress c head chunk i.val j.val)=
      .mk .real (TermBody.cast (Graph.output c (read s "q") (read s "k") (read s "v")
        scale head chunk ifcond (i,j,PUnit.unit))) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases earlier : chunk<n
    · rw [state,finished_output_before _ _ _ _ _ _ _ _ _ (output_earlier c head chunk n i j earlier)]
      exact ih earlier
    · have equal : chunk=n := by omega
      subst chunk
      rw [state,finished_output _ _ _ _ _ _ _ _ i j active]
      change Cell.mk .real (TermBody.cast (TermBody.output c
        (read (state c scale head storeFlag ifcond s n) "q")
        (read (state c scale head storeFlag ifcond s n) "k")
        (read (state c scale head storeFlag ifcond s n) "v") scale head n
        (Graph.carry c (read s "k") (read s "v") head ifcond n) ifcond (i,j,PUnit.unit)))=_
      rw [reads c scale head storeFlag ifcond s n "q" (by decide) (by decide),
        reads c scale head storeFlag ifcond s n "k" (by decide) (by decide),
        reads c scale head storeFlag ifcond s n "v" (by decide) (by decide)]
      rfl

theorem history [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (ifcond : Bool)
    (s : State (Term α)) (n : Nat) (chunk : Nat) (reached : chunk<n) (i j : Fin c.D) :
    (state c scale head Bool.true ifcond s n).mem "h" (historyAddress c head chunk i.val j.val)=
      .mk .real (TermBody.cast (Graph.carry c (read s "k") (read s "v") head ifcond chunk (i,j,PUnit.unit))) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases earlier : chunk<n
    · rw [state,finished_history_before _ _ _ _ _ _ _ _ _ (history_earlier c head chunk n i j earlier)]
      exact ih earlier
    · have equal : chunk=n := by omega
      subst chunk
      rw [state,finished_history]

theorem output_before [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) (off : Nat) (before : off<head*c.headSize) :
    (state c scale head storeFlag ifcond s n).mem "o" off=s.mem "o" off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [state,finished_output_before _ _ _ _ _ _ _ _ off (by omega)]
    exact ih

theorem history_before [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (n : Nat) (off : Nat) (before : off<head*c.historyHead) :
    (state c scale head storeFlag ifcond s n).mem "h" off=s.mem "h" off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [state,finished_history_before _ _ _ _ _ _ _ _ off (by omega)]
    exact ih

theorem history_disabled [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (ifcond : Bool)
    (s : State (Term α)) (n : Nat) (off : Nat) :
    (state c scale head Bool.false ifcond s n).mem "h" off=s.mem "h" off := by
  induction n with
  | zero => rfl
  | succ n ih => rw [state,finished_history_disabled]; exact ih

theorem frame [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat) (storeFlag ifcond : Bool)
    (s : State (Term α)) (headBound : head<c.heads) (n : Nat) (count : n≤c.chunks)
    (r : RegionName) (off : Nat) (ho : r≠"o" ∨ c.mainSize≤off)
    (hh : storeFlag=Bool.false ∨ r≠"h" ∨ c.historySize≤off) :
    (state c scale head storeFlag ifcond s n).mem r off=s.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [state,finished_frame _ _ _ _ _ _ _ _ headBound (by omega) r off ho hh]
    exact ih (by omega)

#axiomsClean output
#axiomsClean history
#axiomsClean frame
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.SharedLoop
