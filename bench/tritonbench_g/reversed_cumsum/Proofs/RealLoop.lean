import bench.tritonbench_g.reversed_cumsum.Proofs.RealStep
import VeriTile.Triton.Memory.Frame.Basic
import bench.tritonbench_g.reversed_cumsum.Proofs.Memory

/-! Descending chunk invariant derived from initial inputs and unspecified
values. It preserves later outputs and frames complete tagged cells outside
the active output window; the complete source-loop connection is separate. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop
open VeriTile.Triton RealProgram RealStep
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def carry (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) (j : Fin p.BS) : ℝ :=
  RealMath.carry (fun t => value p l a padding t j) p.BT (numChunks p) c

def expected (p : Layout) (l : Launch) (a padding : Nat → ℝ) (t : Nat) (j : Fin p.BS) : ℝ :=
  RealMath.suffixValue (fun k => value p l a padding k j) (numChunks p * p.BT) t

/-- Only active cells must have distinct addresses, across the whole time span. -/
def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (t u : Nat) (i j : Fin p.BS), RealStep.active p l t i.val → RealStep.active p l u j.val →
    address p l t i.val = address p l u j.val → t = u ∧ i = j

def footprint (p : Layout) (l : Launch) (o : RegionName) : WriteFootprint :=
  fun cell => cell.1 = o ∧ ∃ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val ∧
    cell.2 = address p l t j.val

theorem next_carry (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) :
    nextCarry p l a padding c (carry p l a padding (c + 1)) = carry p l a padding c := by
  funext j
  exact RealMath.carry_predecessor (fun t => value p l a padding t j) p.BT (numChunks p) c

/-- The observed suffix includes the tail oracle from the initial state. -/
theorem suffix_at_chunk (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (i : Fin p.BT) (j : Fin p.BS) :
    carry p l a padding (c + 1) j + chunkSuffix p l a padding c i j =
      expected p l a padding (c * p.BT + i.val) j := by
  exact RealMath.suffix_from_chunk (fun t => value p l a padding t j) p.BT (numChunks p) c i

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (s o : RegionName)
    (a : Nat → ℝ) (c : Nat) (u : BlockState) : Prop where
  undef : u.undef = initial.undef
  count_le : c ≤ numChunks p
  feature : u.regs .nat [] "i_s" = some (Tile.scalar l.feature)
  head : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  triangular : u.regs .real [p.BT, p.BT] "m_s" = some (RealMath.upperTriTile p.BT)
  accumulated : u.regs .real [p.BS] "b_z" =
    some (Tile.vec fun j => some (carry p l a (initial.undef s) c j))
  input : ∀ off, u.readMem s off = a off
  output : ∀ (t : Nat) (j : Fin p.BS), RealStep.active p l t j.val → c * p.BT ≤ t →
    u.mem o (address p l t j.val) = MemCell.real (expected p l a (initial.undef s) t j)
  frame : BlockState.WriteWithin (footprint p l o) initial u

/-- The scatter changes only active cells of the current chunk. -/
theorem scatter_frame (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (acc : Fin p.BS → ℝ) (u : BlockState) :
    BlockState.WriteWithin (footprint p l o) u (scatter p l o a padding c acc u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  exact ⟨rfl, c * p.BT + idx.1.val, idx.2.1, ha, rfl⟩

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (s o : RegionName)
    (a : Nat → ℝ) (hSO : s ≠ o) (_hBT : 0 < p.BT) (hinj : NoCollision p l)
    (c : Nat) (u : BlockState) (hc : c < numChunks p)
    (h : Holds initial p l s o a (c + 1) u) :
    ∃ v, stepStmts (loopBody p s o) (u.setReg "i_t" .int [] (Tile.scalar (c : Int))) = some v ∧
      Holds initial p l s o a c v := by
  let entry := u.setReg "i_t" .int [] (Tile.scalar (c : Int))
  let pre := beforeStore p l s o a (initial.undef s) c (carry p l a (initial.undef s) (c + 1)) entry
  let v := result p l s o a (initial.undef s) c (carry p l a (initial.undef s) (c + 1)) entry
  have he := RealStep.executes p l s o a (initial.undef s) c
    (carry p l a (initial.undef s) (c + 1)) entry (by intro off; simp [entry, h.undef])
    (by simp [entry, h.feature]) (by simp [entry, h.head]) (by simp [entry])
    (by intro off; simpa [entry] using h.input off)
    (by simp [entry, h.triangular]) (by simp [entry, h.accumulated])
  refine ⟨v, he, ?_, Nat.le_of_lt hc, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have hu := stepStmts_undef (loopBody p s o) entry v (by
      simp [loopBody, pointerOp, carryOp, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
      all_goals simp [Op.FlattenOk.eq_def]) he
    exact hu.trans (by simpa [entry] using h.undef)
  · simp [v, result, scatter, beforeStore, entry, h.feature]
  · simp [v, result, scatter, beforeStore, entry, h.head]
  · simp [v, result, scatter, beforeStore, entry, h.triangular]
  · simp [v, result, next_carry p l a (initial.undef s) c]
  · intro off
    change (scatter p l o a (initial.undef s) c (carry p l a (initial.undef s) (c + 1)) pre).readMem s off = a off
    rw [scatter, BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hSO]
    simpa [pre, beforeStore, entry] using h.input off
  · intro t j ha ht
    change (scatter p l o a (initial.undef s) c (carry p l a (initial.undef s) (c + 1)) pre).mem o (address p l t j.val) = _
    by_cases hprior : (c + 1) * p.BT ≤ t
    · rw [scatter, foldl_store_preserve_cell]
      · simpa [pre, beforeStore, entry] using h.output t j ha hprior
      · intro idx _ hidx heq
        have := (hinj _ _ idx.2.1 j hidx ha heq.2).1
        have hi := idx.1.isLt
        rw [Nat.add_mul, Nat.one_mul] at hprior
        omega
    · have hi : t - c * p.BT < p.BT := by
        rw [Nat.add_mul, Nat.one_mul] at hprior
        omega
      let i : Fin p.BT := ⟨t - c * p.BT, hi⟩
      have hti : c * p.BT + i.val = t := by dsimp [i]; omega
      have hactive : RealStep.active p l (c * p.BT + i.val) j.val := by simpa [hti] using ha
      have hread := Memory.scatter_cell o [p.BT,p.BS] pre
        (fun idx : TileIndex [p.BT, p.BS] => address p l (c * p.BT + idx.1.val) idx.2.1.val)
        (fun idx => carry p l a (initial.undef s) (c + 1) idx.2.1 + chunkSuffix p l a (initial.undef s) c idx.1 idx.2.1)
        (fun idx => RealStep.active p l (c * p.BT + idx.1.val) idx.2.1.val)
        (i,j,PUnit.unit) hactive (by
          rintro ⟨k,e,⟨⟩⟩ hk heq
          obtain ⟨hrow, hcol⟩ := hinj _ _ e j hk hactive heq
          dsimp only at hrow hcol
          have hi' : k = i := Fin.ext (by omega)
          subst k; subst e; rfl)
      change (scatter p l o a (initial.undef s) c (carry p l a (initial.undef s) (c + 1)) pre).mem o
        (address p l (c * p.BT + i.val) j.val) = _ at hread
      dsimp only at hread
      rw [hti] at hread
      rw [hread, suffix_at_chunk p l a (initial.undef s) c i j, hti]
  · have hs := scatter_frame p l o a (initial.undef s) c (carry p l a (initial.undef s) (c + 1)) pre
    intro r off hn
    exact (h.frame r off hn).trans (hs r off hn)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop
