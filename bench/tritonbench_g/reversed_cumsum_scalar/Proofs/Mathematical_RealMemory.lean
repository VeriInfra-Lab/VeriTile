import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Mathematical_RealValues
import VeriTile.Triton.Memory.Frame.Basic
import VeriTile.Triton.KernelLemmas.ScatterStore

/-! The source's masked block store, including full tagged-cell output and
frame properties. Each chunk writes its actual computed value at active lanes
only; this module does not assume a loop-level output formula. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealMemory
open VeriTile.Triton RealValues
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def storedValue (BT c : Nat) (f : Nat → ℝ) (acc : ℝ) (i : Fin BT) : ℝ :=
  f (c * BT + i.val) - chunkPrefix BT c f i + acc

def scatter (T BT row : Nat) (o : RegionName) (c : Nat) (f : Nat → ℝ)
    (acc : ℝ) (u : BlockState) : BlockState :=
  (TileShape.allIndices [BT]).foldl (fun t idx =>
    if c * BT + idx.1.val < T then
      t.writeMem o (address T row (c * BT + idx.1.val)) (storedValue BT c f acc idx.1)
    else t) u

def footprint (T row : Nat) (o : RegionName) : WriteFootprint :=
  fun cell => cell.1 = o ∧ ∃ t : Nat, t < T ∧ cell.2 = address T row t

theorem store_eval (T BT row : Nat) (o : RegionName) (c : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState)
    (hp : u.regs .blockPtr [BT] "p_o" = some (pointer T BT row o c))
    (hv : u.regs .real [BT] "b_o" = some (outputTile BT c f acc)) :
    stepStmt (.store .real [BT] (.blockPtr (.ref .blockPtr [BT] "p_o") [0])
      (.ref .real [BT] "b_o") .none) u = some (scatter T BT row o c f acc u) := by
  simp only [stepStmt, evalOp_ref, hp, hv, Option.bind_some, Option.bind_eq_bind,
    Bool.true_and, pointer, TileShape.indexToList, BlockPtr.inBounds_1d,
    BlockPtr.address_1d, Nat.mul_one, decide_eq_true_eq]
  rfl

theorem frame (T BT row : Nat) (o : RegionName) (c : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) :
    BlockState.WriteWithin (footprint T row o) u (scatter T BT row o c f acc u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  exact ⟨rfl, c * BT + idx.1.val, ha, rfl⟩

theorem cell (T BT row : Nat) (o : RegionName) (c : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) (i : Fin BT)
    (hi : c * BT + i.val < T) :
    (scatter T BT row o c f acc u).mem o (address T row (c * BT + i.val)) =
      MemCell.real (storedValue BT c f acc i) := by
  let idx : TileIndex [BT] := (i, PUnit.unit)
  obtain ⟨before, after, he⟩ := List.append_of_mem (TileShape.mem_allIndices [BT] idx)
  have hn := TileShape.allIndices_nodup [BT]
  rw [he, List.nodup_append, List.nodup_cons] at hn
  have hafter : idx ∉ after := hn.2.1.1
  unfold scatter
  rw [he, List.foldl_append, List.foldl_cons, if_pos hi]
  rw [foldl_store_preserve_cell]
  · simp [BlockState.writeMem, idx]
  · rintro ⟨k,⟨⟩⟩ hk _ heq
    have hk' : k = i := by
      apply Fin.ext
      have haddr := heq.2
      simp only [address] at haddr
      omega
    subst k
    exact hafter hk

theorem preserves_later (T BT row : Nat) (o : RegionName) (c : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) (t : Nat)
    (ht : (c + 1) * BT ≤ t) :
    (scatter T BT row o c f acc u).mem o (address T row t) = u.mem o (address T row t) := by
  unfold scatter
  apply foldl_store_preserve_cell
  rintro ⟨i,⟨⟩⟩ _ _ heq
  have haddr := heq.2
  simp only [address] at haddr
  have hi := i.isLt
  rw [Nat.add_mul, Nat.one_mul] at ht
  omega

theorem preserves_input (T BT row : Nat) (s o : RegionName) (c : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) (hSO : s ≠ o) (off : Nat) :
    (scatter T BT row o c f acc u).readMem s off = u.readMem s off := by
  unfold scatter
  exact BlockState.scatter_prop_masked_preserves_other_region o
    (fun idx : TileIndex [BT] => address T row (c * BT + idx.1.val))
    (fun idx => storedValue BT c f acc idx.1)
    (fun idx => c * BT + idx.1.val < T) s hSO off _ u

#axiomsClean store_eval
#axiomsClean frame
#axiomsClean cell
#axiomsClean preserves_later
#axiomsClean preserves_input
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealMemory
