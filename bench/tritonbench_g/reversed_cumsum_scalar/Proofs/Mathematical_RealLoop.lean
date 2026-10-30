import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Mathematical_RealStep
import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_SignedRange

/-! A complete reverse-loop invariant. The carry is derived from initial
input and unspecified values; every completed output has its full real cell
tag, and all cells outside the row's output footprint are preserved. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealLoop
open VeriTile.Triton RealValues RealProgram
noncomputable section
set_option maxHeartbeats 2400000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def accumulated (T BT row : Nat) (input padding : Nat → ℝ) (c : Nat) : ℝ :=
  Formula.carry (value T row input padding) BT (numChunks T BT) c

def expected (T BT row : Nat) (input padding : Nat → ℝ) (t : Nat) : ℝ :=
  Formula.suffix (value T row input padding) (numChunks T BT * BT) t

structure Holds (initial : BlockState) (T BT row : Nat) (s o : RegionName)
    (a : Nat → ℝ) (remaining : Nat) (u : BlockState) : Prop where
  undef : u.undef = initial.undef
  count_le : remaining ≤ numChunks T BT
  head : u.regs .nat [] "i_bh" = some (Tile.scalar row)
  carry : u.regs .real [] "b_z" =
    some (Tile.scalar (some (accumulated T BT row a (initial.undef s) remaining)))
  input : ∀ off, u.readMem s off = a off
  output : ∀ t, t < T → remaining * BT ≤ t →
    u.mem o (address T row t) = MemCell.real (expected T BT row a (initial.undef s) t)
  frame : BlockState.WriteWithin (RealMemory.footprint T row o) initial u

private theorem fold_undef {ι : Type} (items : List ι) (o : RegionName)
    (offset : ι → Nat) (value : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (u : BlockState) :
    (items.foldl (fun t i => if active i then t.writeMem o (offset i) (value i) else t) u).undef =
      u.undef := by
  induction items generalizing u with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : active i <;>
      simp only [List.foldl_cons, hi, ite_true, ite_false, ih, BlockState.writeMem_undef]
    all_goals rfl

theorem iteration (initial : BlockState) (swapped : Bool) (T BT row : Nat)
    (s o : RegionName) (input : Nat → ℝ) (hSO : s ≠ o) (j : Int) (u : BlockState)
    (hj : j < 1) (hr : u.regs .int [] "__rev_i_t" = some (Tile.scalar j))
    (h : Holds initial T BT row s o input ((-j).toNat + 1) u) :
    ∃ v, stepStmts (loopBody swapped T BT s o) u = some v ∧
      Holds initial T BT row s o input (-j).toNat v := by
  let c := (-j).toNat
  let f := value T row input (initial.undef s)
  let acc := accumulated T BT row input (initial.undef s) (c + 1)
  let pre := RealStep.beforeStore T BT row s o c f acc u
  let v := RealStep.result T BT row s o c f acc u
  have hnext : acc + chunkSum BT c f = accumulated T BT row input (initial.undef s) c := by
    exact Formula.carry_predecessor f BT (numChunks T BT) c
  have he := RealStep.executes swapped T BT row s o input (initial.undef s) acc j u hj hr
    h.head h.carry h.input (by intro off; rw [h.undef])
  refine ⟨v, he, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [v, RealStep.result, RealMemory.scatter, fold_undef, RealStep.beforeStore,
      BlockState.setReg_undef, h.undef]
    exact h.undef
  · have := h.count_le
    omega
  · simp [v, RealStep.result, RealMemory.scatter, RealStep.beforeStore, h.head]
  · simp [v, RealStep.result, RealMemory.scatter, RealStep.beforeStore, hnext]
    rfl
  · intro off
    change (RealMemory.scatter T BT row o c f (acc + chunkSum BT c f) pre).readMem s off = _
    rw [RealMemory.preserves_input T BT row s o c f _ pre hSO]
    simpa [pre, RealStep.beforeStore] using h.input off
  · intro t ht hlower
    change (RealMemory.scatter T BT row o c f (acc + chunkSum BT c f) pre).mem o (address T row t) = _
    by_cases hlater : (c + 1) * BT ≤ t
    · rw [RealMemory.preserves_later T BT row o c f _ pre t hlater]
      simpa [pre, RealStep.beforeStore] using h.output t ht hlater
    · have hi : t - c * BT < BT := by
        change c * BT ≤ t at hlower
        rw [Nat.add_mul, Nat.one_mul] at hlater
        omega
      let i : Fin BT := ⟨t - c * BT, hi⟩
      have hti : c * BT + i.val = t := by dsimp [i]; change c * BT ≤ t at hlower; omega
      have hactive : c * BT + i.val < T := by simpa [hti] using ht
      have hread := RealMemory.cell T BT row o c f (acc + chunkSum BT c f) pre i hactive
      rw [hti] at hread
      rw [hread]
      apply congrArg MemCell.real
      change f (c * BT + i.val) - chunkPrefix BT c f i +
        (acc + chunkSum BT c f) = expected T BT row input (initial.undef s) t
      rw [hnext]
      exact (Formula.scalar_from_chunk f BT (numChunks T BT) c i).trans (by rw [hti]; rfl)
  · have hs := RealMemory.frame T BT row o c f (acc + chunkSum BT c f) pre
    intro r off hn
    exact (h.frame r off hn).trans (hs r off hn)

#axiomsClean iteration
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealLoop
