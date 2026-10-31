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

end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealLoop
