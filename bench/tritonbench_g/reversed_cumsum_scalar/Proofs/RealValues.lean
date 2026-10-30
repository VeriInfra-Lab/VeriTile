import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Scan
import VeriTile.Triton.Semantics.BlockPadding

/-! Real evaluation of the load, reduction, inclusive scan and output formula
of one source chunk. The incoming carry is arbitrary and the boundary load
uses the state's unspecified-value oracle exactly as the source requests. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def address (T row t : Nat) : Nat := row * T + t

def value (T row : Nat) (input padding : Nat → ℝ) (t : Nat) : ℝ :=
  if t < T then input (address T row t) else padding (address T row t)

def pointer (T BT row : Nat) (r : RegionName) (c : Nat) : Tile .blockPtr [BT] :=
  ⟨fun _ =>
    { region := r, baseOffset := row * T, parentShape := [T],
      blockShape := [BT], strides := [1], offsets := [c * BT] }⟩

def inputTile (BT c : Nat) (f : Nat → ℝ) : Tile .real [BT] :=
  Tile.vec fun i => some (f (c * BT + i.val))

def chunkSum (BT c : Nat) (f : Nat → ℝ) : ℝ := ∑ i : Fin BT, f (c * BT + i.val)

def chunkPrefix (BT c : Nat) (f : Nat → ℝ) (i : Fin BT) : ℝ :=
  ∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val), f (c * BT + k.val)

def outputTile (BT c : Nat) (f : Nat → ℝ) (updatedCarry : ℝ) : Tile .real [BT] :=
  Tile.vec fun i => some (f (c * BT + i.val) - chunkPrefix BT c f i + updatedCarry)

def outputOp (BT : Nat) : Op .real [BT] :=
  .add .real (.consR .nil)
    (.sub .real (.consSame .nil) (.ref .real [BT] "b_s")
      (.scan .sum ⟨0, by simp⟩ .forward (.ref .real [BT] "b_s")))
    (.expandDim ⟨0, by simp⟩ (.ref .real [] "b_z"))

theorem load_eval (T BT row : Nat) (r : RegionName) (c : Nat)
    (input padding : Nat → ℝ) (u : BlockState)
    (hp : u.regs .blockPtr [BT] "p_s" = some (pointer T BT row r c))
    (hm : ∀ off, u.readMem r off = input off)
    (hu : ∀ off, u.undef r off = padding off) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [BT] "p_s") [0]) .none .unspecified) u =
      some (inputTile BT c (value T row input padding)) := by
  simp only [evalOp, evalOp_ref, hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [pointer, inputTile, Tile.vec, value, address,
    TileShape.indexToList, BlockPtr.inBounds_1d, BlockPtr.address_1d,
    BlockState.readMemValue_real, Nat.mul_one, decide_eq_true_eq]
  by_cases hi : c * BT + i.val < T <;>
    simp only [hi, hm, ite_true, ite_false, BlockState.paddingValue,
      BlockState.unspecifiedValue, pointer, hu]

theorem scan_value (BT c : Nat) (f : Nat → ℝ) (i : Fin BT) :
    (Tile.scan .sum ⟨0, by simp⟩ .forward (inputTile BT c f)).data (i, PUnit.unit) =
      some (chunkPrefix BT c f i) :=
  Scan.inclusive_sum BT (fun k => f (c * BT + k.val)) i

theorem sum_value (BT c : Nat) (f : Nat → ℝ) :
    (Tile.reduceSum (⟨0, by simp⟩ : Fin [BT].length) Bool.false (inputTile BT c f)).data PUnit.unit =
      some (chunkSum BT c f) := by
  rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
  change (∑ k : Fin BT, ((f (c * BT + k.val) : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

theorem sum_eval (BT c : Nat) (f : Nat → ℝ) (u : BlockState)
    (hb : u.regs .real [BT] "b_s" = some (inputTile BT c f)) :
    evalOp (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [BT] "b_s")) u =
      some (Tile.scalar (some (chunkSum BT c f))) := by
  rw [evalOp_reduceSum, evalOp_ref, hb]
  simp only [Option.bind_some, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨⟩
  exact sum_value BT c f

theorem carry_eval (BT c : Nat) (f : Nat → ℝ) (acc : ℝ) (u : BlockState)
    (ha : u.regs .real [] "b_z" = some (Tile.scalar (some acc)))
    (hb : u.regs .real [] "b_zz" = some (Tile.scalar (some (chunkSum BT c f)))) :
    evalOp (.add .real .nil (.ref .real [] "b_z") (.ref .real [] "b_zz")) u =
      some (Tile.scalar (some (acc + chunkSum BT c f))) := by
  rw [evalOp_add, evalOp_ref, ha, evalOp_ref, hb]
  rfl

theorem output_eval (BT c : Nat) (f : Nat → ℝ) (acc : ℝ) (u : BlockState)
    (ha : u.regs .real [] "b_z" = some (Tile.scalar (some acc)))
    (hb : u.regs .real [BT] "b_s" = some (inputTile BT c f)) :
    evalOp (outputOp BT) u = some (outputTile BT c f acc) := by
  have hs : evalOp (.scan .sum ⟨0, by simp⟩ .forward (.ref .real [BT] "b_s")) u =
      some (Tile.scan .sum ⟨0, by simp⟩ .forward (inputTile BT c f)) := by
    simp only [evalOp, hb]
    rfl
  rw [outputOp, evalOp_add, evalOp_sub, evalOp_ref, hb, hs]
  erw [evalOp_expandDim, evalOp_ref, ha]
  simp only [Option.bind_some, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim,
    TileShape.dropInsertedIndex, Tile.scalar_data, Tile.vec, NumericDType.add,
    NumericDType.sub, outputTile, inputTile, Option.map₂_some_some]
  erw [scan_value BT c f i]
  rfl

/-- The actual source expression connects to the independent padded suffix
once the reverse-loop invariant supplies the updated carry. -/
theorem output_formula (BT chunks c : Nat) (f : Nat → ℝ) :
    outputTile BT c f (Formula.carry f BT chunks c) =
      Tile.vec (fun i => some (Formula.suffix f (chunks * BT) (c * BT + i.val))) := by
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  exact congrArg some (Formula.scalar_from_chunk f BT chunks c i)

#axiomsClean load_eval
#axiomsClean sum_eval
#axiomsClean carry_eval
#axiomsClean output_eval
#axiomsClean output_formula
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues
