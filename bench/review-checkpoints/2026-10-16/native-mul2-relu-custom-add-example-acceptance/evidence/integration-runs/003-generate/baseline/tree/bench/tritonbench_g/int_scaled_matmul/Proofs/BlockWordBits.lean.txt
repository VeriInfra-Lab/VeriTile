import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordComplete
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000

/-- Physical byte cells are required only for in-bounds loads. The arbitrary
out-of-bounds oracle is separately interpreted as an eight-bit payload. -/
def ByteStorage {α : Type} {shape : TileShape} (s : State α)
    (ptr : BlockPtr) (i : TileIndex shape) : Prop :=
  ptr.inBounds (TileShape.indexToList shape i) [0,1] = Bool.true →
    ∃ w : BitVec 8, s.mem ptr.region (ptr.address (TileShape.indexToList shape i)) = .mk .int w.toInt

def ByteInputs {α : Type} (p : Parameters) (s : State α) : Prop :=
  (∀ k, k < steps p → ∀ i : AIndex p, ByteStorage s (aPointer p s k) i) ∧
  (∀ k, k < steps p → ∀ i : BIndex p, ByteStorage s (bPointer p s k) i)

def loadedBits {α : Type} [Inhabited α] {shape : TileShape}
    (s : State α) (ptr : BlockPtr) (i : TileIndex shape) : BitVec 8 :=
  let ix := TileShape.indexToList shape i
  if ptr.inBounds ix [0,1] then BitVec.ofInt 8 ((s.mem ptr.region (ptr.address ix)).read .int)
  else BitVec.ofInt 8 (s.undef .int ptr.region (ptr.address ix))

theorem loaded_bits {α : Type} [Inhabited α] {shape : TileShape}
    (s : State α) (ptr : BlockPtr) (i : TileIndex shape) :
    loaded s ptr i = (loadedBits s ptr i).toInt := by
  simp only [loaded, loadedBits, wrap]
  split <;> rfl

theorem in_bounds_byte {α : Type} [Inhabited α] {shape : TileShape}
    (s : State α) (ptr : BlockPtr) (i : TileIndex shape)
    (h : ByteStorage s ptr i) (hi : ptr.inBounds (TileShape.indexToList shape i) [0,1] = Bool.true) :
    ∃ w : BitVec 8, s.mem ptr.region (ptr.address (TileShape.indexToList shape i)) = .mk .int w.toInt ∧
      loaded s ptr i = w.toInt := by
  obtain ⟨w, hw⟩ := h hi
  exact ⟨w, hw, by simp [loaded, hi, hw, wrap]⟩

theorem outside_byte {α : Type} [Inhabited α] {shape : TileShape}
    (s : State α) (ptr : BlockPtr) (i : TileIndex shape)
    (hi : ptr.inBounds (TileShape.indexToList shape i) [0,1] = Bool.false) :
    loaded s ptr i = (BitVec.ofInt 8 (s.undef .int ptr.region
      (ptr.address (TileShape.indexToList shape i)))).toInt := by
  simp [loaded, hi, wrap]

theorem dot_operand_ranges {α : Type} [Inhabited α] (p : Parameters) (s : State α) (k : Nat) :
    FP.IntegerDot.valid (α := α) (aValue p s k) ∧ FP.IntegerDot.valid (α := α) (bValue p s k) := by
  constructor <;> intro i
  all_goals constructor
  all_goals first
    | exact BitVec.le_toInt _
    | exact BitVec.toInt_lt

theorem accumulator_bits {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (k : Nat) (i : CIndex p) : ∃ w : BitVec 32, accumulation p s k i = w.toInt := by
  cases k with
  | zero => exact ⟨0, rfl⟩
  | succ k => exact ⟨_, rfl⟩

theorem accumulator_wrap_shape (p : Parameters) :
    (body p)[2]? = some (.assign .int [p.blockRows,p.blockColumns] "accumulator"
      (.alg (.intToSigned 32 (.add .int (.consSame (.consSame .nil))
        (.ref .int [p.blockRows,p.blockColumns] "accumulator")
        (Op.dotInt (batch := []) (.ref .int [p.blockRows,p.blockInner] "a")
          (.ref .int [p.blockInner,p.blockColumns] "b")))))) := by rfl

theorem wrap_overflow : wrap 32 (2147483647 + 1) = -2147483648 := by decide +kernel

#axiomsClean loaded_bits
#axiomsClean in_bounds_byte
#axiomsClean outside_byte
#axiomsClean dot_operand_ranges
#axiomsClean accumulator_bits
#axiomsClean accumulator_wrap_shape
#axiomsClean wrap_overflow
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord
