import bench.tritonbench_g.int4_matmul.Proofs.FPTypedSource
namespace VeriTile.Bench.TritonBenchG.Int4Matmul.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

structure Parameters where
  (input output scale weight zero : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride scaleGroupStride scaleColumnStride
    zeroGroupStride zeroColumnStride groupSize blockRows blockColumns blockInner
    groupRows steps : Nat)

def source (p : Parameters) : ComputeKernel := typed .fp16 p.input p.output p.scale p.weight p.zero
  p.rows p.columns p.inner p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleGroupStride p.scaleColumnStride
  p.zeroGroupStride p.zeroColumnStride p.groupSize p.blockRows p.blockColumns p.blockInner p.groupRows p.steps

def prelude (p : Parameters) := (source p).surfaceBody.take 17

def body (p : Parameters) : List ComputeStmt :=
  match (source p).surfaceBody[17]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []

def epilogue (p : Parameters) := (source p).surfaceBody.drop 18

abbrev AIndex (p : Parameters) := TileIndex [p.blockRows, p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner, p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows, p.blockColumns]

def blocksM (p : Parameters) : Nat := (p.rows + p.blockRows - 1) / p.blockRows
def blocksN (p : Parameters) : Nat := (p.columns + p.blockColumns - 1) / p.blockColumns

def firstRow {α : Type} (p : Parameters) (s : State α) : Nat :=
  (s.pids 0 / (p.groupRows * blocksN p)) * p.groupRows

def rowsInGroup {α : Type} (p : Parameters) (s : State α) : Nat :=
  min (blocksM p - firstRow p s) p.groupRows

def rowBlock {α : Type} (p : Parameters) (s : State α) : Nat :=
  firstRow p s + s.pids 0 % rowsInGroup p s

def columnBlock {α : Type} (p : Parameters) (s : State α) : Nat :=
  (s.pids 0 % (p.groupRows * blocksN p)) / rowsInGroup p s

def inputRow {α : Type} (p : Parameters) (s : State α) (i : Fin p.blockRows) : Nat :=
  (rowBlock p s * p.blockRows + i.val) % p.rows

def inputColumn {α : Type} (p : Parameters) (s : State α) (j : Fin p.blockColumns) : Nat :=
  (columnBlock p s * p.blockColumns + j.val) % p.columns

def inputInner {α : Type} (p : Parameters) (s : State α) (j : Fin p.blockInner) : Nat :=
  s.pids 1 * p.blockInner + j.val

/-- Pointer increments follow the actual source, without replacing integer
quotients by a mathematically more convenient divisible-block formula. -/
def aOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : AIndex p) : Nat :=
  inputRow p s i.1 * p.inputRowStride + inputInner p s i.2.1 * p.inputInnerStride +
    step * (p.blockInner * p.inputInnerStride)

def bOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) : Nat :=
  inputInner p s i.1 / 8 * p.weightInnerStride + inputColumn p s i.2.1 * p.weightColumnStride +
    step * (p.blockInner * p.weightInnerStride / 8)

def scaleOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) : Nat :=
  (inputInner p s i.1 + step * p.blockInner) / p.groupSize * p.scaleGroupStride +
    inputColumn p s i.2.1 * p.scaleColumnStride

def zeroOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) : Nat :=
  (inputInner p s i.1 + step * p.blockInner) / p.groupSize * p.zeroGroupStride +
    inputColumn p s i.2.1 / 8 * p.zeroColumnStride

def outputOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Nat :=
  p.outputRowStride * (rowBlock p s * p.blockRows + i.1.val) +
    p.outputColumnStride * (columnBlock p s * p.blockColumns + i.2.1.val)

def active {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Prop :=
  rowBlock p s * p.blockRows + i.1.val < p.rows ∧
    columnBlock p s * p.blockColumns + i.2.1.val < p.columns
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

def difference {α : Type} [Inhabited α] (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) : Int :=
  MatmulDequantInt4.FPTyped.nibble ((s.mem p.weight (bOffset p s step i)).read .int) (inputInner p s i.1) -
    MatmulDequantInt4.FPTyped.nibble ((s.mem p.zero (zeroOffset p s step i)).read .int) (inputColumn p s i.2.1)

/-- A supplied dot implementation retains precision, options and layout. Its
native binding is separate; no real sum, associativity or reduction order is
assumed by the structural loop proof. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α)

def backend {α : Type} (rest : Algebra α) (D : Dot α) : Algebra α :=
  { rest with dotTyped := fun precision left right out options {batch} {m} {k} {n} a b =>
      match left, right, out with
      | .fp16, .fp16, .fp32 => some (D.apply precision options a b)
      | left, right, out => rest.dotTyped precision left right out options a b }

def aValue {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (step : Nat) (i : AIndex p) : α := (s.mem p.input (aOffset p s step i)).read .fp16

def bValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (step : Nat) (i : BIndex p) : α :=
  M.cast none .fp16 .fp16
    (M.binary (some .fp32) .fp16 .mul
      (M.cast (some .fp32) .real .fp16 (M.fromInt (some .fp32) (difference p s step i)))
      ((s.mem p.scale (scaleOffset p s step i)).read .fp16))

/-- An independent initial-memory recurrence: one native dot and one fp32
addition per actual K block, followed by the source's destination cast. -/
def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) : Nat → CIndex p → α
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | step + 1 => fun i =>
      M.binary (some .fp32) .real .add (accumulation M D p s step i)
        (M.cast (some .fp32) .fp32 .real
          (D.apply (some .fp32) {} (batch := []) (aValue p s step)
            (fun j => M.cast (some .fp32) .fp16 .fp16 (bValue M p s step j)) i))

def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) (i : CIndex p) : α :=
  M.cast none .real .fp16 (accumulation M D p s p.steps i)

/-- Signed integer facts needed by the native int32 subtraction and integer
promotion, independent of any floating or dot callback. -/
theorem difference_bounds {α : Type} [Inhabited α]
    (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) :
    -15 ≤ difference p s step i ∧ difference p s step i ≤ 15 := by
  have ha := MatmulDequantInt4.FPBits.mask_bound (BitVec.ofInt 32
    (((s.mem p.weight (bOffset p s step i)).read .int) >>> (inputInner p s i.1 % 8 * 4)))
  have hb := MatmulDequantInt4.FPBits.mask_bound (BitVec.ofInt 32
    (((s.mem p.zero (zeroOffset p s step i)).read .int) >>> (inputColumn p s i.2.1 % 8 * 4)))
  change 0 ≤ MatmulDequantInt4.FPTyped.nibble _ _ ∧ MatmulDequantInt4.FPTyped.nibble _ _ ≤ 15 at ha
  change 0 ≤ MatmulDequantInt4.FPTyped.nibble _ _ ∧ MatmulDequantInt4.FPTyped.nibble _ _ ≤ 15 at hb
  dsimp only [difference]
  omega
/-- Initial machine-storage representation, for every load in the complete
accepted K loop. It assumes input bits, never a computed output. -/
def PackedInputs {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Prop :=
  ∀ step, step < p.steps → ∀ i : BIndex p,
    (∃ w : BitVec 32, s.mem p.weight (bOffset p s step i) = .mk .int w.toInt) ∧
    (∃ z : BitVec 32, s.mem p.zero (zeroOffset p s step i) = .mk .int z.toInt)

def packedDifference {α : Type} (p : Parameters) (s : State α) (i : BIndex p)
    (word zero : BitVec 32) : Int :=
  ((word >>> (inputInner p s i.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt -
    ((zero >>> (inputColumn p s i.2.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt

theorem packed_difference_bridge {α : Type} [Inhabited α]
    (p : Parameters) (s : State α) (h : PackedInputs p s) (step : Nat) (hs : step < p.steps)
    (i : BIndex p) : ∃ w z : BitVec 32,
      s.mem p.weight (bOffset p s step i) = .mk .int w.toInt ∧
      s.mem p.zero (zeroOffset p s step i) = .mk .int z.toInt ∧
      difference p s step i = packedDifference p s i w z := by
  obtain ⟨⟨w, hw⟩, ⟨z, hz⟩⟩ := h step hs i
  refine ⟨w, z, hw, hz, ?_⟩
  simp [difference, packedDifference, hw, hz, MatmulDequantInt4.FPTyped.nibble_packed_word]

theorem difference_int32_exact {α : Type} [Inhabited α]
    (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) :
    (BitVec.ofInt 32 (difference p s step i)).toInt = difference p s step i := by
  have h := difference_bounds p s step i
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega

#axiomsClean packed_difference_bridge
#axiomsClean difference_int32_exact
#axiomsClean difference_bounds
end
end VeriTile.Bench.TritonBenchG.Int4Matmul.FPTyped
