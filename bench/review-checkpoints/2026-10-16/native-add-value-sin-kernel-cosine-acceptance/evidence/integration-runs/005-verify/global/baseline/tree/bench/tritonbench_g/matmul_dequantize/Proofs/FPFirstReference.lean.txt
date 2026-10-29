import bench.tritonbench_g.matmul_dequantize.Proofs.FPTypedSource
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

structure Parameters where
  (input output scale weight zero : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride scaleGroupStride scaleColumnStride
    zeroGroupStride zeroColumnStride groupSize blockRows blockColumns blockInner
    groupRows : Nat)
  noGroups : Bool

def source (p : Parameters) : ComputeKernel := FPTyped.matmul4Typed .fp16 p.input p.output p.scale p.weight p.zero
  p.rows p.columns p.inner p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleGroupStride p.scaleColumnStride
  p.zeroGroupStride p.zeroColumnStride p.groupSize p.noGroups p.blockRows p.blockColumns p.blockInner p.groupRows

def prelude (p : Parameters) := (source p).surfaceBody.take 24

def body (p : Parameters) : List ComputeStmt :=
  match (source p).surfaceBody[24]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []

def epilogue (p : Parameters) := (source p).surfaceBody.drop 25

abbrev AIndex (p : Parameters) := TileIndex [p.blockRows, p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner, p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows, p.blockColumns]

def steps (p : Parameters) : Nat := (p.inner + p.blockInner - 1) / p.blockInner

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
  rowBlock p s * p.blockRows + i.val

def inputColumn {α : Type} (p : Parameters) (s : State α) (j : Fin p.blockColumns) : Nat :=
  columnBlock p s * p.blockColumns + j.val

def inputInner {α : Type} (p : Parameters) (s : State α) (j : Fin p.blockInner) : Nat :=
  j.val

/-- Pointer increments follow the actual source, without replacing integer
quotients by a mathematically more convenient divisible-block formula. -/
def aOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : AIndex p) : Nat :=
  inputRow p s i.1 * p.inputRowStride + inputInner p s i.2.1 * p.inputInnerStride +
    step * (p.blockInner * p.inputInnerStride)

def bOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) : Nat :=
  inputInner p s i.1 / 8 * p.weightInnerStride + inputColumn p s i.2.1 * p.weightColumnStride +
    step * (p.blockInner / 8 * p.weightInnerStride)

def groupIndex (p : Parameters) (step : Nat) : Nat :=
  if p.noGroups then 0 else step / (p.groupSize / p.blockInner)

def scaleOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : TileIndex [p.blockColumns]) : Nat :=
  inputColumn p s i.1 * p.scaleColumnStride + groupIndex p step * p.scaleGroupStride

def zeroOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : TileIndex [p.blockColumns]) : Nat :=
  inputColumn p s i.1 / 8 * p.zeroColumnStride + groupIndex p step * p.zeroGroupStride

def outputOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Nat :=
  p.outputRowStride * (rowBlock p s * p.blockRows + i.1.val) +
    p.outputColumnStride * (columnBlock p s * p.blockColumns + i.2.1.val)

def active {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Prop :=
  rowBlock p s * p.blockRows + i.1.val < p.rows ∧
    columnBlock p s * p.blockColumns + i.2.1.val < p.columns
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

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

def aValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (step : Nat) (i : AIndex p) : α :=
  if inputRow p s i.1 < p.rows then (s.mem p.input (aOffset p s step i)).read .fp16
  else M.cast none .real .fp16 (M.literal none .real 0)

def weightNibble {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (step : Nat) (i : BIndex p) : Int :=
  MatmulDequantInt4.FPTyped.nibble ((s.mem p.weight (bOffset p s step i)).read .int) i.1.val

def zeroNibble {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (step : Nat) (j : TileIndex [p.blockColumns]) : Int :=
  MatmulDequantInt4.FPTyped.nibble ((s.mem p.zero (zeroOffset p s step j)).read .int) (inputColumn p s j.1)

def scaleValue {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (step : Nat) (j : TileIndex [p.blockColumns]) : α :=
  (s.mem p.scale (scaleOffset p s step j)).read .fp16

def zeroValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (step : Nat) (j : TileIndex [p.blockColumns]) : α :=
  M.binary none .fp16 .mul
    (M.cast (some .fp32) .real .fp16 (M.fromInt (some .fp32) (zeroNibble p s step j)))
    (scaleValue p s step j)

def bValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (step : Nat) (i : BIndex p) : α :=
  M.binary none .fp16 .sub
    (M.binary (some .fp32) .fp16 .mul
      (M.cast (some .fp32) .real .fp16 (M.fromInt (some .fp32) (weightNibble p s step i)))
      (scaleValue p s step (i.2.1, PUnit.unit)))
    (zeroValue M p s step (i.2.1, PUnit.unit))

/-- An independent initial-memory recurrence: one native dot and one fp32
addition per actual K block, followed by the source's destination cast. -/
def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) : Nat → CIndex p → α
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | step + 1 => fun i =>
      M.binary (some .fp32) .real .add (accumulation M D p s step i)
        (M.cast (some .fp32) .fp32 .real
          (D.apply (some .fp32) {} (batch := []) (aValue M p s step)
            (bValue M p s step) i))

def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) (i : CIndex p) : α :=
  M.cast none .real .fp16 (accumulation M D p s (steps p) i)

theorem weightNibble_bounds {α : Type} [Inhabited α]
    (p : Parameters) (s : State α) (step : Nat) (i : BIndex p) :
    0 ≤ weightNibble p s step i ∧ weightNibble p s step i ≤ 15 := by
  exact MatmulDequantInt4.FPBits.mask_bound _

theorem zeroNibble_bounds {α : Type} [Inhabited α]
    (p : Parameters) (s : State α) (step : Nat) (i : TileIndex [p.blockColumns]) :
    0 ≤ zeroNibble p s step i ∧ zeroNibble p s step i ≤ 15 := by
  exact MatmulDequantInt4.FPBits.mask_bound _

/-- Every actual packed load has a signed int32 storage representation. -/
def PackedInputs {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Prop :=
  ∀ step, step < steps p → ∀ i : BIndex p,
    (∃ w : BitVec 32, s.mem p.weight (bOffset p s step i) = .mk .int w.toInt) ∧
    (∃ z : BitVec 32, s.mem p.zero (zeroOffset p s step (i.2.1, PUnit.unit)) = .mk .int z.toInt)

theorem packed_nibbles_bridge {α : Type} [Inhabited α]
    (p : Parameters) (s : State α) (h : PackedInputs p s) (step : Nat) (hs : step < steps p)
    (i : BIndex p) : ∃ w z : BitVec 32,
      s.mem p.weight (bOffset p s step i) = .mk .int w.toInt ∧
      s.mem p.zero (zeroOffset p s step (i.2.1, PUnit.unit)) = .mk .int z.toInt ∧
      weightNibble p s step i = ((w >>> (i.1.val % 8 * 4)) &&& (15 : BitVec 32)).toInt ∧
      zeroNibble p s step (i.2.1, PUnit.unit) =
        ((z >>> (inputColumn p s i.2.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt := by
  obtain ⟨⟨w, hw⟩, ⟨z, hz⟩⟩ := h step hs i
  refine ⟨w, z, hw, hz, ?_, ?_⟩
  · simp [weightNibble, hw, MatmulDequantInt4.FPTyped.nibble_packed_word]
  · simp [zeroNibble, hz, MatmulDequantInt4.FPTyped.nibble_packed_word]

#axiomsClean packed_nibbles_bridge
#axiomsClean weightNibble_bounds
#axiomsClean zeroNibble_bounds
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst
